
import datetime
import logging
import math
import os
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Set

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from logger import get_logger
from model.box_ops import fast_diag_box_iou
from model.data_misc import BatchedDatapoint
from model.sam3_tracker_utils import fill_holes_in_mask_scores, mask_to_box
from perflib.masks_ops import mask_iou
from train.masks_ops import rle_encode
from torch import nn, Tensor

logger = get_logger(__name__)

class MaskletConfirmationStatus(Enum):
    UNCONFIRMED = 1
    CONFIRMED = 2

class Sam3VideoBase(nn.Module):
    def __init__(
        self,
        detector: nn.Module,
        tracker: nn.Module,
        score_threshold_detection=0.5,
        det_nms_thresh=0.0,
        assoc_iou_thresh=0.5,
        trk_assoc_iou_thresh=0.5,
        new_det_thresh=0.0,
        hotstart_delay=0,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=3,
        suppress_unmatched_only_within_hotstart=True,
        init_trk_keep_alive=0,
        max_trk_keep_alive=8,
        min_trk_keep_alive=-4,
        suppress_overlapping_based_on_recent_occlusion_threshold=0.0,
        decrease_trk_keep_alive_for_empty_masklets=False,
        o2o_matching_masklets_enable=False,
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        max_num_objects=-1,
        recondition_every_nth_frame=-1,
        masklet_confirmation_enable=False,
        masklet_confirmation_consecutive_det_thresh=3,
        reconstruction_bbox_iou_thresh=0.0,
        reconstruction_bbox_det_score=0.0,
    ):
        super().__init__()
        self.detector = detector
        self.tracker = tracker
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh

        if hotstart_delay > 0:
            assert hotstart_unmatch_thresh <= hotstart_delay
            assert hotstart_dup_thresh <= hotstart_delay
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.suppress_unmatched_only_within_hotstart = (
            suppress_unmatched_only_within_hotstart
        )
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.suppress_det_close_to_boundary = suppress_det_close_to_boundary
        self.decrease_trk_keep_alive_for_empty_masklets = (
            decrease_trk_keep_alive_for_empty_masklets
        )
        self.o2o_matching_masklets_enable = o2o_matching_masklets_enable
        self.fill_hole_area = fill_hole_area
        self.eval()
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self._dist_pg_cpu = None

        if max_num_objects > 0:
            num_obj_for_compile = math.ceil(max_num_objects / self.world_size)
        else:
            max_num_objects = 10000
            num_obj_for_compile = 16
        logger.info(f"setting {max_num_objects=} and {num_obj_for_compile=}")
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.masklet_confirmation_enable = masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = (
            masklet_confirmation_consecutive_det_thresh
        )
        self.reconstruction_bbox_iou_thresh = reconstruction_bbox_iou_thresh
        self.reconstruction_bbox_det_score = reconstruction_bbox_det_score

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def _init_dist_pg_cpu(self):
        timeout_sec = int(os.getenv("SAM3_COLLECTIVE_OP_TIMEOUT_SEC", "180"))
        timeout = datetime.timedelta(seconds=timeout_sec)
        self._dist_pg_cpu = dist.new_group(backend="gloo", timeout=timeout)

    def broadcast_python_obj_cpu(self, python_obj_list, src):
        if self._dist_pg_cpu is None:
            self._init_dist_pg_cpu()
        dist.broadcast_object_list(python_obj_list, src=src, group=self._dist_pg_cpu)

    def _det_track_one_frame(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        tracker_states_local: List[Any],
        tracker_metadata_prev: Dict[str, Any],
        feature_cache: Dict,
        orig_vid_height: int,
        orig_vid_width: int,
        is_image_only: bool = False,
        allow_new_detections: bool = True,
    ):
        """
        This function handles one-step inference for the DenseTracking model in an SPMD manner.
        At a high-level, all GPUs execute the same function calls as if it's done on a single GPU,
        while under the hood, some function calls involve distributed computation based on sharded
        SAM2 states.

        - `input_batch` contains image and other inputs on the entire video; it should be identical across GPUs
        - `tracker_states_local` holds the local masklet information in this GPU shard
        - `tracker_metadata_prev` manages the metadata for SAM2 objects, such as which masklet is hold on which GPUs
          it contains both global and local masklet information
        """

        det_out = self.run_backbone_and_detection(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            feature_cache=feature_cache,
            allow_new_detections=allow_new_detections,
        )

        if tracker_metadata_prev == {}:
            tracker_metadata_prev.update(self._initialize_metadata())
        tracker_low_res_masks_global, tracker_obj_scores_global = (
            self.run_tracker_propagation(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                tracker_states_local=tracker_states_local,
                tracker_metadata_prev=tracker_metadata_prev,
            )
        )

        tracker_update_plan, tracker_metadata_new = (
            self.run_tracker_update_planning_phase(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                det_out=det_out,
                tracker_low_res_masks_global=tracker_low_res_masks_global,
                tracker_obj_scores_global=tracker_obj_scores_global,
                tracker_metadata_prev=tracker_metadata_prev,
                tracker_states_local=tracker_states_local,
                is_image_only=is_image_only,
            )
        )

        reconditioned_obj_ids = tracker_update_plan.get("reconditioned_obj_ids", set())
        det_to_matched_trk_obj_ids = tracker_update_plan.get(
            "det_to_matched_trk_obj_ids", {}
        )

        tracker_states_local_new = self.run_tracker_update_execution_phase(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            det_out=det_out,
            tracker_states_local=tracker_states_local,
            tracker_update_plan=tracker_update_plan,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            feature_cache=feature_cache,
        )

        if self.rank == 0:
            obj_id_to_mask = self.build_outputs(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                det_out=det_out,
                tracker_low_res_masks_global=tracker_low_res_masks_global,
                tracker_obj_scores_global=tracker_obj_scores_global,
                tracker_metadata_prev=tracker_metadata_prev,
                tracker_update_plan=tracker_update_plan,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                reconditioned_obj_ids=reconditioned_obj_ids,
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
            )
            obj_id_to_score = tracker_metadata_new["obj_id_to_score"]
        else:
            obj_id_to_mask, obj_id_to_score = {}, {}
        frame_stats = {
            "num_obj_tracked": np.sum(tracker_metadata_new["num_obj_per_gpu"]),
            "num_obj_dropped": tracker_update_plan["num_obj_dropped_due_to_limit"],
        }
        if tracker_obj_scores_global.shape[0] > 0:
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid().tolist()
            tracker_obj_ids = tracker_metadata_prev["obj_ids_all_gpu"]
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][
                frame_idx
            ].update(dict(zip(tracker_obj_ids, tracker_obj_scores_global)))
        return (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,
        )

    def _suppress_detections_close_to_boundary(self, boxes, margin=0.025):
        """
        Suppress detections too close to image edges (for normalized boxes).

        boxes: (N, 4) in xyxy format, normalized [0,1]
        margin: fraction of image
        """
        x_min, y_min, x_max, y_max = boxes.unbind(-1)
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        keep = (
            (x_c > margin)
            & (x_c < 1.0 - margin)
            & (y_c > margin)
            & (y_c < 1.0 - margin)
        )

        return keep

    def run_backbone_and_detection(
        self,
        frame_idx: int,
        num_frames: int,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        feature_cache: Dict,
        reverse: bool,
        allow_new_detections: bool,
    ):
        text_batch_key = tuple(input_batch.find_text_batch)
        if "text" not in feature_cache or text_batch_key not in feature_cache["text"]:
            text_outputs = self.detector.backbone.forward_text(
                input_batch.find_text_batch, device=self.device
            )
            feature_cache["text"] = {text_batch_key: text_outputs}
        else:
            text_outputs = feature_cache["text"][text_batch_key]

        if "multigpu_buffer" not in feature_cache:
            feature_cache["multigpu_buffer"] = {}

        tracking_bounds = feature_cache.get("tracking_bounds", {})
        max_frame_num_to_track = tracking_bounds.get("max_frame_num_to_track")
        start_frame_idx = tracking_bounds.get("propagate_in_video_start_frame_idx")

        sam3_image_out, _ = self.detector.forward_video_grounding_multigpu(
            backbone_out={
                "img_batch_all_stages": input_batch.img_batch,
                **text_outputs,
            },
            find_inputs=input_batch.find_inputs,
            geometric_prompt=geometric_prompt,
            frame_idx=frame_idx,
            num_frames=num_frames,
            multigpu_buffer=feature_cache["multigpu_buffer"],
            track_in_reverse=reverse,
            return_tracker_backbone_feats=True,
            run_nms=self.det_nms_thresh > 0.0,
            nms_prob_thresh=self.score_threshold_detection,
            nms_iou_thresh=self.det_nms_thresh,
            max_frame_num_to_track=max_frame_num_to_track,
            propagate_in_video_start_frame_idx=start_frame_idx,
        )
        pred_probs = sam3_image_out["pred_logits"].squeeze(-1).sigmoid()
        if not allow_new_detections:
            pred_probs = pred_probs - 1e8
        pred_boxes_xyxy = sam3_image_out["pred_boxes_xyxy"]
        pred_masks = sam3_image_out["pred_masks"]
        pos_pred_idx = torch.where(pred_probs > self.score_threshold_detection)
        det_out = {
            "bbox": pred_boxes_xyxy[pos_pred_idx[0], pos_pred_idx[1]],
            "mask": pred_masks[pos_pred_idx[0], pos_pred_idx[1]],
            "scores": pred_probs[pos_pred_idx[0], pos_pred_idx[1]],
        }

        backbone_cache = {}
        sam_mask_decoder = self.tracker.sam_mask_decoder
        tracker_backbone_fpn = [
            sam_mask_decoder.conv_s0(sam3_image_out["tracker_backbone_fpn_0"]),
            sam_mask_decoder.conv_s1(sam3_image_out["tracker_backbone_fpn_1"]),
            sam3_image_out["tracker_backbone_fpn_2"],
        ]
        tracker_backbone_out = {
            "vision_features": tracker_backbone_fpn[-1],
            "vision_pos_enc": sam3_image_out["tracker_backbone_pos_enc"],
            "backbone_fpn": tracker_backbone_fpn,
        }
        backbone_cache["tracker_backbone_out"] = tracker_backbone_out
        feature_cache[frame_idx] = (
            input_batch.img_batch[frame_idx],
            backbone_cache,
        )
        feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)
        return det_out

    def run_tracker_propagation(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        tracker_states_local: List[Any],
        tracker_metadata_prev: Dict[str, npt.NDArray],
    ):
        obj_ids_local, low_res_masks_local, obj_scores_local = (
            self._propogate_tracker_one_frame_local_gpu(
                tracker_states_local, frame_idx=frame_idx, reverse=reverse
            )
        )

        assert np.all(
            obj_ids_local == tracker_metadata_prev["obj_ids_per_gpu"][self.rank]
        ), "{} != {}".format(
            obj_ids_local, tracker_metadata_prev["obj_ids_per_gpu"][self.rank]
        )

        _, H_mask, W_mask = low_res_masks_local.shape
        if self.world_size > 1:
            low_res_masks_local = low_res_masks_local.float().contiguous()
            obj_scores_local = obj_scores_local.float().contiguous()
            num_obj_this_gpu = tracker_metadata_prev["num_obj_per_gpu"][self.rank]
            assert low_res_masks_local.size(0) == num_obj_this_gpu
            assert obj_scores_local.size(0) == num_obj_this_gpu
            low_res_masks_peers = [
                low_res_masks_local.new_empty(num_obj, H_mask, W_mask)
                for num_obj in tracker_metadata_prev["num_obj_per_gpu"]
            ]
            obj_scores_peers = [
                obj_scores_local.new_empty(num_obj)
                for num_obj in tracker_metadata_prev["num_obj_per_gpu"]
            ]
            dist.all_gather(low_res_masks_peers, low_res_masks_local)
            dist.all_gather(obj_scores_peers, obj_scores_local)
            low_res_masks_global = torch.cat(low_res_masks_peers, dim=0)
            obj_scores_global = torch.cat(obj_scores_peers, dim=0)
        else:
            low_res_masks_global = low_res_masks_local
            obj_scores_global = obj_scores_local
        return low_res_masks_global, obj_scores_global

    def _recondition_masklets(
        self,
        frame_idx,
        det_out: Dict[str, Tensor],
        trk_id_to_max_iou_high_conf_det: List[int],
        tracker_states_local: List[Any],
        tracker_metadata: Dict[str, npt.NDArray],
        tracker_obj_scores_global: Tensor,
    ):
        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            new_mask = det_out["mask"][det_idx : det_idx + 1]
            input_mask_res = self.tracker.input_mask_size
            new_mask_binary = (
                F.interpolate(
                    new_mask.unsqueeze(1),
                    size=(input_mask_res, input_mask_res),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)[0]
                > 0
            )
            HIGH_CONF_THRESH = 0.8
            reconditioned_states_idx = set()
            obj_idx = np.where(tracker_metadata["obj_ids_all_gpu"] == trk_obj_id)[
                0
            ].item()
            obj_score = tracker_obj_scores_global[obj_idx]
            for state_idx, inference_state in enumerate(tracker_states_local):
                if (
                    trk_obj_id in inference_state["obj_ids"]
                    and obj_score > HIGH_CONF_THRESH
                ):
                    logger.debug(
                        f"Adding new mask for track {trk_obj_id} at frame {frame_idx}. Objects {inference_state['obj_ids']} are all reconditioned."
                    )
                    self.tracker.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=trk_obj_id,
                        mask=new_mask_binary,
                    )
                    reconditioned_states_idx.add(state_idx)

            for idx in reconditioned_states_idx:
                self.tracker.propagate_in_video_preflight(
                    tracker_states_local[idx], run_mem_encoder=True
                )
        return tracker_states_local

    def run_tracker_update_planning_phase(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        tracker_low_res_masks_global: Tensor,
        tracker_obj_scores_global: Tensor,
        tracker_metadata_prev: Dict[str, npt.NDArray],
        tracker_states_local: List[Any],
        is_image_only: bool = False,
    ):
        tracker_metadata_new = {
            "obj_ids_per_gpu": deepcopy(tracker_metadata_prev["obj_ids_per_gpu"]),
            "obj_ids_all_gpu": None,
            "num_obj_per_gpu": deepcopy(tracker_metadata_prev["num_obj_per_gpu"]),
            "obj_id_to_score": deepcopy(tracker_metadata_prev["obj_id_to_score"]),
            "obj_id_to_tracker_score_frame_wise": deepcopy(
                tracker_metadata_prev["obj_id_to_tracker_score_frame_wise"]
            ),
            "obj_id_to_last_occluded": {},
            "max_obj_id": deepcopy(tracker_metadata_prev["max_obj_id"]),
        }

        reconditioned_obj_ids = set()

        det_mask_preds: Tensor = det_out["mask"]
        det_scores_np: npt.NDArray = det_out["scores"].float().cpu().numpy()
        det_bbox_xyxy: Tensor = det_out["bbox"]
        if self.rank == 0:
            (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            ) = self._associate_det_trk(
                det_masks=det_mask_preds,
                det_scores_np=det_scores_np,
                trk_masks=tracker_low_res_masks_global,
                trk_obj_ids=tracker_metadata_prev["obj_ids_all_gpu"],
            )
            if self.suppress_det_close_to_boundary:
                keep = self._suppress_detections_close_to_boundary(
                    det_bbox_xyxy[new_det_fa_inds]
                )
                new_det_fa_inds = new_det_fa_inds[keep.cpu().numpy()]

            prev_obj_num = np.sum(tracker_metadata_prev["num_obj_per_gpu"])
            new_det_num = len(new_det_fa_inds)
            num_obj_dropped_due_to_limit = 0
            if not is_image_only and prev_obj_num + new_det_num > self.max_num_objects:
                logger.warning(
                    f"hitting {self.max_num_objects=} with {new_det_num=} and {prev_obj_num=}"
                )
                new_det_num_to_keep = self.max_num_objects - prev_obj_num
                num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
                new_det_fa_inds = self._drop_new_det_with_obj_limit(
                    new_det_fa_inds, det_scores_np, new_det_num_to_keep
                )
                assert len(new_det_fa_inds) == new_det_num_to_keep
                new_det_num = len(new_det_fa_inds)

            new_det_start_obj_id = tracker_metadata_prev["max_obj_id"] + 1
            new_det_obj_ids = new_det_start_obj_id + np.arange(new_det_num)
            prev_workload_per_gpu = tracker_metadata_prev["num_obj_per_gpu"]
            new_det_gpu_ids = self._assign_new_det_to_gpus(
                new_det_num=new_det_num,
                prev_workload_per_gpu=prev_workload_per_gpu,
            )

            rank0_metadata_new = deepcopy(tracker_metadata_prev["rank0_metadata"])
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                obj_ids_newly_removed, rank0_metadata_new = self._process_hotstart(
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    reverse=reverse,
                    det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                    new_det_obj_ids=new_det_obj_ids,
                    empty_trk_obj_ids=empty_trk_obj_ids,
                    unmatched_trk_obj_ids=unmatched_trk_obj_ids,
                    rank0_metadata=rank0_metadata_new,
                    tracker_metadata=tracker_metadata_prev,
                )
            else:
                obj_ids_newly_removed = set()
            tracker_metadata_new["rank0_metadata"] = rank0_metadata_new

        NUM_BROADCAST_ITEMS = 9
        if self.rank == 0 and self.world_size > 1:
            num_obj_per_gpu_on_rank0 = tracker_metadata_prev["num_obj_per_gpu"]
            update_plan = [
                new_det_fa_inds,
                new_det_obj_ids,
                new_det_gpu_ids,
                num_obj_per_gpu_on_rank0,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                obj_ids_newly_removed,
                num_obj_dropped_due_to_limit,
                trk_id_to_max_iou_high_conf_det,
            ]
            assert len(update_plan) == NUM_BROADCAST_ITEMS, (
                f"Manually update NUM_BROADCAST_ITEMS to be: {len(update_plan)}"
            )
            self.broadcast_python_obj_cpu(update_plan, src=0)
        elif self.rank > 0 and self.world_size > 1:
            update_plan = [
                None
            ] * NUM_BROADCAST_ITEMS
            self.broadcast_python_obj_cpu(update_plan, src=0)
            (
                new_det_fa_inds,
                new_det_obj_ids,
                new_det_gpu_ids,
                num_obj_per_gpu_on_rank0,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                obj_ids_newly_removed,
                num_obj_dropped_due_to_limit,
                trk_id_to_max_iou_high_conf_det,
            ) = update_plan
            if not np.all(
                num_obj_per_gpu_on_rank0 == tracker_metadata_prev["num_obj_per_gpu"]
            ):
                raise RuntimeError(
                    f"{self.rank=} received {num_obj_per_gpu_on_rank0=}, which is inconsistent with local record "
                    f"{tracker_metadata_prev['num_obj_per_gpu']=}. There's likely a bug in update planning or execution."
                )

        tracker_update_plan = {
            "new_det_fa_inds": new_det_fa_inds,
            "new_det_obj_ids": new_det_obj_ids,
            "new_det_gpu_ids": new_det_gpu_ids,
            "unmatched_trk_obj_ids": unmatched_trk_obj_ids,
            "det_to_matched_trk_obj_ids": det_to_matched_trk_obj_ids,
            "obj_ids_newly_removed": obj_ids_newly_removed,
            "num_obj_dropped_due_to_limit": num_obj_dropped_due_to_limit,
            "trk_id_to_max_iou_high_conf_det": trk_id_to_max_iou_high_conf_det,
            "reconditioned_obj_ids": reconditioned_obj_ids,
        }

        should_recondition_iou = False

        if (
            self.reconstruction_bbox_iou_thresh > 0
            and len(trk_id_to_max_iou_high_conf_det) > 0
        ):
            for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
                det_box = det_out["bbox"][det_idx]
                det_score = det_out["scores"][det_idx]

                try:
                    trk_idx = list(tracker_metadata_prev["obj_ids_all_gpu"]).index(
                        trk_obj_id
                    )
                except ValueError:
                    continue

                tracker_mask = tracker_low_res_masks_global[trk_idx]
                mask_binary = tracker_mask > 0
                mask_area = mask_binary.sum().item()

                if mask_area == 0:
                    continue

                tracker_box_pixels = (
                    mask_to_box(mask_binary.unsqueeze(0).unsqueeze(0))
                    .squeeze(0)
                    .squeeze(0)
                )
                mask_height, mask_width = tracker_mask.shape[-2:]
                tracker_box_normalized = torch.tensor(
                    [
                        tracker_box_pixels[0] / mask_width,
                        tracker_box_pixels[1] / mask_height,
                        tracker_box_pixels[2] / mask_width,
                        tracker_box_pixels[3] / mask_height,
                    ],
                    device=tracker_box_pixels.device,
                )

                det_box_batch = det_box.unsqueeze(0)
                tracker_box_batch = tracker_box_normalized.unsqueeze(0)
                iou = fast_diag_box_iou(det_box_batch, tracker_box_batch)[0]

                if (
                    iou < self.reconstruction_bbox_iou_thresh
                    and det_score >= self.reconstruction_bbox_det_score
                ):
                    should_recondition_iou = True
                    reconditioned_obj_ids.add(trk_obj_id)

        should_recondition_periodic = (
            self.recondition_every_nth_frame > 0
            and frame_idx % self.recondition_every_nth_frame == 0
            and len(trk_id_to_max_iou_high_conf_det) > 0
        )

        if should_recondition_periodic or should_recondition_iou:
            self._recondition_masklets(
                frame_idx,
                det_out,
                trk_id_to_max_iou_high_conf_det,
                tracker_states_local,
                tracker_metadata_prev,
                tracker_obj_scores_global,
            )

        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                if self.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0:
                    tracker_low_res_masks_global = (
                        self._suppress_overlapping_based_on_recent_occlusion(
                            frame_idx,
                            tracker_low_res_masks_global,
                            tracker_metadata_prev,
                            tracker_metadata_new,
                            obj_ids_newly_removed,
                            reverse,
                        )
                    )

            self._tracker_update_memories(
                tracker_states_local,
                frame_idx,
                tracker_metadata=tracker_metadata_prev,
                low_res_masks=tracker_low_res_masks_global,
            )

        for rank in range(self.world_size):
            new_det_obj_ids_this_gpu = new_det_obj_ids[new_det_gpu_ids == rank]
            updated_obj_ids_this_gpu = tracker_metadata_new["obj_ids_per_gpu"][rank]
            if len(new_det_obj_ids_this_gpu) > 0:
                updated_obj_ids_this_gpu = np.concatenate(
                    [updated_obj_ids_this_gpu, new_det_obj_ids_this_gpu]
                )
            if len(obj_ids_newly_removed) > 0:
                is_removed = np.isin(
                    updated_obj_ids_this_gpu, list(obj_ids_newly_removed)
                )
                updated_obj_ids_this_gpu = updated_obj_ids_this_gpu[~is_removed]
            tracker_metadata_new["obj_ids_per_gpu"][rank] = updated_obj_ids_this_gpu
            tracker_metadata_new["num_obj_per_gpu"][rank] = len(
                updated_obj_ids_this_gpu
            )
        tracker_metadata_new["obj_ids_all_gpu"] = np.concatenate(
            tracker_metadata_new["obj_ids_per_gpu"]
        )
        if len(new_det_obj_ids) > 0:
            tracker_metadata_new["obj_id_to_score"].update(
                zip(new_det_obj_ids, det_scores_np[new_det_fa_inds])
            )
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][
                frame_idx
            ].update(zip(new_det_obj_ids, det_scores_np[new_det_fa_inds]))
            tracker_metadata_new["max_obj_id"] = max(
                tracker_metadata_new["max_obj_id"],
                np.max(new_det_obj_ids),
            )
        for obj_id in obj_ids_newly_removed:
            tracker_metadata_new["obj_id_to_score"][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx][
                obj_id
            ] = -1e4
            tracker_metadata_new["obj_id_to_last_occluded"].pop(obj_id, None)
        assert ("rank0_metadata" in tracker_metadata_new) == (self.rank == 0)
        if self.rank == 0 and self.masklet_confirmation_enable:
            rank0_metadata = self.update_masklet_confirmation_status(
                rank0_metadata=tracker_metadata_new["rank0_metadata"],
                obj_ids_all_gpu_prev=tracker_metadata_prev["obj_ids_all_gpu"],
                obj_ids_all_gpu_updated=tracker_metadata_new["obj_ids_all_gpu"],
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
            )
            tracker_metadata_new["rank0_metadata"] = rank0_metadata

        return tracker_update_plan, tracker_metadata_new

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        tracker_low_res_masks_global: Tensor,
        tracker_metadata_prev: Dict[str, Any],
        tracker_metadata_new: Dict[str, Any],
        obj_ids_newly_removed: Set[int],
        reverse: bool = False,
    ):
        """
        Suppress overlapping masks based on the most recent occlusion information. If an object is removed by hotstart, we always suppress it if it overlaps with any other object.
        Args:
            frame_idx (int): The current frame index.
            tracker_low_res_masks_global (Tensor): The low-resolution masks for the current frame.
            tracker_metadata_prev (Dict[str, Any]): The metadata from the previous frame.
            tracker_metadata_new (Dict[str, Any]): The metadata for the current frame.
            obj_ids_newly_removed (Set[int]): The object IDs that have been removed.
        Return:
            Tensor: The updated low-resolution masks with some objects suppressed.
        """
        obj_ids_global = tracker_metadata_prev["obj_ids_all_gpu"]
        binary_tracker_low_res_masks_global = tracker_low_res_masks_global > 0
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            assert len(obj_ids_global) == batch_size, (
                f"Mismatch in number of objects: {len(obj_ids_global)} vs {batch_size}"
            )
            NEVER_OCCLUDED = -1
            ALWAYS_OCCLUDED = 100000
            last_occluded_prev = torch.cat(
                [
                    tracker_metadata_prev["obj_id_to_last_occluded"].get(
                        obj_id,
                        torch.full(
                            (1,),
                            fill_value=(
                                NEVER_OCCLUDED
                                if obj_id not in obj_ids_newly_removed
                                else ALWAYS_OCCLUDED
                            ),
                            device=binary_tracker_low_res_masks_global.device,
                            dtype=torch.long,
                        ),
                    )
                    for obj_id in obj_ids_global
                ],
                dim=0,
            )
            to_suppress = self._get_objects_to_suppress_based_on_most_recently_occluded(
                binary_tracker_low_res_masks_global,
                last_occluded_prev,
                obj_ids_global,
                frame_idx,
                reverse,
            )

            is_obj_occluded = ~(binary_tracker_low_res_masks_global.any(dim=(-1, -2)))
            is_obj_occluded_or_suppressed = is_obj_occluded | to_suppress
            last_occluded_new = last_occluded_prev.clone()
            last_occluded_new[is_obj_occluded_or_suppressed] = frame_idx
            tracker_metadata_new["obj_id_to_last_occluded"] = {
                obj_id: last_occluded_new[obj_idx : obj_idx + 1]
                for obj_idx, obj_id in enumerate(obj_ids_global)
            }

            NO_OBJ_LOGIT = -10
            tracker_low_res_masks_global[to_suppress] = NO_OBJ_LOGIT

        return tracker_low_res_masks_global

    def run_tracker_update_execution_phase(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        tracker_states_local: List[Any],
        tracker_update_plan: Dict[str, npt.NDArray],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: Dict,
    ):
        new_det_fa_inds: npt.NDArray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: npt.NDArray = tracker_update_plan["new_det_obj_ids"]
        new_det_gpu_ids: npt.NDArray = tracker_update_plan["new_det_gpu_ids"]
        is_on_this_gpu: npt.NDArray = new_det_gpu_ids == self.rank
        new_det_obj_ids_local: npt.NDArray = new_det_obj_ids[is_on_this_gpu]
        new_det_fa_inds_local: npt.NDArray = new_det_fa_inds[is_on_this_gpu]
        obj_ids_newly_removed: Set[int] = tracker_update_plan["obj_ids_newly_removed"]

        if len(new_det_fa_inds_local) > 0:
            new_det_fa_inds_local_t = torch.from_numpy(new_det_fa_inds_local)
            new_det_masks: Tensor = det_out["mask"][new_det_fa_inds_local_t]
            tracker_states_local = self._tracker_add_new_objects(
                frame_idx=frame_idx,
                num_frames=num_frames,
                new_obj_ids=new_det_obj_ids_local,
                new_obj_masks=new_det_masks,
                tracker_states_local=tracker_states_local,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                feature_cache=feature_cache,
            )

        if len(obj_ids_newly_removed) > 0:
            self._tracker_remove_objects(tracker_states_local, obj_ids_newly_removed)

        return tracker_states_local

    def build_outputs(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        tracker_low_res_masks_global: Tensor,
        tracker_obj_scores_global: Tensor,
        tracker_metadata_prev: Dict[str, npt.NDArray],
        tracker_update_plan: Dict[str, npt.NDArray],
        orig_vid_height: int,
        orig_vid_width: int,
        reconditioned_obj_ids: set = None,
        det_to_matched_trk_obj_ids: dict = None,
    ):
        new_det_fa_inds: npt.NDArray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: npt.NDArray = tracker_update_plan["new_det_obj_ids"]
        obj_id_to_mask = {}

        existing_masklet_obj_ids = tracker_metadata_prev["obj_ids_all_gpu"]
        existing_masklet_video_res_masks = F.interpolate(
            tracker_low_res_masks_global.unsqueeze(1),
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )
        existing_masklet_binary = existing_masklet_video_res_masks > 0
        assert len(existing_masklet_obj_ids) == len(existing_masklet_binary)
        for obj_id, mask in zip(existing_masklet_obj_ids, existing_masklet_binary):
            obj_id_to_mask[obj_id] = mask

        new_det_fa_inds_t = torch.from_numpy(new_det_fa_inds)
        new_det_low_res_masks = det_out["mask"][new_det_fa_inds_t].unsqueeze(1)
        new_det_low_res_masks = fill_holes_in_mask_scores(
            new_det_low_res_masks,
            max_area=self.fill_hole_area,
            fill_holes=True,
            remove_sprinkles=True,
        )
        new_masklet_video_res_masks = F.interpolate(
            new_det_low_res_masks,
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )

        new_masklet_binary = new_masklet_video_res_masks > 0
        assert len(new_det_obj_ids) == len(new_masklet_video_res_masks)
        for obj_id, mask in zip(new_det_obj_ids, new_masklet_binary):
            obj_id_to_mask[obj_id] = mask

        if reconditioned_obj_ids is not None and len(reconditioned_obj_ids) > 0:
            trk_id_to_max_iou_high_conf_det = tracker_update_plan.get(
                "trk_id_to_max_iou_high_conf_det", {}
            )

            for obj_id in reconditioned_obj_ids:
                det_idx = trk_id_to_max_iou_high_conf_det.get(obj_id)

                if det_idx is not None:
                    det_mask = det_out["mask"][det_idx]
                    det_mask = det_mask.unsqueeze(0).unsqueeze(0)
                    det_mask_resized = (
                        F.interpolate(
                            det_mask.float(),
                            size=(orig_vid_height, orig_vid_width),
                            mode="bilinear",
                            align_corners=False,
                        )
                        > 0
                    )

                    det_mask_final = det_mask_resized.squeeze(0)
                    obj_id_to_mask[obj_id] = det_mask_final

        return obj_id_to_mask

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: Tensor,
        last_occluded: List[int],
        obj_ids: List[int],
        frame_idx: int = None,
        reverse: bool = False,
    ):
        assert binary_low_res_masks.dtype == torch.bool, (
            f"Expected boolean tensor, got {binary_low_res_masks.dtype}"
        )
        to_suppress = torch.zeros(
            binary_low_res_masks.size(0),
            device=binary_low_res_masks.device,
            dtype=torch.bool,
        )
        if len(obj_ids) <= 1:
            return to_suppress

        iou = mask_iou(binary_low_res_masks, binary_low_res_masks)

        mask_iou_thresh = (
            iou >= self.suppress_overlapping_based_on_recent_occlusion_threshold
        )
        overlapping_pairs = torch.triu(mask_iou_thresh, diagonal=1)

        last_occ_expanded_i = last_occluded.unsqueeze(1)
        last_occ_expanded_j = last_occluded.unsqueeze(0)
        cmp_op = torch.gt if not reverse else torch.lt
        suppress_i_mask = (
            overlapping_pairs
            & cmp_op(
                last_occ_expanded_i, last_occ_expanded_j
            )
            & (
                last_occ_expanded_j > -1
            )
        )
        suppress_j_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_j, last_occ_expanded_i)
            & (
                last_occ_expanded_i > -1
            )
        )
        to_suppress = suppress_i_mask.any(dim=1) | suppress_j_mask.any(dim=0)

        if (
            self.rank == 0
            and logger.isEnabledFor(logging.DEBUG)
            and frame_idx is not None
        ):
            suppress_i_mask = suppress_i_mask.cpu().numpy()
            suppress_j_mask = suppress_j_mask.cpu().numpy()
            last_occluded = last_occluded.cpu().numpy()

            batch_size = suppress_i_mask.shape[0]

            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_i_mask[i, j]:
                        logger.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[i]} last occluded {last_occluded[i]} in favor of {obj_ids[j]} last occluded {last_occluded[j]}"
                        )

            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_j_mask[i, j]:
                        logger.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[j]} last occluded {last_occluded[j]} in favor of {obj_ids[i]} last occluded {last_occluded[i]}"
                        )

        return to_suppress

    def _propogate_tracker_one_frame_local_gpu(
        self,
        inference_states: List[Any],
        frame_idx: int,
        reverse: bool,
        run_mem_encoder: bool = False,
    ):
        """
        inference_states: List of inference states, each state corresponds to a different set of objects.
        """
        obj_ids_local = []
        low_res_masks_list = []
        obj_scores_list = []
        for inference_state in inference_states:
            if len(inference_state["obj_ids"]) == 0:
                continue

            num_frames_propagated = 0
            for out in self.tracker.propagate_in_video(
                inference_state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=0,
                reverse=reverse,
                tqdm_disable=True,
                run_mem_encoder=run_mem_encoder,
            ):
                out_frame_idx, out_obj_ids, out_low_res_masks, _, out_obj_scores = out
                num_frames_propagated += 1

            assert num_frames_propagated == 1 and out_frame_idx == frame_idx, (
                f"num_frames_propagated: {num_frames_propagated}, out_frame_idx: {out_frame_idx}, frame_idx: {frame_idx}"
            )
            assert isinstance(out_obj_ids, list)
            obj_ids_local.extend(out_obj_ids)
            low_res_masks_list.append(out_low_res_masks.squeeze(1))
            obj_scores_list.append(out_obj_scores.squeeze(1))

        H_mask = W_mask = self.tracker.low_res_mask_size
        if len(low_res_masks_list) > 0:
            low_res_masks_local = torch.cat(low_res_masks_list, dim=0)
            obj_scores_local = torch.cat(obj_scores_list, dim=0)
            assert low_res_masks_local.shape[1:] == (H_mask, W_mask)

            low_res_masks_local = fill_holes_in_mask_scores(
                low_res_masks_local.unsqueeze(1),
                max_area=self.fill_hole_area,
                fill_holes=True,
                remove_sprinkles=True,
            )
            low_res_masks_local = low_res_masks_local.squeeze(1)
        else:
            low_res_masks_local = torch.zeros(0, H_mask, W_mask, device=self.device)
            obj_scores_local = torch.zeros(0, device=self.device)

        return obj_ids_local, low_res_masks_local, obj_scores_local

    def _associate_det_trk(
        self,
        det_masks: Tensor,
        det_scores_np: npt.NDArray,
        trk_masks: Tensor,
        trk_obj_ids: npt.NDArray,
    ):
        """
        Match detections on the current frame with the existing masklets.

        Args:
          - det_masks: (N, H, W) tensor of predicted masks
          - det_scores_np: (N,) array of detection scores
          - trk_masks: (M, H, W) tensor of track masks
          - trk_obj_ids: (M,) array of object IDs corresponding to trk_masks

        Returns:
          - new_det_fa_inds: array of new object indices.
          - unmatched_trk_obj_ids: array of existing masklet object IDs that are not matched
            to any detections on this frame (for unmatched, we only count masklets with >0 area)
          - det_to_matched_trk_obj_ids: dict[int, npt.NDArray]: mapping from detector's detection indices
            to the list of matched tracklet object IDs
          - empty_trk_obj_ids: array of existing masklet object IDs with zero area in SAM2 prediction
        """
        iou_threshold = self.assoc_iou_thresh
        iou_threshold_trk = self.trk_assoc_iou_thresh
        new_det_thresh = self.new_det_thresh

        assert det_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.size(0) == len(trk_obj_ids), (
            f"trk_masks and trk_obj_ids should have the same length, {trk_masks.size(0)} vs {len(trk_obj_ids)}"
        )
        if trk_masks.size(0) == 0:
            new_det_fa_inds = np.arange(det_masks.size(0))
            unmatched_trk_obj_ids = np.array([], np.int64)
            empty_trk_obj_ids = np.array([], np.int64)
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )
        elif det_masks.size(0) == 0:
            new_det_fa_inds = np.array([], np.int64)
            trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2)).cpu().numpy()
            unmatched_trk_obj_ids = trk_obj_ids[trk_is_nonempty]
            empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )

        if det_masks.shape[-2:] != trk_masks.shape[-2:]:
            if np.prod(det_masks.shape[-2:]) < np.prod(trk_masks.shape[-2:]):
                trk_masks = F.interpolate(
                    trk_masks.unsqueeze(1),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                det_masks = F.interpolate(
                    det_masks.unsqueeze(1),
                    size=trk_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        det_masks_binary = det_masks > 0
        trk_masks_binary = trk_masks > 0
        ious = mask_iou(det_masks_binary, trk_masks_binary)

        ious_np = ious.cpu().numpy()
        if self.o2o_matching_masklets_enable:
            from scipy.optimize import linear_sum_assignment

            cost_matrix = 1 - ious_np
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            trk_is_matched = np.zeros(trk_masks.size(0), dtype=bool)
            for d, t in zip(row_ind, col_ind):
                if ious_np[d, t] >= iou_threshold_trk:
                    trk_is_matched[t] = True
        else:
            trk_is_matched = (ious_np >= iou_threshold_trk).any(axis=0)
        trk_is_nonempty = trk_masks_binary.any(dim=(1, 2)).cpu().numpy()
        trk_is_unmatched = np.logical_and(trk_is_nonempty, ~trk_is_matched)
        unmatched_trk_obj_ids = trk_obj_ids[trk_is_unmatched]
        empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]

        is_new_det = np.logical_and(
            det_scores_np >= new_det_thresh,
            np.logical_not(np.any(ious_np >= iou_threshold, axis=1)),
        )
        new_det_fa_inds = np.nonzero(is_new_det)[0]

        det_to_matched_trk_obj_ids = {}
        trk_id_to_max_iou_high_conf_det = {}
        HIGH_CONF_THRESH = 0.8
        HIGH_IOU_THRESH = 0.8
        det_to_max_iou_trk_idx = np.argmax(ious_np, axis=1)
        det_is_high_conf = (det_scores_np >= HIGH_CONF_THRESH) & ~is_new_det
        det_is_high_iou = np.max(ious_np, axis=1) >= HIGH_IOU_THRESH
        det_is_high_conf_and_iou = set(
            np.nonzero(det_is_high_conf & det_is_high_iou)[0]
        )
        for d in range(det_masks.size(0)):
            det_to_matched_trk_obj_ids[d] = trk_obj_ids[ious_np[d, :] >= iou_threshold]
            if d in det_is_high_conf_and_iou:
                trk_obj_id = trk_obj_ids[det_to_max_iou_trk_idx[d]].item()
                trk_id_to_max_iou_high_conf_det[trk_obj_id] = d

        return (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        )

    def _assign_new_det_to_gpus(self, new_det_num, prev_workload_per_gpu):
        """Distribute the new objects to the GPUs with the least workload."""
        workload_per_gpu: npt.NDArray = prev_workload_per_gpu.copy()
        new_det_gpu_ids = np.zeros(new_det_num, np.int64)

        for i in range(len(new_det_gpu_ids)):
            min_gpu = np.argmin(workload_per_gpu)
            new_det_gpu_ids[i] = min_gpu
            workload_per_gpu[min_gpu] += 1
        return new_det_gpu_ids

    def _process_hotstart(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: Dict[int, npt.NDArray],
        new_det_obj_ids: npt.NDArray,
        empty_trk_obj_ids: npt.NDArray,
        unmatched_trk_obj_ids: npt.NDArray,
        rank0_metadata: Dict[str, Any],
        tracker_metadata: Dict[str, Any],
    ):
        """Handle hotstart heuristics to remove unmatched or duplicated objects."""
        obj_first_frame_idx = rank0_metadata["obj_first_frame_idx"]
        unmatched_frame_inds = rank0_metadata["unmatched_frame_inds"]
        trk_keep_alive = rank0_metadata["trk_keep_alive"]
        overlap_pair_to_frame_inds = rank0_metadata["overlap_pair_to_frame_inds"]
        removed_obj_ids = rank0_metadata["removed_obj_ids"]
        suppressed_obj_ids = rank0_metadata["suppressed_obj_ids"][frame_idx]

        obj_ids_newly_removed = set()
        hotstart_diff = (
            frame_idx - self.hotstart_delay
            if not reverse
            else frame_idx + self.hotstart_delay
        )

        for obj_id in new_det_obj_ids:
            if obj_id not in obj_first_frame_idx:
                obj_first_frame_idx[obj_id] = frame_idx
            assert obj_id not in trk_keep_alive
            trk_keep_alive[obj_id] = self.init_trk_keep_alive

        matched_trks = set()
        for matched_trks_per_det in det_to_matched_trk_obj_ids.values():
            matched_trks.update(matched_trks_per_det)
        for obj_id in matched_trks:
            trk_keep_alive[obj_id] = min(
                self.max_trk_keep_alive, trk_keep_alive[obj_id] + 1
            )
        for obj_id in unmatched_trk_obj_ids:
            unmatched_frame_inds[obj_id].append(frame_idx)
            trk_keep_alive[obj_id] = max(
                self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1
            )
        if self.decrease_trk_keep_alive_for_empty_masklets:
            for obj_id in empty_trk_obj_ids:
                trk_keep_alive[obj_id] = max(
                    self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1
                )

        for obj_id, frame_indices in unmatched_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue
            if len(frame_indices) >= self.hotstart_unmatch_thresh:
                is_within_hotstart = (
                    obj_first_frame_idx[obj_id] > hotstart_diff and not reverse
                ) or (obj_first_frame_idx[obj_id] < hotstart_diff and reverse)
                if is_within_hotstart:
                    obj_ids_newly_removed.add(obj_id)
                    logger.debug(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it is unmatched for frames: {frame_indices}"
                    )
            if (
                trk_keep_alive[obj_id] <= 0
                and not self.suppress_unmatched_only_within_hotstart
                and obj_id not in removed_obj_ids
                and obj_id not in obj_ids_newly_removed
            ):
                logger.debug(
                    f"Suppressing object {obj_id} at frame {frame_idx}, due to being unmatched"
                )
                suppressed_obj_ids.add(obj_id)

        for _, matched_trk_obj_ids in det_to_matched_trk_obj_ids.items():
            if len(matched_trk_obj_ids) < 2:
                continue
            first_appear_obj_id = (
                min(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
                if not reverse
                else max(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
            )
            for obj_id in matched_trk_obj_ids:
                if obj_id != first_appear_obj_id:
                    key = (first_appear_obj_id, obj_id)
                    overlap_pair_to_frame_inds[key].append(frame_idx)

        for (first_obj_id, obj_id), frame_indices in overlap_pair_to_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue
            if (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                obj_first_frame_idx[obj_id] < hotstart_diff and reverse
            ):
                if len(frame_indices) >= self.hotstart_dup_thresh:
                    obj_ids_newly_removed.add(obj_id)
                    logger.debug(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it overlaps with another track {first_obj_id} at frames: {frame_indices}"
                    )

        removed_obj_ids.update(obj_ids_newly_removed)
        return obj_ids_newly_removed, rank0_metadata

    def _tracker_update_memories(
        self,
        tracker_inference_states: List[Any],
        frame_idx: int,
        tracker_metadata: Dict[str, Any],
        low_res_masks: Tensor,
    ):
        """
        Run Sam2 memory encoder, enforcing non-overlapping constraints globally.
        """
        if len(tracker_inference_states) == 0:
            return
        high_res_H, high_res_W = (
            self.tracker.maskmem_backbone.mask_downsampler.interpol_size
        )
        high_res_masks = F.interpolate(
            low_res_masks.unsqueeze(1),
            size=(high_res_H, high_res_W),
            mode="bilinear",
            align_corners=False,
        )
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            high_res_masks = self.tracker._suppress_object_pw_area_shrinkage(
                high_res_masks
            )
        object_score_logits = torch.where(
            (high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0
        )

        start_idx_gpu = sum(tracker_metadata["num_obj_per_gpu"][: self.rank])
        start_idx_state = start_idx_gpu
        for tracker_state in tracker_inference_states:
            num_obj_per_state = len(tracker_state["obj_ids"])
            if num_obj_per_state == 0:
                continue
            end_idx_state = start_idx_state + num_obj_per_state
            local_high_res_masks = high_res_masks[start_idx_state:end_idx_state]
            local_object_score_logits = object_score_logits[
                start_idx_state:end_idx_state
            ]
            local_batch_size = local_high_res_masks.size(0)

            encoded_mem = self.tracker._run_memory_encoder(
                tracker_state,
                frame_idx,
                local_batch_size,
                local_high_res_masks,
                local_object_score_logits,
                is_mask_from_pts=False,
            )
            local_maskmem_features, local_maskmem_pos_enc = encoded_mem
            output_dict = tracker_state["output_dict"]
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                if frame_idx not in output_dict[storage_key]:
                    continue
                output_dict[storage_key][frame_idx]["maskmem_features"] = (
                    local_maskmem_features
                )
                output_dict[storage_key][frame_idx]["maskmem_pos_enc"] = [
                    pos for pos in local_maskmem_pos_enc
                ]
                self.tracker._add_output_per_object(
                    inference_state=tracker_state,
                    frame_idx=frame_idx,
                    current_out=output_dict[storage_key][frame_idx],
                    storage_key=storage_key,
                )
            start_idx_state += num_obj_per_state

    def _tracker_add_new_objects(
        self,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: List[int],
        new_obj_masks: Tensor,
        tracker_states_local: List[Any],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: Dict,
    ):
        """Add a new object to SAM2 inference states."""
        prev_tracker_state = (
            tracker_states_local[0] if len(tracker_states_local) > 0 else None
        )

        new_tracker_state = self.tracker.init_state(
            cached_features=feature_cache,
            video_height=orig_vid_height,
            video_width=orig_vid_width,
            num_frames=num_frames,
        )
        new_tracker_state["backbone_out"] = (
            prev_tracker_state.get("backbone_out", None)
            if prev_tracker_state is not None
            else None
        )

        assert len(new_obj_ids) == new_obj_masks.size(0)
        assert new_obj_masks.is_floating_point()
        input_mask_res = self.tracker.input_mask_size
        new_obj_masks = F.interpolate(
            new_obj_masks.unsqueeze(1),
            size=(input_mask_res, input_mask_res),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        new_obj_masks = new_obj_masks > 0

        for new_obj_id, new_mask in zip(new_obj_ids, new_obj_masks):
            self.tracker.add_new_mask(
                inference_state=new_tracker_state,
                frame_idx=frame_idx,
                obj_id=new_obj_id,
                mask=new_mask,
                add_mask_to_memory=True,
            )
        self.tracker.propagate_in_video_preflight(
            new_tracker_state, run_mem_encoder=True
        )
        tracker_states_local.append(new_tracker_state)
        return tracker_states_local

    def _tracker_remove_object(self, tracker_states_local: List[Any], obj_id: int):
        """
        Remove an object from SAM2 inference states. This would remove the object from
        all frames in the video.
        """
        tracker_states_local_before_removal = tracker_states_local.copy()
        tracker_states_local.clear()
        for tracker_inference_state in tracker_states_local_before_removal:
            new_obj_ids, _ = self.tracker.remove_object(
                tracker_inference_state, obj_id, strict=False, need_output=False
            )
            if len(new_obj_ids) > 0:
                tracker_states_local.append(tracker_inference_state)

    def _tracker_remove_objects(
        self, tracker_states_local: List[Any], obj_ids: list[int]
    ):
        """
        Remove an object from SAM2 inference states. This would remove the object from
        all frames in the video.
        """
        for obj_id in obj_ids:
            self._tracker_remove_object(tracker_states_local, obj_id)

    def _initialize_metadata(self):
        """Initialize metadata for the masklets."""
        tracker_metadata = {
            "obj_ids_per_gpu": [np.array([], np.int64) for _ in range(self.world_size)],
            "obj_ids_all_gpu": np.array([], np.int64),
            "num_obj_per_gpu": np.zeros(self.world_size, np.int64),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "obj_id_to_tracker_score_frame_wise": defaultdict(dict),
            "obj_id_to_last_occluded": {},
        }
        if self.rank == 0:
            rank0_metadata = {
                "obj_first_frame_idx": {},
                "unmatched_frame_inds": defaultdict(list),
                "trk_keep_alive": defaultdict(
                    int
                ),
                "overlap_pair_to_frame_inds": defaultdict(list),
                "removed_obj_ids": set(),
                "suppressed_obj_ids": defaultdict(
                    set
                ),
            }
            if self.masklet_confirmation_enable:
                rank0_metadata["masklet_confirmation"] = {
                    "status": np.array([], np.int64),
                    "consecutive_det_num": np.array([], np.int64),
                }
            tracker_metadata["rank0_metadata"] = rank0_metadata

        return tracker_metadata

    def update_masklet_confirmation_status(
        self,
        rank0_metadata: Dict[str, Any],
        obj_ids_all_gpu_prev: npt.NDArray,
        obj_ids_all_gpu_updated: npt.NDArray,
        det_to_matched_trk_obj_ids: Dict[int, npt.NDArray],
        new_det_obj_ids: npt.NDArray,
    ):
        confirmation_data = rank0_metadata["masklet_confirmation"]

        status_prev = confirmation_data["status"]
        consecutive_det_num_prev = confirmation_data["consecutive_det_num"]
        assert status_prev.shape == obj_ids_all_gpu_prev.shape, (
            f"Got {status_prev.shape} vs {obj_ids_all_gpu_prev.shape}"
        )

        obj_id_to_updated_idx = {
            obj_id: idx for idx, obj_id in enumerate(obj_ids_all_gpu_updated)
        }
        prev_elem_is_in_updated = np.isin(obj_ids_all_gpu_prev, obj_ids_all_gpu_updated)
        prev_elem_obj_ids_in_updated = obj_ids_all_gpu_prev[prev_elem_is_in_updated]
        prev_elem_inds_in_updated = np.array(
            [obj_id_to_updated_idx[obj_id] for obj_id in prev_elem_obj_ids_in_updated],
            dtype=np.int64,
        )
        unconfirmed_val = MaskletConfirmationStatus.UNCONFIRMED.value
        status = np.full_like(obj_ids_all_gpu_updated, fill_value=unconfirmed_val)
        status[prev_elem_inds_in_updated] = status_prev[prev_elem_is_in_updated]
        consecutive_det_num = np.zeros_like(obj_ids_all_gpu_updated)
        consecutive_det_num[prev_elem_inds_in_updated] = consecutive_det_num_prev[
            prev_elem_is_in_updated
        ]

        is_matched = np.isin(obj_ids_all_gpu_updated, new_det_obj_ids)
        for matched_trk_obj_ids in det_to_matched_trk_obj_ids.values():
            is_matched |= np.isin(obj_ids_all_gpu_updated, matched_trk_obj_ids)
        consecutive_det_num = np.where(is_matched, consecutive_det_num + 1, 0)

        change_to_confirmed = (
            consecutive_det_num >= self.masklet_confirmation_consecutive_det_thresh
        )
        status[change_to_confirmed] = MaskletConfirmationStatus.CONFIRMED.value

        confirmation_data["status"] = status
        confirmation_data["consecutive_det_num"] = consecutive_det_num
        return rank0_metadata

    def forward(self, input: BatchedDatapoint, is_inference: bool = False):
        raise NotImplementedError("Evaluation outside demo is not implemented yet")

    def _load_checkpoint(self, ckpt_path: str, strict: bool = True):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=strict)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            logger.warning(f"Loaded ckpt with {missing_keys=}, {unexpected_keys=}")
        else:
            logger.info("Loaded ckpt successfully without missing or unexpected keys")

    def prep_for_evaluator(self, video_frames, tracking_res, scores_labels):
        """This method is only used for benchmark eval (not used in the demo)."""
        num_frames = len(video_frames)
        w, h = video_frames[0].size
        zero_mask = torch.zeros((1, h, w), dtype=torch.bool)
        object_ids = list(scores_labels.keys())
        preds = {"scores": [], "labels": [], "boxes": [], "masks_rle": []}
        for oid in object_ids:
            o_masks = []
            o_score = scores_labels[oid][0].item()
            o_label = scores_labels[oid][1]
            for frame_idx in range(num_frames):
                if frame_idx not in tracking_res:
                    o_masks.append(zero_mask)
                else:
                    o_masks.append(tracking_res[frame_idx].get(oid, zero_mask))

            o_masks = torch.cat(o_masks, dim=0)
            preds["scores"].append(o_score)
            preds["labels"].append(o_label)
            preds["boxes"].append(mask_to_box(o_masks.unsqueeze(1)).squeeze())
            preds["masks_rle"].append(rle_encode(o_masks, return_areas=True))

        preds["boxes"] = (
            torch.stack(preds["boxes"], dim=0)
            if len(preds["boxes"]) > 0
            else torch.empty(
                (0, num_frames, 4), dtype=torch.float32, device=self.device
            )
        )
        preds["scores"] = (
            torch.tensor(preds["scores"], device=self.device)
            if len(preds["scores"]) > 0
            else torch.empty((0,), device=self.device)
        )
        preds["per_frame_scores"] = preds["scores"]
        preds["labels"] = (
            torch.tensor(preds["labels"], device=self.device)
            if len(preds["labels"]) > 0
            else torch.empty((0,), device=self.device)
        )
        return preds

    def _encode_prompt(self, **kwargs):
        return self.detector._encode_prompt(**kwargs)

    def _drop_new_det_with_obj_limit(self, new_det_fa_inds, det_scores_np, num_to_keep):
        """
        Drop a few new detections based on the maximum number of objects. We drop new objects based
        on their detection scores, keeping the high-scoring ones and dropping the low-scoring ones.
        """
        assert 0 <= num_to_keep <= len(new_det_fa_inds)
        if num_to_keep == 0:
            return np.array([], np.int64)
        if num_to_keep == len(new_det_fa_inds):
            return new_det_fa_inds

        score_order = np.argsort(det_scores_np[new_det_fa_inds])[::-1]
        new_det_fa_inds = new_det_fa_inds[score_order[:num_to_keep]]
        return new_det_fa_inds
