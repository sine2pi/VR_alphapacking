
"""
Base predictor class shared by SAM3 and SAM3.1 (multiplex) video predictors.

Provides the common handle_request/handle_stream_request API and session management.
Subclasses only need to override methods where their behavior differs.
"""

import gc
import time
import uuid
from typing import Dict, List, Optional

import torch
from sam3.logger import get_logger

logger = get_logger(__name__)

_CLEAR_CACHE_THRESHOLD = 100

class Sam3BasePredictor:
    """
    Base class for SAM3 video predictors. Provides:
    - Session management (start, reset, close)
    - Request dispatch (handle_request / handle_stream_request)
    - Common add_prompt / propagate_in_video / remove_object / reset_session / close_session

    Subclasses must set `self.model` and `self._all_inference_states` before use.
    """

    def __init__(self):
        self.model = None
        self._all_inference_states: Dict[str, dict] = {}
    
    def set_cache_threshold(self, max_frame=None):
        """Set the threshold for clearing the CUDA cache on session close."""
        global _CLEAR_CACHE_THRESHOLD
        _CLEAR_CACHE_THRESHOLD = min(max_frame, 200) if max_frame is not None else 100

    @torch.inference_mode()
    def handle_request(self, request):
        """Dispatch a request based on its type."""
        request_type = request["type"]
        if request_type == "start_session":
            return self.start_session(
                resource_path=request["resource_path"],
                session_id=request.get("session_id", None),
                offload_video_to_cpu=request.get("offload_video_to_cpu", False),
                offload_state_to_cpu=request.get("offload_state_to_cpu", False),
            )
        elif request_type == "add_prompt":
            return self.add_prompt(
                session_id=request["session_id"],
                frame_idx=request["frame_idx"],
                text=request.get("text", None),
                points=request.get("points", None),
                point_labels=request.get("point_labels", None),
                clear_old_points=request.get("clear_old_points", True),
                bounding_boxes=request.get("bounding_boxes", None),
                bounding_box_labels=request.get("bounding_box_labels", None),
                clear_old_boxes=request.get("clear_old_boxes", True),
                output_prob_thresh=request.get(
                    "output_prob_thresh",
                    getattr(self, "default_output_prob_thresh", 0.5),
                ),
                obj_id=request.get("obj_id", None),
                rel_coordinates=request.get("rel_coordinates", True),
            )
        elif request_type == "remove_object":
            return self.remove_object(
                session_id=request["session_id"],
                frame_idx=request.get("frame_idx", 0),
                obj_id=request["obj_id"],
            )
        elif request_type == "reset_session":
            return self.reset_session(session_id=request["session_id"])
        elif request_type == "cancel_propagation":
            return self.cancel_propagation(session_id=request["session_id"])
        elif request_type == "close_session":
            return self.close_session(
                session_id=request["session_id"],
                run_gc_collect=request.get("run_gc_collect", True),
                clear_cache_threshold=int(
                    request.get("clear_cache_threshold", _CLEAR_CACHE_THRESHOLD)
                ),
            )
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    @torch.inference_mode()
    def handle_stream_request(self, request):
        """Dispatch a stream request based on its type."""
        request_type = request["type"]
        if request_type == "propagate_in_video":
            yield from self.propagate_in_video(
                session_id=request["session_id"],
                propagation_direction=request.get("propagation_direction", "both"),
                start_frame_idx=request.get("start_frame_idx", None),
                max_frame_num_to_track=request.get("max_frame_num_to_track", None),
                output_prob_thresh=request.get(
                    "output_prob_thresh",
                    getattr(self, "default_output_prob_thresh", 0.5),
                ),
            )
        else:
            raise RuntimeError(f"invalid request type: {request_type}")

    def start_session(
        self,
        resource_path,
        session_id=None,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        """Start a new inference session on a video directory or path."""
        init_kwargs = dict(
            resource_path=resource_path,
            offload_video_to_cpu=offload_video_to_cpu,
        )
        if hasattr(self, "async_loading_frames"):
            init_kwargs["async_loading_frames"] = self.async_loading_frames
        if hasattr(self, "video_loader_type"):
            init_kwargs["video_loader_type"] = self.video_loader_type
        inference_state = self.model.init_state(**init_kwargs)

        if not session_id:
            session_id = str(uuid.uuid4())
        self._all_inference_states[session_id] = {
            "state": inference_state,
            "session_id": session_id,
            "start_time": time.time(),
            "last_use_time": time.time(),
        }
        logger.info(f"started new session {session_id}")
        return {"session_id": session_id}

    def add_prompt(
        self,
        session_id: str,
        frame_idx: int,
        text: Optional[str] = None,
        points=None,
        point_labels=None,
        clear_old_points: bool = True,
        bounding_boxes=None,
        bounding_box_labels=None,
        clear_old_boxes: bool = True,
        output_prob_thresh: float = 0.5,
        obj_id: Optional[int] = None,
        rel_coordinates: bool = True,
    ):
        """Add text, box and/or point prompt on a specific video frame."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)

        if points is not None and not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if point_labels is not None and not isinstance(point_labels, torch.Tensor):
            point_labels = torch.tensor(point_labels, dtype=torch.int32)
        if bounding_boxes is not None and not isinstance(bounding_boxes, torch.Tensor):
            bounding_boxes = torch.tensor(bounding_boxes, dtype=torch.float32)
        if bounding_box_labels is not None and not isinstance(
            bounding_box_labels, torch.Tensor
        ):
            bounding_box_labels = torch.tensor(bounding_box_labels, dtype=torch.int32)

        kwargs = dict(
            inference_state=inference_state,
            frame_idx=frame_idx,
            text_str=text,
            points=points,
            point_labels=point_labels,
            clear_old_points=clear_old_points,
            boxes_xywh=bounding_boxes,
            box_labels=bounding_box_labels,
            clear_old_boxes=clear_old_boxes,
            output_prob_thresh=output_prob_thresh,
            rel_coordinates=rel_coordinates,
        )
        if obj_id is not None:
            kwargs["obj_id"] = obj_id

        import inspect

        sig = inspect.signature(self.model.add_prompt)
        valid_params = set(sig.parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            frame_idx, outputs = self.model.add_prompt(**filtered_kwargs)

        return {"frame_idx": frame_idx, "outputs": outputs}

    def remove_object(
        self,
        session_id: str,
        frame_idx: int = 0,
        obj_id: int = 0,
        is_user_action: bool = True,
    ):
        """Remove an object from tracking."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)

        result = self.model.remove_object(
            inference_state, obj_id, frame_idx=frame_idx, is_user_action=is_user_action
        )
        if result is None or (isinstance(result, tuple) and result[1] is None):
            import numpy as np

            out_obj_ids = torch.zeros(0, dtype=torch.int64)
            out_binary_masks = torch.zeros(
                0,
                inference_state["orig_height"],
                inference_state["orig_width"],
                dtype=torch.bool,
            )
            out_boxes_xywh = torch.zeros(0, 4, dtype=torch.float32)
            outputs = {
                "out_obj_ids": out_obj_ids.cpu().numpy(),
                "out_boxes_xywh": out_boxes_xywh.cpu().numpy(),
                "out_binary_masks": out_binary_masks.cpu().numpy(),
            }
        elif isinstance(result, tuple):
            _, outputs = result
        else:
            outputs = result
       
        return {"frame_idx": frame_idx, "outputs": outputs}

    def cancel_propagation(self, session_id):
        """Cancel any ongoing propagation. No-op if not supported by the model."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)
        if hasattr(self.model, "cancel_propagation"):
            self.model.cancel_propagation(inference_state)
        return {"is_success": True}

    def propagate_in_video(
        self,
        session_id,
        propagation_direction="both",
        start_frame_idx=None,
        max_frame_num_to_track=None,
        output_prob_thresh=0.5,
        **kwargs,
    ):
        """Propagate the added prompts to get results on all video frames."""
        try:
            session = self._get_session(session_id)
            inference_state = session["state"]
            self._extend_expiration_time(session)
            if propagation_direction not in ["both", "forward", "backward"]:
                raise ValueError(
                    f"invalid propagation direction: {propagation_direction}"
                )

            propagate_kwargs = dict(
                inference_state=inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
            )
            print(f'max_frame_num_to_track: {max_frame_num_to_track}')
            print(f'propagation_direction: {propagation_direction}')
            print(f'output_prob_thresh: {output_prob_thresh}')
            self.set_cache_threshold(max_frame_num_to_track)
            import inspect

            sig = inspect.signature(self.model.propagate_in_video)
            if "output_prob_thresh" in sig.parameters:
                propagate_kwargs["output_prob_thresh"] = output_prob_thresh
            for k, v in kwargs.items():
                if k in sig.parameters:
                    propagate_kwargs[k] = v

            if propagation_direction in ["both", "forward"]:
                for frame_idx, outputs in self.model.propagate_in_video(
                    **propagate_kwargs,
                    reverse=False,
                ):
                    yield {"frame_idx": frame_idx, "outputs": outputs}
            if propagation_direction in ["both", "backward"]:
                for frame_idx, outputs in self.model.propagate_in_video(
                    **propagate_kwargs,
                    reverse=True,
                ):
                    yield {"frame_idx": frame_idx, "outputs": outputs}
        finally:
            logger.info(f"propagation ended in session {session_id}")

    def reset_session(self, session_id):
        """Reset the session to its initial state."""
        session = self._get_session(session_id)
        inference_state = session["state"]
        self._extend_expiration_time(session)
        self.model.reset_state(inference_state)
        return {"is_success": True}

    def close_session(
        self,
        session_id,
        run_gc_collect=True,
        clear_cache_threshold: int = _CLEAR_CACHE_THRESHOLD,
    ):
        """Close a session. Idempotent.

        ``run_gc_collect=True`` (the default) also returns the session's
        freed CUDA tensors back to the device by calling
        ``torch.cuda.empty_cache()`` after ``gc.collect()``. Without this,
        PyTorch's caching allocator retains the freed allocations in its
        per-process pool, so reserved memory keeps climbing across
        long-running workloads even though the Python-level objects are gone.

        ``empty_cache()`` itself triggers a CUDA sync, so it is gated on
        device memory pressure via the ``gpu_mem`` snapshot. Callers can
        override the threshold per-call via ``clear_cache_threshold``.

        When ``run_gc_collect=True``, the response includes a ``gpu_mem``
        snapshot (free / total / allocated / reserved bytes, plus active
        session count) so clients can decide whether the device has
        headroom for their next session — no separate admission RPC
        needed. See ``_gpu_mem_snapshot`` for the field shape. The
        first snapshot (after ``gc.collect()``) drives the cache-eviction
        decision; if ``empty_cache()`` fires, a second snapshot is taken
        so the response reflects the post-cleanup state and the freed
        bytes are logged. Old callers that only read ``is_success`` work
        unchanged — additional dict keys are ignored at the JSON layer.
        """
        session = self._all_inference_states.pop(session_id, None)
        result = {"is_success": True}
        if session is None:
            logger.warning(f"cannot close session {session_id} as it does not exist")
        else:
            state = session.get("state")
            if isinstance(state, dict):
                state.clear()
            session.clear()
            del session
            if run_gc_collect:
                gc.collect()
                gpu_mem = self._gpu_mem_snapshot()
                if (
                    torch.cuda.is_available()
                    and gpu_mem["total_bytes"] > 0
                    and (100.0 - gpu_mem["free_pct"]) >= clear_cache_threshold
                ):
                    torch.cuda.empty_cache()
                    post_gpu_mem = self._gpu_mem_snapshot()
                    logger.info(
                        f"empty_cache freed "
                        f"{post_gpu_mem['free_bytes'] - gpu_mem['free_bytes']} bytes "
                        f"(free_pct {gpu_mem['free_pct']:.1f}% -> "
                        f"{post_gpu_mem['free_pct']:.1f}%, reserved "
                        f"{gpu_mem['reserved_bytes']} -> "
                        f"{post_gpu_mem['reserved_bytes']} bytes)"
                    )
                    gpu_mem = post_gpu_mem
                result["gpu_mem"] = gpu_mem
            logger.info(f"removed session {session_id}")
        return result

    def _gpu_mem_snapshot(self) -> dict:
        """Snapshot of current GPU memory state for inclusion in
        session-close responses.

        Lets clients track free HBM across the fleet without a separate
        admission RPC: every ``close_session`` naturally exposes the
        post-cleanup state (taken AFTER any ``empty_cache`` call), which
        is exactly what the next session will face.

        Fields:
          - ``free_bytes`` / ``total_bytes`` — raw
            ``torch.cuda.mem_get_info()``.
          - ``allocated_bytes`` — ``torch.cuda.memory_allocated()``,
            live tensor footprint (no caching pool overhead).
          - ``reserved_bytes`` — ``torch.cuda.memory_reserved()``,
            caching-allocator pool size.
          - ``free_pct`` — ``free_bytes / total_bytes * 100`` for
            convenient % thresholds.
          - ``active_session_count`` — sessions still resident on this
            predictor instance after the close.

        Fail-open: any error reading the device returns zero stats so
        a broken CUDA context (e.g., CPU-only test env) NEVER breaks
        the session-close response.
        """
        active_count = len(self._all_inference_states)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        except RuntimeError:
            return {
                "free_bytes": 0,
                "total_bytes": 0,
                "allocated_bytes": 0,
                "reserved_bytes": 0,
                "free_pct": 0.0,
                "active_session_count": active_count,
            }
        free_pct = (free_bytes / total_bytes) * 100 if total_bytes > 0 else 0.0
        allocated_bytes = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )
        reserved_bytes = (
            torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
        )
        return {
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "allocated_bytes": allocated_bytes,
            "reserved_bytes": reserved_bytes,
            "free_pct": free_pct,
            "active_session_count": active_count,
        }

    def _get_session(self, session_id):
        session = self._all_inference_states.get(session_id, None)
        if session is None:
            raise RuntimeError(
                f"Cannot find session {session_id}; it might have expired"
            )
        return session

    def _extend_expiration_time(self, session):
        """Update last-use time for session expiration tracking."""
        session["last_use_time"] = time.time()

    def shutdown(self):
        """Shutdown the predictor and clear all sessions."""
        self._all_inference_states.clear()
