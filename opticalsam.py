
"""
Augments SAM3's deep memory bank with RAFT optical flow as an explicit
spatial prior. Works by injecting RAFT-warped mask prompts into SAM3's
per_frame_geometric_prompt slots, so the tracker sees BOTH its own
temporal memory AND our geometric motion prediction each frame.
"""

import cv2, torch, subprocess, numpy as np, json, logging
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans
import torchvision.transforms.functional as TVF
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from model_builder import build_sam3_video_predictor

try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    HAS_RAFT = True
except ImportError:
    HAS_RAFT = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def metadata(path):
    cmd_key = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pict_type',
        '-of', 'csv=p=0', '-skip_frame', 'nokey', path
    ]
    res_key = subprocess.run(cmd_key, capture_output=True, text=True)
    lines = res_key.stdout.strip().split('\n')
    num_keyframes = len(lines)

    cmd_stream = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', 
        '-show_streams', '-select_streams', 'v:0', path
    ]
    res_stream = subprocess.run(cmd_stream, capture_output=True, text=True)
    data = json.loads(res_stream.stdout)
    
    if not data.get('streams'):
        return None, None, None, None, None, None
        
    stream = data['streams'][0]
    width = int(stream['width'])
    height = int(stream['height'])
    duration = float(stream.get('duration', 0))
        
    fps_str = stream.get('r_frame_rate', '30/1')
    try:
        num, denom = map(int, fps_str.split('/'))
        fps = num / denom if denom != 0 else 30.0
    except:
        fps = 30.0

    f_tot = stream.get('nb_frames')
    if f_tot:
        nb_frames = int(f_tot)
    else:
        nb_frames = int(duration * fps) if duration > 0 else 0
    return nb_frames, num_keyframes, width, height, duration, fps

def ffmpeg_pipe(out_path, width, height, fps):
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-c:v', 'hevc_nvenc', '-preset', 'fast', '-cq', '20',
        '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
        '-color_trc', 'bt709', '-color_range', 'tv', out_path
    ]
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

class RaftMotionCompensator:
    def __init__(self, device=None, max_size=256, flow_scale=0.5, interp_mode="bicubic"):
        self.device = torch.device(device) if isinstance(device, str) else (device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.max_size = max_size
        self.flow_scale = flow_scale
        self.interp_mode = interp_mode
        self.model = None
        self.transforms = None

    def _load_model(self):
        if self.model is None:
            if not HAS_RAFT:
                raise ImportError("torchvision.models.optical_flow is required for RAFT.")
            weights = Raft_Small_Weights.DEFAULT
            self.transforms = weights.transforms()
            self.model = raft_small(weights=weights, progress=False).to(self.device).eval()

    def _compute_raft_flow(self, img1, img2):
        orig_H, orig_W = img1.shape[2], img1.shape[3]
        scale_factor = self.flow_scale
        current_H, current_W = orig_H * scale_factor, orig_W * scale_factor
        if max(current_H, current_W) > self.max_size:
            scale_factor = scale_factor * (self.max_size / float(max(current_H, current_W)))
        if scale_factor != 1.0:
            new_H, new_W = int(orig_H * scale_factor), int(orig_W * scale_factor)
            img1_s = F.interpolate(img1, size=(new_H, new_W), mode=self.interp_mode, antialias=True)
            img2_s = F.interpolate(img2, size=(new_H, new_W), mode=self.interp_mode, antialias=True)
        else:
            img1_s, img2_s = img1, img2
            
        img1_t, img2_t = self.transforms(img1_s, img2_s)
        _, _, H_s, W_s = img1_t.shape
        pad_h, pad_w = (8 - H_s % 8) % 8, (8 - W_s % 8) % 8
        if pad_h > 0 or pad_w > 0:
            img1_t = F.pad(img1_t, (0, pad_w, 0, pad_h))
            img2_t = F.pad(img2_t, (0, pad_w, 0, pad_h))
            
        with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.float32):
            flow = self.model(img1_t, img2_t)[-1].float()
            
        flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :H_s, :W_s]
        if scale_factor != 1.0:
            flow = F.interpolate(flow, size=(orig_H, orig_W), mode=self.interp_mode)
            flow = flow / scale_factor
        return flow

    def _warp_frame(self, pt_frame, flow, t=1.0):
        C, H, W = pt_frame.shape
        flow_scaled = flow * t
        y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        x_norm = 2.0 * (x + flow_scaled[0]) / max(W - 1, 1) - 1.0
        y_norm = 2.0 * (y + flow_scaled[1]) / max(H - 1, 1) - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
        return F.grid_sample(
            pt_frame.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False
        ).squeeze(0)

    def stabilize_alpha_sequence(self, rgb_frames, alpha_masks, blend_weights=(0.2, 0.6, 0.2)):
        self._load_model()
        is_numpy = isinstance(rgb_frames, np.ndarray)
        if is_numpy:
            t_rgb = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
            if alpha_masks.ndim == 3:
                t_alpha = torch.from_numpy(alpha_masks).unsqueeze(1).float().div(255.0).to(self.device)
            else:
                t_alpha = torch.from_numpy(alpha_masks).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
        else:
            t_rgb = rgb_frames.to(self.device).float()
            t_alpha = alpha_masks.to(self.device).float()

        num_frames = t_rgb.shape[0]
        if num_frames < 3: return alpha_masks

        stabilized_alphas = torch.zeros_like(t_alpha)
        stabilized_alphas[0] = t_alpha[0]
        stabilized_alphas[-1] = t_alpha[-1]
        w_prev, w_curr, w_next = blend_weights

        with torch.no_grad():
            for i in tqdm(range(1, num_frames - 1)):
                prev_rgb, curr_rgb, next_rgb = t_rgb[i-1:i+2]
                prev_alpha, curr_alpha, next_alpha = t_alpha[i-1:i+2]
                flow_forward = self._compute_raft_flow(prev_rgb.unsqueeze(0), curr_rgb.unsqueeze(0)).squeeze(0)
                flow_backward = self._compute_raft_flow(next_rgb.unsqueeze(0), curr_rgb.unsqueeze(0)).squeeze(0)
                alpha_prev_warped = self._warp_frame(prev_alpha, flow_forward, t=1.0)
                alpha_next_warped = self._warp_frame(next_alpha, flow_backward, t=1.0)
                merged_alpha = (w_prev * alpha_prev_warped) + (w_curr * curr_alpha) + (w_next * alpha_next_warped)
                stabilized_alphas[i] = torch.clamp(merged_alpha, 0.0, 1.0)

        if is_numpy:
            if alpha_masks.ndim == 3:
                return (stabilized_alphas.squeeze(1).cpu().numpy() * 255.0).astype(np.uint8)
            return (stabilized_alphas.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
        return stabilized_alphas

class HybridSam3MotionLoop:

    def __init__(self, video_predictor=None, raft_compensator=None, target_res=(256, 256)):

        self.predictor = build_sam3_video_predictor(
        offload_video_to_cpu=True,
        async_loading_frames=True) or video_predictor

        self.raft = raft_compensator or RaftMotionCompensator()
        self.device = self.raft.device
        self.target_res = target_res

    def process_batch(self, frames_pil, frames_bgr, prompt_text=None, bbox=None):
        """
        Process a batch of frames end-to-end:
        1. Run vanilla propagation -> baseline vanilla masks & soft probabilities.
        2. Run true inline augmentation loop:
           - Prompt frame 0.
           - For frame i in 1..N-1:
             - Warp prev_logits to frame i via RAFT flow.
             - Inject warped logits into low-level tracker as the prior on frame i.
             - Run tracker step to combine the warped prior with deep memory & image features.
             - Keep track of predictions.
        Returns vanilla_masks, final_masks (inline-stabilized), sam_soft (vanilla), stabilized_soft (inline).
        """
        self.raft._load_model()
        height, width = frames_bgr[0].shape[:2]
        chunk_length = len(frames_pil)

        res_vanilla = self.predictor.handle_request(dict(
            type="start_session",
            resource_path=frames_pil
        ))
        sid_vanilla = res_vanilla["session_id"]

        prompt_req_vanilla = dict(type="add_prompt", session_id=sid_vanilla, frame_index=0, obj_id=0)
        if prompt_text is not None:
            prompt_req_vanilla["text"] = prompt_text
        if bbox is not None:
            prompt_req_vanilla["bounding_boxes"] = [bbox]
            prompt_req_vanilla["bounding_box_labels"] = [1]

        self.predictor.handle_request(prompt_req_vanilla)

        out_buffer_vanilla = []
        for st in self.predictor.handle_stream_request(dict(
            type="propagate_in_video",
            session_id=sid_vanilla,
            propagation_direction="forward",
            start_frame_index=0,
            max_frame_num_to_track=chunk_length,
        )):
            out_buffer_vanilla.append(st["outputs"])

        self.predictor.handle_request(dict(type="close_session", session_id=sid_vanilla))

        vanilla_masks = []
        sam_soft = []
        for i in range(chunk_length):
            if i < len(out_buffer_vanilla):
                outputs = out_buffer_vanilla[i]
                mask_bin = np.zeros((height, width), dtype=np.uint8)
                mask_prob = np.zeros((height, width), dtype=np.float32)

                if "out_mask_logits" in outputs:
                    for logit_tensor in outputs["out_mask_logits"]:
                        if isinstance(logit_tensor, torch.Tensor):
                            logit_np = logit_tensor.cpu().numpy()
                        else:
                            logit_np = np.array(logit_tensor)
                        if logit_np.shape != (height, width):
                            logit_np = cv2.resize(logit_np, (width, height), interpolation=cv2.INTER_LINEAR)
                        prob = 1.0 / (1.0 + np.exp(-logit_np))
                        mask_prob = np.maximum(mask_prob, prob)
                        mask_bin = np.maximum(mask_bin, (prob > 0.5).astype(np.uint8) * 255)
                elif "out_binary_masks" in outputs:
                    for m in outputs["out_binary_masks"]:
                        if isinstance(m, torch.Tensor):
                            m = m.cpu().numpy()
                        if m.shape != (height, width):
                            m = cv2.resize(m.astype(np.float32), (width, height),
                                           interpolation=cv2.INTER_NEAREST)
                        mask_prob = np.maximum(mask_prob, m.astype(np.float32))
                        mask_bin = np.maximum(mask_bin, (m > 0.5).astype(np.uint8) * 255)
                
                sam_soft.append(mask_prob)
                vanilla_masks.append(mask_bin)
            else:
                sam_soft.append(np.zeros((height, width), dtype=np.float32))
                vanilla_masks.append(np.zeros((height, width), dtype=np.uint8))

        res_inline = self.predictor.handle_request(dict(
            type="start_session",
            resource_path=frames_pil
        ))
        sid_inline = res_inline["session_id"]

        prompt_req_inline = dict(type="add_prompt", session_id=sid_inline, frame_index=0, obj_id=0)
        if prompt_text is not None:
            prompt_req_inline["text"] = prompt_text
        if bbox is not None:
            prompt_req_inline["bounding_boxes"] = [bbox]
            prompt_req_inline["bounding_box_labels"] = [1]

        self.predictor.handle_request(prompt_req_inline)

        session_inline = self.predictor._get_session(sid_inline)
        inference_state = session_inline["state"]
        tracker_states = inference_state["tracker_inference_states"]
        if len(tracker_states) == 0:
            raise RuntimeError("No tracker state found after adding prompt to inline session!")
        tracker_state = tracker_states[0]

        tensors_rgb = []
        for f_bgr in frames_bgr:
            f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
            t_rgb = torch.from_numpy(f_rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
            tensors_rgb.append(t_rgb)

        prev_logits = tracker_state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(self.device).float()

        self.predictor.model.tracker.propagate_in_video_preflight(
            tracker_state, run_mem_encoder=True
        )

        with torch.inference_mode():
            for frame_idx in range(1, chunk_length):
                prev_tensor = tensors_rgb[frame_idx - 1]
                curr_tensor = tensors_rgb[frame_idx]

                self.predictor.model._prepare_backbone_feats(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    reverse=False
                )

                _, _, h_mask, w_mask = prev_logits.shape
                flow = self.raft._compute_raft_flow(prev_tensor.unsqueeze(0), curr_tensor.unsqueeze(0)).squeeze(0)
                flow_downscaled = torch.nn.functional.interpolate(
                    flow.unsqueeze(0), size=(h_mask, w_mask), mode="bilinear", align_corners=False
                ).squeeze(0)
                flow_downscaled[0] *= (w_mask / width)
                flow_downscaled[1] *= (h_mask / height)
                
                warped_logits = self.raft._warp_frame(prev_logits.squeeze(0), flow_downscaled)
                warped_logits = warped_logits.unsqueeze(0)

                dummy_point_inputs = {
                    "point_coords": torch.zeros(1, 1, 2, device=self.device),
                    "point_labels": -torch.ones(1, 1, dtype=torch.int32, device=self.device)
                }

                current_out, _ = self.predictor.model.tracker._run_single_frame_inference(
                    inference_state=tracker_state,
                    output_dict=tracker_state["output_dict"],
                    frame_idx=frame_idx,
                    batch_size=1,
                    is_init_cond_frame=False,
                    point_inputs=dummy_point_inputs,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=warped_logits,
                )

                tracker_state["output_dict"]["non_cond_frame_outputs"][frame_idx] = current_out
                self.predictor.model.tracker._add_output_per_object(
                    tracker_state, frame_idx, current_out, "non_cond_frame_outputs"
                )
                tracker_state["frames_already_tracked"][frame_idx] = {"reverse": False}

                prev_logits = current_out["pred_masks"].to(self.device).float()

        final_masks = []
        stabilized_soft = []
        for i in range(chunk_length):
            storage_key = "cond_frame_outputs" if i == 0 else "non_cond_frame_outputs"
            out = tracker_state["output_dict"][storage_key][i]
            
            logits_gpu = out["pred_masks_high_res"].to(self.device) if "pred_masks_high_res" in out else out["pred_masks"].to(self.device)
            logits_resized = torch.nn.functional.interpolate(
                logits_gpu,
                size=(height, width),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)
            
            prob = torch.sigmoid(logits_resized).cpu().numpy()
            stabilized_soft.append(prob)
            final_masks.append((prob > 0.5).astype(np.uint8) * 255)

        self.predictor.handle_request(dict(type="close_session", session_id=sid_inline))

        print(f"[DEBUG] Raw SAM3 masks sums for first 5 frames: {[np.sum(vanilla_masks[i]) for i in range(min(5, chunk_length))]}")
        print(f"[DEBUG] Final masks sums for first 5 frames: {[np.sum(final_masks[i]) for i in range(min(5, chunk_length))]}")

        return vanilla_masks, final_masks, sam_soft, stabilized_soft

def evaluate_mask_sequence(frames_bgr, hard_masks, soft_masks, raft):
    """
    Computes objective tracking stability metrics for a mask sequence.
    Supports both binarized (hard) and continuous probability (soft) mask lists.
    """
    num_frames = len(frames_bgr)
    if num_frames < 2:
        return {}
    
    warping_ious = []
    soft_warping_ious = []
    consecutive_ious = []
    soft_consecutive_ious = []
    perimeters = []
    
    for m in hard_masks:
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(cnt, True) for cnt in contours)
        perimeters.append(perimeter)
        
    perimeter_diffs = []
    for i in range(num_frames - 1):
        perimeter_diffs.append(abs(perimeters[i+1] - perimeters[i]))
        
    for i in tqdm(range(num_frames - 1), desc="Calculating Flow Alignment"):
        m1_h = hard_masks[i]
        m2_h = hard_masks[i+1]
        m1_s = soft_masks[i]
        m2_s = soft_masks[i+1]
        
        intersection_h = np.logical_and(m1_h > 127, m2_h > 127).sum()
        union_h = np.logical_or(m1_h > 127, m2_h > 127).sum()
        consec_iou_h = intersection_h / max(union_h, 1e-8)
        consecutive_ious.append(consec_iou_h)
        
        intersection_s = np.minimum(m1_s, m2_s).sum()
        union_s = np.maximum(m1_s, m2_s).sum()
        consec_iou_s = intersection_s / max(union_s, 1e-8)
        soft_consecutive_ious.append(consec_iou_s)
        
        f1_rgb = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2RGB)
        f2_rgb = cv2.cvtColor(frames_bgr[i+1], cv2.COLOR_BGR2RGB)
        
        t_f1 = torch.from_numpy(f1_rgb).permute(2, 0, 1).float().div(255.0).to(raft.device)
        t_f2 = torch.from_numpy(f2_rgb).permute(2, 0, 1).float().div(255.0).to(raft.device)
        
        with torch.no_grad():
            flow = raft._compute_raft_flow(t_f1.unsqueeze(0), t_f2.unsqueeze(0)).squeeze(0)
            
            t_m1_h = torch.from_numpy(m1_h).unsqueeze(0).float().div(255.0).to(raft.device)
            m1_h_warped = raft._warp_frame(t_m1_h, flow, t=1.0)
            m1_h_warped_np = (m1_h_warped.squeeze(0).cpu().numpy() > 0.5).astype(np.uint8) * 255
            
            t_m1_s = torch.from_numpy(m1_s).unsqueeze(0).to(raft.device)
            m1_s_warped = raft._warp_frame(t_m1_s, flow, t=1.0)
            m1_s_warped_np = m1_s_warped.squeeze(0).cpu().numpy()
            
        warp_inter_h = np.logical_and(m1_h_warped_np > 127, m2_h > 127).sum()
        warp_union_h = np.logical_or(m1_h_warped_np > 127, m2_h > 127).sum()
        warp_iou_h = warp_inter_h / max(warp_union_h, 1e-8)
        warping_ious.append(warp_iou_h)
        
        warp_inter_s = np.minimum(m1_s_warped_np, m2_s).sum()
        warp_union_s = np.maximum(m1_s_warped_np, m2_s).sum()
        warp_iou_s = warp_inter_s / max(warp_union_s, 1e-8)
        soft_warping_ious.append(warp_iou_s)
        
    return {
        "mean_warping_iou": float(np.mean(warping_ious)),
        "std_warping_iou": float(np.std(warping_ious)),
        "mean_consec_iou": float(np.mean(consecutive_ious)),
        "std_consec_iou": float(np.std(consecutive_ious)),
        "mean_soft_consec_iou": float(np.mean(soft_consecutive_ious)),
        "std_soft_consec_iou": float(np.std(soft_consecutive_ious)),
        "mean_soft_warping_iou": float(np.mean(soft_warping_ious)),
        "std_soft_warping_iou": float(np.std(soft_warping_ious)),
        "mean_perimeter_diff": float(np.mean(perimeter_diffs)),
        "std_perimeter_diff": float(np.std(perimeter_diffs)),
    }

def test_simple_video_batch(video_path, out_path, prompt_text="One girl", batch_size=100):

    predictor = build_sam3_video_predictor(
        offload_video_to_cpu=True,
        async_loading_frames=True,
    )
    hybrid_loop = HybridSam3MotionLoop(
        video_predictor=predictor,
        raft_compensator=RaftMotionCompensator(device="cuda"),
    )

    total_frames, num_keyframes, width, height, duration, fps = metadata(video_path)
    cap = cv2.VideoCapture(video_path)
    writer = ffmpeg_pipe(out_path, width * 2, height, fps)

    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing Batches")

    all_frames = []
    all_vanilla_masks = []
    all_hybrid_masks = []
    all_vanilla_soft_masks = []
    all_hybrid_soft_masks = []

    while frame_count < total_frames:
        frames_bgr = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret: break
            frames_bgr.append(frame)

        if not frames_bgr:
            break

        chunk_length = len(frames_bgr)

        frames_pil = []
        for f in frames_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames_pil.append(Image.fromarray(rgb))

        print(f"\nProcessing Batch (Frames {frame_count} to {frame_count+chunk_length})...")

        vanilla_masks, final_masks, sam_soft, stabilized_soft = hybrid_loop.process_batch(
            frames_pil=frames_pil,
            frames_bgr=frames_bgr,
            prompt_text=prompt_text,
        )
        
        torch.cuda.empty_cache()

        all_frames.extend(frames_bgr)
        all_vanilla_masks.extend(vanilla_masks)
        all_hybrid_masks.extend(final_masks)
        all_vanilla_soft_masks.extend(sam_soft)
        all_hybrid_soft_masks.extend(stabilized_soft)

        for i in range(chunk_length):
            mask_vanilla = np.zeros_like(frames_bgr[i])
            mask_vanilla[:, :, 1] = vanilla_masks[i]
            overlay_vanilla = cv2.addWeighted(frames_bgr[i], 0.7, mask_vanilla, 0.3, 0)
            cv2.putText(overlay_vanilla, "Vanilla SAM3", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            
            mask_hybrid = np.zeros_like(frames_bgr[i])
            mask_hybrid[:, :, 2] = final_masks[i]
            overlay_hybrid = cv2.addWeighted(frames_bgr[i], 0.7, mask_hybrid, 0.3, 0)
            cv2.putText(overlay_hybrid, "Hybrid (SAM3 + RAFT)", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
            
            sbs_frame = np.hstack((overlay_vanilla, overlay_hybrid))
            writer.stdin.write(sbs_frame.tobytes())

        frame_count += chunk_length
        pbar.update(chunk_length)

    cap.release()
    writer.stdin.close()
    writer.wait()
    print("Side-by-side comparison video complete!")

    print("\nEvaluating quantitative tracking stability metrics...")
    vanilla_results = evaluate_mask_sequence(all_frames, all_vanilla_masks, all_vanilla_soft_masks, hybrid_loop.raft)
    hybrid_results = evaluate_mask_sequence(all_frames, all_hybrid_masks, all_hybrid_soft_masks, hybrid_loop.raft)

    diff_pixel_count = 0
    total_pixels = len(all_frames) * height * width
    for m_v, m_h in zip(all_vanilla_masks, all_hybrid_masks):
        diff_pixel_count += np.sum(m_v != m_h)
    
    print("\n" + "="*75)
    print("           QUANTITATIVE TRACKING STABILITY REPORT")
    print("="*75)
    print(f"{'Metric':<35} | {'Vanilla SAM3':<15} | {'Hybrid SAM3+RAFT':<18}")
    print("-"*75)
    print(f"{'Mean Warping IoU (higher=best)':<35} | {vanilla_results['mean_warping_iou']:.6f}          | {hybrid_results['mean_warping_iou']:.6f}")
    print(f"{'Std Warping IoU (lower=best)':<35} | {vanilla_results['std_warping_iou']:.6f}          | {hybrid_results['std_warping_iou']:.6f}")
    print(f"{'Mean Consecutive IoU (higher=best)':<35} | {vanilla_results['mean_consec_iou']:.6f}          | {hybrid_results['mean_consec_iou']:.6f}")
    print(f"{'Std Consecutive IoU (lower=best)':<35} | {vanilla_results['std_consec_iou']:.6f}          | {hybrid_results['std_consec_iou']:.6f}")
    print(f"{'Mean Perimeter Jitter (lower=best)':<35} | {vanilla_results['mean_perimeter_diff']:.2f}           | {hybrid_results['mean_perimeter_diff']:.2f}")
    print(f"{'Std Perimeter Jitter (lower=best)':<35} | {vanilla_results['std_perimeter_diff']:.2f}           | {hybrid_results['std_perimeter_diff']:.2f}")
    print("-"*75)
    print(f"{'Mean Soft Warping IoU (higher=best)':<35} | {vanilla_results['mean_soft_warping_iou']:.6f}          | {hybrid_results['mean_soft_warping_iou']:.6f}")
    print(f"{'Std Soft Warping IoU (lower=best)':<35} | {vanilla_results['std_soft_warping_iou']:.6f}          | {hybrid_results['std_soft_warping_iou']:.6f}")
    print(f"{'Mean Soft Consecutive IoU (hi=best)':<35} | {vanilla_results['mean_soft_consec_iou']:.6f}          | {hybrid_results['mean_soft_consec_iou']:.6f}")
    print(f"{'Std Soft Consecutive IoU (lo=best)':<35} | {vanilla_results['std_soft_consec_iou']:.6f}          | {hybrid_results['std_soft_consec_iou']:.6f}")
    print("="*75)
    print(f"[DEBUG] Total binarized mask pixels: {total_pixels}")
    print(f"[DEBUG] Differing binarized mask pixels: {diff_pixel_count} ({diff_pixel_count / total_pixels * 100:.6f}%)")
    print("="*75)

test_simple_video_batch(
    video_path="test.mp4",
    out_path="output.mp4",
    prompt_text="One girl",
    batch_size=100,
)

# ===========================================================================
#            QUANTITATIVE TRACKING STABILITY REPORT
# ===========================================================================
# Metric                              | Vanilla SAM3    | Hybrid SAM3+RAFT
# ---------------------------------------------------------------------------
# Mean Warping IoU (higher=best)      | 0.996138          | 0.996198
# Std Warping IoU (lower=best)        | 0.001417          | 0.001358
# Mean Consecutive IoU (higher=best)  | 0.997778          | 0.997840
# Std Consecutive IoU (lower=best)    | 0.001145          | 0.001085
# Mean Perimeter Jitter (lower=best)  | 2.89           | 11.41
# Std Perimeter Jitter (lower=best)   | 3.17           | 18.24
# ---------------------------------------------------------------------------
# Mean Soft Warping IoU (higher=best) | 0.992781          | 0.993976
# Std Soft Warping IoU (lower=best)   | 0.001172          | 0.001645
# Mean Soft Consecutive IoU (hi=best) | 0.997778          | 0.997697
# Std Soft Consecutive IoU (lo=best)  | 0.001145          | 0.001956
# ===========================================================================
# [DEBUG] Total binarized mask pixels: 89128960
# [DEBUG] Differing binarized mask pixels: 49015 (0.054993%)
# ===========================================================================

# ===========================================================================
#            QUANTITATIVE TRACKING STABILITY REPORT
# ===========================================================================
# Metric                              | Vanilla SAM3    | Hybrid SAM3+RAFT
# ---------------------------------------------------------------------------
# Mean Warping IoU (higher=best)      | 0.993838          | 0.993795
# Std Warping IoU (lower=best)        | 0.003816          | 0.003412
# Mean Consecutive IoU (higher=best)  | 0.995593          | 0.995479
# Std Consecutive IoU (lower=best)    | 0.003007          | 0.002649
# Mean Perimeter Jitter (lower=best)  | 7.99           | 4.14
# Std Perimeter Jitter (lower=best)   | 12.00           | 4.88
# ---------------------------------------------------------------------------
# Mean Soft Warping IoU (higher=best) | 0.991230          | 0.991679
# Std Soft Warping IoU (lower=best)   | 0.003379          | 0.003716
# Mean Soft Consecutive IoU (hi=best) | 0.995593          | 0.994691
# Std Soft Consecutive IoU (lo=best)  | 0.003007          | 0.003739
# ===========================================================================
# [DEBUG] Total binarized mask pixels: 634388480
# [DEBUG] Differing binarized mask pixels: 696522 (0.109794%)
# ===========================================================================