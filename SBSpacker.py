import cv2, torch, subprocess, numpy as np, json, logging
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

def eye_frames(video_path, start_frame, num_frames):
    cap_chunk = cv2.VideoCapture(video_path)
    cap_chunk.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_l, frames_r = [], []
    for _ in range(num_frames):
        ret, frame = cap_chunk.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mid = frame.shape[1] // 2
        frames_l.append(Image.fromarray(frame[:, :mid]))
        frames_r.append(Image.fromarray(frame[:, mid:]))
    cap_chunk.release()
    return frames_l, frames_r

def ffmpeg_pipe(out_path, width, height, fps):
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-c:v', 'hevc_nvenc', '-preset', 'fast', '-cq', '20',
        '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
        '-color_trc', 'bt709', '-color_range', 'tv', out_path
    ]
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

def process_single_eye_video(
    video_path, out_path, hybrid_loop, total_frames, fps, width, height, bbox
):

    res = hybrid_loop.predictor.handle_request(dict(
        type="start_session",
        resource_path=video_path
    ))
    sid = res["session_id"]
    prompt_req = dict(type="add_prompt", session_id=sid, frame_index=0, obj_id=0)
    if bbox is not None:
        prompt_req["bounding_boxes"] = [bbox]
        prompt_req["bounding_box_labels"] = [1]
    hybrid_loop.predictor.handle_request(prompt_req)

    session = hybrid_loop.predictor._get_session(sid)
    inference_state = session["state"]
    tracker_state = inference_state["tracker_inference_states"][0]

    hybrid_loop.predictor.model.tracker.propagate_in_video_preflight(
        tracker_state, run_mem_encoder=True
    )

    cap = cv2.VideoCapture(video_path)
    writer = ffmpeg_pipe(out_path, width, height, fps)

    ret, prev_frame_bgr = cap.read()
    prev_frame_rgb = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2RGB)
    prev_tensor = torch.from_numpy(prev_frame_rgb).permute(2, 0, 1).float().div(255.0).to(hybrid_loop.device)

    prev_logits = tracker_state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(hybrid_loop.device).float()
    batch_size = len(tracker_state["obj_ids"])

    with torch.inference_mode():
        for frame_idx in tqdm(range(1, total_frames), desc="Processing Single Eye"):
            ret, curr_frame_bgr = cap.read()
            if not ret: break

            curr_frame_rgb = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2RGB)
            curr_tensor = torch.from_numpy(curr_frame_rgb).permute(2, 0, 1).float().div(255.0).to(hybrid_loop.device)

            hybrid_loop.predictor.model._prepare_backbone_feats(
                inference_state=inference_state,
                frame_idx=frame_idx,
                reverse=False
            )

            _, _, h_mask, w_mask = prev_logits.shape
            flow = hybrid_loop.raft._compute_raft_flow(prev_tensor.unsqueeze(0), curr_tensor.unsqueeze(0)).squeeze(0)
            flow_downscaled = torch.nn.functional.interpolate(
                flow.unsqueeze(0), size=(h_mask, w_mask), mode="bilinear", align_corners=False
            ).squeeze(0)
            flow_downscaled[0] *= (w_mask / width)
            flow_downscaled[1] *= (h_mask / height)

            warped_logits = hybrid_loop.raft._warp_frame(prev_logits, flow_downscaled)

            dummy_point_inputs = {
                "point_coords": torch.zeros(batch_size, 1, 2, device=hybrid_loop.device),
                "point_labels": -torch.ones(batch_size, 1, dtype=torch.int32, device=hybrid_loop.device)
            }

            current_out, _ = hybrid_loop.predictor.model.tracker._run_single_frame_inference(
                inference_state=tracker_state,
                output_dict=tracker_state["output_dict"],
                frame_idx=frame_idx,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=dummy_point_inputs,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=warped_logits,
            )

            tracker_state["output_dict"]["non_cond_frame_outputs"][frame_idx] = current_out
            hybrid_loop.predictor.model.tracker._add_output_per_object(
                tracker_state, frame_idx, current_out, "non_cond_frame_outputs"
            )
            tracker_state["frames_already_tracked"][frame_idx] = {"reverse": False}

            prev_logits = current_out["pred_masks"].to(hybrid_loop.device).float()
            prev_tensor = curr_tensor

            logits_gpu = current_out["pred_masks_high_res"] if "pred_masks_high_res" in current_out else current_out["pred_masks"]
            if logits_gpu.shape[0] > 0:
                logits_gpu = torch.max(logits_gpu, dim=0, keepdim=True).values
            else:
                logits_gpu = torch.zeros((1, 1, height, width), device=hybrid_loop.device)
            logits_resized = torch.nn.functional.interpolate(
                logits_gpu.to(hybrid_loop.device),
                size=(height, width),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)
            
            prob = torch.sigmoid(logits_resized).cpu().numpy()
            mask_bin = ((prob > 0.5).astype(np.uint8) * 255)

            mask_rgb = np.zeros_like(curr_frame_bgr)
            mask_rgb[:, :, 2] = mask_bin
            overlay = cv2.addWeighted(curr_frame_bgr, 0.7, mask_rgb, 0.3, 0)
            writer.stdin.write(overlay.tobytes())

            if frame_idx % 10 == 0: 
                torch.cuda.empty_cache()

    cap.release()
    writer.stdin.close()
    writer.wait()
    hybrid_loop.predictor.handle_request(dict(type="close_session", session_id=sid))

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
        if pt_frame.ndim == 3:
            C, H, W = pt_frame.shape
            flow_scaled = flow * t
            y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
            x_norm = 2.0 * (x + flow_scaled[0]) / max(W - 1, 1) - 1.0
            y_norm = 2.0 * (y + flow_scaled[1]) / max(H - 1, 1) - 1.0
            grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
            return F.grid_sample(
                pt_frame.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False
            ).squeeze(0)
        elif pt_frame.ndim == 4:
            N, C, H, W = pt_frame.shape
            flow_scaled = flow * t
            y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
            x_norm = 2.0 * (x + flow_scaled[0]) / max(W - 1, 1) - 1.0
            y_norm = 2.0 * (y + flow_scaled[1]) / max(H - 1, 1) - 1.0
            grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
            grid = grid.expand(N, -1, -1, -1)
            return F.grid_sample(
                pt_frame, grid, mode='bilinear', padding_mode='border', align_corners=False
            )
        else:
            raise ValueError(f"Unexpected pt_frame dimensions: {pt_frame.ndim}")

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

    def interpolate_video(self, input_video, output_video, src_fps, dst_fps, width, height, encoder_opts, read_cmd=None):
        self._load_model()
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_video]
        duration = float(subprocess.check_output(cmd).decode().strip())
        num_src_frames = int(duration * src_fps)
        num_dst_frames = int(duration * dst_fps)
        step = src_fps / dst_fps
        if read_cmd is None:
            read_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", input_video, "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"]
        write_cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(dst_fps), "-i", "-"] + encoder_opts + [output_video]
        reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=width*height*3*2)
        writer = subprocess.Popen(write_cmd, stdin=subprocess.PIPE, bufsize=width*height*3*2)
        def read_frame():
            raw = reader.stdout.read(width * height * 3)
            if not raw: return None
            frame = np.frombuffer(raw, dtype=np.uint8).copy().reshape((height, width, 3))
            return torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).to(self.device)
        frame_a = read_frame()
        if frame_a is None:
            raise RuntimeError("Failed to read any frames from FFmpeg!")
        frame_b = read_frame()
        current_idx, out_idx = 0, 0
        with torch.no_grad():
            with tqdm(total=num_dst_frames, unit="frame") as pbar:
                while True:
                    t_idx = out_idx * step
                    while current_idx < int(t_idx) and frame_b is not None:
                        frame_a = frame_b
                        frame_b = read_frame()
                        current_idx += 1
                    if frame_b is None:
                        if int(t_idx) > current_idx: break
                        out_frame = frame_a
                    else:
                        t = float(t_idx - current_idx)
                        if t <= 1e-6: out_frame = frame_a
                        else:
                            flow_fwd = self._compute_raft_flow(frame_a.unsqueeze(0), frame_b.unsqueeze(0)).squeeze(0)
                            flow_bwd = self._compute_raft_flow(frame_b.unsqueeze(0), frame_a.unsqueeze(0)).squeeze(0)
                            
                            warp_a = self._warp_frame(frame_a, flow_fwd, t)
                            warp_b = self._warp_frame(frame_b, flow_bwd, 1.0 - t)
                            mag_fwd = torch.norm(flow_fwd, dim=0, keepdim=True)
                            mag_bwd = torch.norm(flow_bwd, dim=0, keepdim=True)
                            weight_a = torch.exp(-mag_fwd * 0.1) * (1.0 - t)
                            weight_b = torch.exp(-mag_bwd * 0.1) * t
                            Z = weight_a + weight_b
                            mask = (Z > 1e-4).float()
                            norm_a = torch.where(mask > 0, weight_a / (Z + 1e-8), 1.0 - t)
                            norm_b = torch.where(mask > 0, weight_b / (Z + 1e-8), t)
                            out_frame = torch.clamp(warp_a * norm_a + warp_b * norm_b, 0.0, 1.0)
                            
                    writer.stdin.write((out_frame.cpu().permute(1,2,0).numpy() * 255).astype(np.uint8).tobytes())
                    out_idx += 1
                    pbar.update(1)
        reader.stdout.close()
        writer.stdin.close()
        writer.wait()
        reader.wait()

class HybridSam3MotionLoop:

    def __init__(self, video_predictor=None, raft_compensator=None, target_res=(256, 256)):

        self.predictor = build_sam3_video_predictor(
        offload_video_to_cpu=True,
        async_loading_frames=True) or video_predictor

        self.raft = raft_compensator or RaftMotionCompensator()
        self.device = self.raft.device
        self.target_res = target_res

    def process_batch(self, frames_pil, frames_bgr, prompt_text=None, bbox=None):

        self.raft._load_model()
        height, width = frames_bgr[0].shape[:2]
        chunk_length = len(frames_pil)

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
        batch_size = len(tracker_state["obj_ids"])

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
                
                warped_logits = self.raft._warp_frame(prev_logits, flow_downscaled)

                dummy_point_inputs = {
                    "point_coords": torch.zeros(batch_size, 1, 2, device=self.device),
                    "point_labels": -torch.ones(batch_size, 1, dtype=torch.int32, device=self.device)
                }

                current_out, _ = self.predictor.model.tracker._run_single_frame_inference(
                    inference_state=tracker_state,
                    output_dict=tracker_state["output_dict"],
                    frame_idx=frame_idx,
                    batch_size=batch_size,
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

        for i in range(chunk_length):
            storage_key = "cond_frame_outputs" if i == 0 else "non_cond_frame_outputs"
            out = tracker_state["output_dict"][storage_key][i]
            
            logits_gpu = out["pred_masks_high_res"].to(self.device) if "pred_masks_high_res" in out else out["pred_masks"].to(self.device)
            if logits_gpu.shape[0] > 0:
                logits_gpu = torch.max(logits_gpu, dim=0, keepdim=True).values
            else:
                logits_gpu = torch.zeros((1, 1, height, width), device=self.device)
            logits_resized = torch.nn.functional.interpolate(
                logits_gpu,
                size=(height, width),
                mode="bilinear",
                align_corners=False
            ).squeeze(0).squeeze(0)
            
            prob = torch.sigmoid(logits_resized).cpu().numpy()
            final_masks.append((prob > 0.5).astype(np.uint8) * 255)

        self.predictor.handle_request(dict(type="close_session", session_id=sid_inline))

        print(f"[DEBUG] Final masks sums for first 5 frames: {[np.sum(final_masks[i]) for i in range(min(5, chunk_length))]}")

        return final_masks

class AlphaCornerPacker:

    def __init__(self, scale_factor=0.40, padding=0):
        self.scale = scale_factor
        self.padding = padding
        self.vignette_cache = None

    def _get_circular_vignette(self, w, h):
   
        if self.vignette_cache is not None and self.vignette_cache.shape == (h, w):
            return self.vignette_cache

        vignette = np.zeros((h, w), dtype=np.float32)
        
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 2 

        cv2.circle(vignette, center, radius, 1.0, -1, cv2.LINE_AA)
        
        self.vignette_cache = cv2.GaussianBlur(vignette, (15, 15), 0)
        
        return self.vignette_cache

    def pack_frame(self, sbs_rgb, mask_l, mask_r):

        H, SBS_W, C = sbs_rgb.shape
        W = SBS_W // 2

        if mask_l.dtype != np.uint8:
            mask_l = (mask_l * 255).astype(np.uint8)
            mask_r = (mask_r * 255).astype(np.uint8)

        target_w = int(W * self.scale)
        target_h = int(H * self.scale)
        
        mask_l_small = cv2.resize(mask_l, (target_w, target_h), interpolation=cv2.INTER_AREA)
        lh, lw = mask_l_small.shape
        mask_r_small = cv2.resize(mask_r, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rh, rw = mask_r_small.shape
        
        vignette = self._get_circular_vignette(target_w, target_h)
        
        mask_l_vignette = (mask_l_small * vignette).astype(np.uint8)
        mask_r_vignette = (mask_r_small * vignette).astype(np.uint8)

        packed_frame = sbs_rgb.copy()

        h_half = target_h // 2
        top_half_mask = mask_l_vignette[:h_half, :]
        bottom_half_mask = mask_l_vignette[h_half:h_half*2, :]

        w_half = target_w // 2
        q_tl_mask = mask_r_vignette[:h_half, :w_half]
        q_tr_mask = mask_r_vignette[:h_half, w_half:w_half*2]
        q_bl_mask = mask_r_vignette[h_half:h_half*2, :w_half]
        q_br_mask = mask_r_vignette[h_half:h_half*2, w_half:w_half*2]

        def blend_red_mask(roi, mask_1ch):
            alpha = mask_1ch.astype(np.float32) / 255.0
            alpha = np.expand_dims(alpha, axis=2)
            
            red_color = np.array([0, 0, 255], dtype=np.float32)
            blended = (1.0 - alpha) * roi.astype(np.float32) + alpha * red_color
            return blended.astype(np.uint8)

        y1_top = self.padding
        y2_top = y1_top + h_half
        x1_mid = (SBS_W // 2) - (target_w // 2)
        x2_mid = x1_mid + target_w
        
        packed_frame[y1_top:y2_top, x1_mid:x2_mid] = blend_red_mask(
            packed_frame[y1_top:y2_top, x1_mid:x2_mid], bottom_half_mask
        )

        y1_bot = H - self.padding - h_half
        y2_bot = y1_bot + h_half
        packed_frame[y1_bot:y2_bot, x1_mid:x2_mid] = blend_red_mask(
            packed_frame[y1_bot:y2_bot, x1_mid:x2_mid], top_half_mask
        )

        y1_tr = self.padding
        y2_tr = y1_tr + h_half
        x1_tr = SBS_W - self.padding - w_half
        x2_tr = SBS_W - self.padding
        packed_frame[y1_tr:y2_tr, x1_tr:x2_tr] = blend_red_mask(
            packed_frame[y1_tr:y2_tr, x1_tr:x2_tr], q_bl_mask
        )

        y1_tl_l = self.padding
        y2_tl_l = y1_tl_l + h_half
        x1_tl_l = self.padding
        x2_tl_l = self.padding + w_half
        packed_frame[y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l] = blend_red_mask(
            packed_frame[y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l], q_br_mask
        )

        y1_br_r = H - self.padding - h_half
        y2_br_r = y1_br_r + h_half
        x1_br_r = SBS_W - self.padding - w_half
        x2_br_r = SBS_W - self.padding
        packed_frame[y1_br_r:y2_br_r, x1_br_r:x2_br_r] = blend_red_mask(
            packed_frame[y1_br_r:y2_br_r, x1_br_r:x2_br_r], q_tl_mask
        )

        y1_bl_l = H - self.padding - h_half
        y2_bl_l = y1_bl_l + h_half
        x1_bl_l = self.padding
        x2_bl_l = self.padding + w_half
        packed_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l] = blend_red_mask(
            packed_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l], q_tr_mask
        )

        return packed_frame

def process_video_in_batches(
    video_path, 
    out_path, 
    left_bbox=None, 
    right_bbox=None,
    prompt_text=None,
    batch_size=100,
    matte_size=0.4,
    use_class_process_batch=False
):

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
    writer = ffmpeg_pipe(out_path, width, height, fps)
    half_w = width // 2
    
    packer = AlphaCornerPacker(scale_factor=matte_size)
    hybrid_loop.raft._load_model()

    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing SBS Batches")

    def propagate_eye(eye_frames_bgr, bbox):
        chunk_len = len(eye_frames_bgr)
        h, w = eye_frames_bgr[0].shape[:2]
        eye_frames_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in eye_frames_bgr]
        
        res = hybrid_loop.predictor.handle_request(dict(
            type="start_session",
            resource_path=eye_frames_pil
        ))
        sid = res["session_id"]
        
        prompt_req = dict(type="add_prompt", session_id=sid, frame_index=0, obj_id=0)
        if prompt_text is not None:
            prompt_req["text"] = prompt_text
        if bbox is not None:
            prompt_req["bounding_boxes"] = [bbox]
            prompt_req["bounding_box_labels"] = [1]
        hybrid_loop.predictor.handle_request(prompt_req)
        
        session = hybrid_loop.predictor._get_session(sid)
        inference_state = session["state"]
        tracker_state = inference_state["tracker_inference_states"][0]
        
        hybrid_loop.predictor.model.tracker.propagate_in_video_preflight(
            tracker_state, run_mem_encoder=True
        )
        
        tensors_rgb = []
        for f_bgr in eye_frames_bgr:
            f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
            t_rgb = torch.from_numpy(f_rgb).permute(2, 0, 1).float().div(255.0).to(hybrid_loop.device)
            tensors_rgb.append(t_rgb)
            
        prev_logits = tracker_state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(hybrid_loop.device).float()
        batch_size = len(tracker_state["obj_ids"])
        
        eye_masks = []
        
        out_f0 = tracker_state["output_dict"]["cond_frame_outputs"][0]
        logits_f0 = out_f0["pred_masks_high_res"] if "pred_masks_high_res" in out_f0 else out_f0["pred_masks"]
        if logits_f0.shape[0] > 0:
            logits_f0 = torch.max(logits_f0, dim=0, keepdim=True).values
        else:
            logits_f0 = torch.zeros((1, 1, h, w), device=hybrid_loop.device)
        logits_f0_resized = torch.nn.functional.interpolate(
            logits_f0.to(hybrid_loop.device),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        ).squeeze(0).squeeze(0)

        prob_f0 = torch.sigmoid(logits_f0_resized).cpu().numpy()
        eye_masks.append((prob_f0 > 0.5).astype(np.uint8) * 255)
        
        with torch.inference_mode():
            for frame_idx in range(1, chunk_len):
                prev_tensor = tensors_rgb[frame_idx - 1]
                curr_tensor = tensors_rgb[frame_idx]
                
                hybrid_loop.predictor.model._prepare_backbone_feats(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    reverse=False
                )
                
                _, _, h_mask, w_mask = prev_logits.shape
                flow = hybrid_loop.raft._compute_raft_flow(prev_tensor.unsqueeze(0), curr_tensor.unsqueeze(0)).squeeze(0)
                flow_downscaled = torch.nn.functional.interpolate(
                    flow.unsqueeze(0), size=(h_mask, w_mask), mode="bilinear", align_corners=False
                ).squeeze(0)
                flow_downscaled[0] *= (w_mask / w)
                flow_downscaled[1] *= (h_mask / h)
                
                warped_logits = hybrid_loop.raft._warp_frame(prev_logits, flow_downscaled)
                
                dummy_point_inputs = {
                    "point_coords": torch.zeros(batch_size, 1, 2, device=hybrid_loop.device),
                    "point_labels": -torch.ones(batch_size, 1, dtype=torch.int32, device=hybrid_loop.device)
                }
                
                current_out, _ = hybrid_loop.predictor.model.tracker._run_single_frame_inference(
                    inference_state=tracker_state,
                    output_dict=tracker_state["output_dict"],
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=dummy_point_inputs,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=warped_logits,
                )
                
                tracker_state["output_dict"]["non_cond_frame_outputs"][frame_idx] = current_out
                hybrid_loop.predictor.model.tracker._add_output_per_object(
                    tracker_state, frame_idx, current_out, "non_cond_frame_outputs"
                )
                tracker_state["frames_already_tracked"][frame_idx] = {"reverse": False}
                
                prev_logits = current_out["pred_masks"].to(hybrid_loop.device).float()
                
                logits_gpu = current_out["pred_masks_high_res"] if "pred_masks_high_res" in current_out else current_out["pred_masks"]
                if logits_gpu.shape[0] > 0:
                    logits_gpu = torch.max(logits_gpu, dim=0, keepdim=True).values
                else:
                    logits_gpu = torch.zeros((1, 1, h, w), device=hybrid_loop.device)
                logits_resized = torch.nn.functional.interpolate(
                    logits_gpu.to(hybrid_loop.device),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0).squeeze(0)
                prob = torch.sigmoid(logits_resized).cpu().numpy()
                eye_masks.append((prob > 0.5).astype(np.uint8) * 255)
                
        hybrid_loop.predictor.handle_request(dict(type="close_session", session_id=sid))
        return eye_masks

    while frame_count < total_frames:
        frames_bgr = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret: break
            frames_bgr.append(frame)
            
        if not frames_bgr:
            break
            
        chunk_length = len(frames_bgr)
        
        frames_l_bgr = [f[:, :half_w] for f in frames_bgr]
        frames_r_bgr = [f[:, half_w:] for f in frames_bgr]

        target_w = int(half_w * matte_size)
        target_h = int(height * matte_size)
        
        frames_l_bgr_small = [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA) for f in frames_l_bgr]
        frames_r_bgr_small = [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA) for f in frames_r_bgr]
        
        left_bbox_small = [
            left_bbox[0] * matte_size,
            left_bbox[1] * matte_size,
            left_bbox[2] * matte_size,
            left_bbox[3] * matte_size
        ] if left_bbox is not None else None
        
        right_bbox_small = [
            right_bbox[0] * matte_size,
            right_bbox[1] * matte_size,
            right_bbox[2] * matte_size,
            right_bbox[3] * matte_size
        ] if right_bbox is not None else None
        
        print(f"\n[SBS] Tracking Left Eye Batch (Frames {frame_count} to {frame_count+chunk_length-1}) at {int(matte_size*100)}% scale...")
        if use_class_process_batch:
            eye_frames_pil_l = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_l_bgr_small]
            masks_l = hybrid_loop.process_batch(
                frames_pil=eye_frames_pil_l,
                frames_bgr=frames_l_bgr_small,
                prompt_text=prompt_text,
                bbox=left_bbox_small
            )
        else:
            masks_l = propagate_eye(frames_l_bgr_small, left_bbox_small)
        torch.cuda.empty_cache()
        
        print(f"[SBS] Tracking Right Eye Batch (Frames {frame_count} to {frame_count+chunk_length-1}) at {int(matte_size*100)}% scale...")
        if use_class_process_batch:
            eye_frames_pil_r = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_r_bgr_small]
            masks_r = hybrid_loop.process_batch(
                frames_pil=eye_frames_pil_r,
                frames_bgr=frames_r_bgr_small,
                prompt_text=prompt_text,
                bbox=right_bbox_small
            )
        else:
            masks_r = propagate_eye(frames_r_bgr_small, right_bbox_small)
        torch.cuda.empty_cache()
        
        for i in range(chunk_length):
            packed_frame = packer.pack_frame(frames_bgr[i], masks_l[i], masks_r[i])
            writer.stdin.write(packed_frame.astype(np.uint8).tobytes())
            
        frame_count += chunk_length
        pbar.update(chunk_length)
        
    cap.release()
    writer.stdin.close()
    writer.wait()
    print("Stereoscopic chunked processing complete!")

process_video_in_batches(
    video_path="video.mp4",
    out_path="video_out.mp4",
    prompt_text="One girl",
    batch_size=100,
    matte_size=0.4, # deovr uses 40%
    use_class_process_batch=False
)
