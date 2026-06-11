import torch, subprocess, numpy as np, json, logging
import torchvision.transforms.functional as TVF
import torch.nn.functional as F
from tqdm import tqdm
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
    num, denom = map(int, fps_str.split('/'))
    fps = num / denom if denom != 0 else 30.0
    f_tot = stream.get('nb_frames')

    if f_tot:
        nb_frames = int(f_tot)
    else:
        nb_frames = int(duration * fps) if duration > 0 else 0

    return nb_frames, num_keyframes, width, height, duration, fps

def eye_frames(video_path, start_frame, num_frames):

    nb_frames, num_keyframes, width, height, duration, fps = metadata(video_path)
    if width is None: return [], []
    start_time = start_frame / fps if fps > 0 else 0

    cmd = [
        'ffmpeg', '-ss', str(start_time), '-i', video_path,
        '-vframes', str(num_frames), '-f', 'image2pipe', '-pix_fmt', 'rgb24',
        '-vcodec', 'rawvideo', '-'
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames_l, frames_r = [], []
    frame_size = width * height * 3
    mid = width // 2

    for _ in range(num_frames):
        raw = process.stdout.read(frame_size)
        if not raw or len(raw) < frame_size: break
        tensor = torch.frombuffer(raw, dtype=torch.uint8).reshape((height, width, 3))
        frames_l.append(TVF.to_pil_image(tensor[:, :mid, :].permute(2, 0, 1)))
        frames_r.append(TVF.to_pil_image(tensor[:, mid:, :].permute(2, 0, 1)))

    process.stdout.close()
    process.wait()
    return frames_l, frames_r

def ffmpeg_pipe(out_path, width, height, fps):

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps),
        '-i', '-', '-c:v', 'hevc_nvenc', '-preset', 'fast', '-cq', '20',
        '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
        '-color_trc', 'bt709', '-color_range', 'tv', out_path
    ]
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

def process_single_eye_video(video_path, out_path, hybrid_loop, total_frames, fps, width, height, bbox):

    res = hybrid_loop.predictor.handle_request(dict(
        type="start_session", resource_path=video_path))

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

    read_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", 
        "-i", video_path, "-f", "image2pipe", "-pix_fmt", "rgb24", 
        "-vcodec", "rawvideo", "-"
    ]
    reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, bufsize=width*height*3*2)
    writer = ffmpeg_pipe(out_path, width, height, fps)

    frame_bytes = width * height * 3
    raw = reader.stdout.read(frame_bytes)
    if not raw or len(raw) < frame_bytes:
        reader.stdout.close()
        writer.stdin.close()
        return

    prev_frame_rgb = torch.frombuffer(raw, dtype=torch.uint8).reshape((height, width, 3))
    prev_tensor = prev_frame_rgb.permute(2, 0, 1).float().div(255.0).to(hybrid_loop.device)

    prev_logits = tracker_state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(hybrid_loop.device).float()
    batch_size = len(tracker_state["obj_ids"])

    with torch.inference_mode():

        for frame_idx in tqdm(range(1, total_frames), desc="Processing Single Eye"):
            raw = reader.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes: break

            curr_frame_rgb = torch.frombuffer(raw, dtype=torch.uint8).reshape((height, width, 3))
            curr_tensor = curr_frame_rgb.permute(2, 0, 1).float().div(255.0).to(hybrid_loop.device)

            hybrid_loop.predictor.model._prepare_backbone_feats(
                inference_state=inference_state,
                frame_idx=frame_idx,
                reverse=False
            )

            _, _, h_mask, w_mask = prev_logits.shape
            flow = hybrid_loop.raft._compute_raft_flow(prev_tensor.unsqueeze(0), curr_tensor.unsqueeze(0)).squeeze(0)
            flow_downscaled = torch.nn.functional.interpolate(flow.unsqueeze(0), size=(h_mask, w_mask), mode="bilinear", align_corners=False).squeeze(0)
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
            hybrid_loop.predictor.model.tracker._add_output_per_object(tracker_state, frame_idx, current_out, "non_cond_frame_outputs")
            tracker_state["frames_already_tracked"][frame_idx] = {"reverse": False}

            prev_logits = current_out["pred_masks"].to(hybrid_loop.device).float()
            prev_tensor = curr_tensor
            logits_gpu = current_out["pred_masks_high_res"] if "pred_masks_high_res" in current_out else current_out["pred_masks"]

            if logits_gpu.shape[0] > 0:
                logits_gpu = torch.max(logits_gpu, dim=0, keepdim=True).values
            else:
                logits_gpu = torch.zeros((1, 1, height, width), device=hybrid_loop.device)

            logits_resized = torch.nn.functional.interpolate(
                logits_gpu.to(hybrid_loop.device), size=(height, width), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            
            prob = torch.sigmoid(logits_resized)
            mask_bin = (prob > 0.5).float()
            overlay = curr_tensor * 0.7
            overlay[0] += mask_bin * 0.3

            overlay_rgb = overlay.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            writer.stdin.write(overlay_rgb.tobytes())

            if frame_idx % 10 == 0: 
                torch.cuda.empty_cache()

    reader.stdout.close()
    reader.wait()
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
            return F.grid_sample(pt_frame.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=False).squeeze(0)
            
        elif pt_frame.ndim == 4:
            N, C, H, W = pt_frame.shape
            flow_scaled = flow * t
            y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
            x_norm = 2.0 * (x + flow_scaled[0]) / max(W - 1, 1) - 1.0
            y_norm = 2.0 * (y + flow_scaled[1]) / max(H - 1, 1) - 1.0
            grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
            grid = grid.expand(N, -1, -1, -1)
            return F.grid_sample(pt_frame, grid, mode='bilinear', padding_mode='border', align_corners=False)
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
                return stabilized_alphas.squeeze(1).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            return stabilized_alphas.permute(0, 2, 3, 1).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy()
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
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
            return torch.from_numpy(frame).permute(2, 0, 1).to(self.device).float().div(255.0)

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
                            
                    writer.stdin.write(out_frame.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).cpu().numpy().tobytes())
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

    def process_batch(self, frames_tensor_rgb, prompt_text=None, bbox=None):

        B, C, height, width = frames_tensor_rgb.shape
        self.raft._load_model()
        
        frames_cpu = (frames_tensor_rgb * 255.0).clamp(0, 255).to(torch.uint8).cpu()
        frames_pil = [TVF.to_pil_image(frames_cpu[i]) for i in range(B)]

        res_inline = self.predictor.handle_request(dict(type="start_session", resource_path=frames_pil))
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
        
        prev_logits = tracker_state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(self.device).float()
        batch_size_obj = len(tracker_state["obj_ids"])

        self.predictor.model.tracker.propagate_in_video_preflight(
            tracker_state, run_mem_encoder=True)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for frame_idx in range(1, B):
                prev_tensor = frames_tensor_rgb[frame_idx - 1]
                curr_tensor = frames_tensor_rgb[frame_idx]

                self.predictor.model._prepare_backbone_feats(
                    inference_state=inference_state, frame_idx=frame_idx, reverse=False)

                _, _, h_mask, w_mask = prev_logits.shape
                flow = self.raft._compute_raft_flow(prev_tensor.unsqueeze(0), curr_tensor.unsqueeze(0)).squeeze(0)
                flow_downscaled = torch.nn.functional.interpolate(flow.unsqueeze(0), size=(h_mask, w_mask), mode="bicubic", align_corners=False).squeeze(0)
                flow_downscaled[0] *= (w_mask / width)
                flow_downscaled[1] *= (h_mask / height)
                
                warped_logits = self.raft._warp_frame(prev_logits.bfloat16(), flow_downscaled).float()

                dummy_point_inputs = {
                    "point_coords": torch.zeros(batch_size_obj, 1, 2, device=self.device),
                    "point_labels": -torch.ones(batch_size_obj, 1, dtype=torch.int32, device=self.device)
                }

                current_out, _ = self.predictor.model.tracker._run_single_frame_inference(
                    inference_state=tracker_state,
                    output_dict=tracker_state["output_dict"],
                    frame_idx=frame_idx,
                    batch_size=batch_size_obj,
                    is_init_cond_frame=False,
                    point_inputs=dummy_point_inputs,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=warped_logits,
                )

                tracker_state["output_dict"]["non_cond_frame_outputs"][frame_idx] = current_out
                self.predictor.model.tracker._add_output_per_object(tracker_state, frame_idx, current_out, "non_cond_frame_outputs")

                tracker_state["frames_already_tracked"][frame_idx] = {"reverse": False}
                prev_logits = current_out["pred_masks"].to(self.device).float()

        final_masks = []

        for i in range(B):

            storage_key = "cond_frame_outputs" if i == 0 else "non_cond_frame_outputs"
            out = tracker_state["output_dict"][storage_key][i]
            logits_gpu = out["pred_masks_high_res"].to(self.device) if "pred_masks_high_res" in out else out["pred_masks"].to(self.device)

            if logits_gpu.shape[0] > 0:
                logits_gpu = torch.max(logits_gpu, dim=0, keepdim=True).values
            else:
                logits_gpu = torch.zeros((1, 1, height, width), device=self.device)
            logits_resized = torch.nn.functional.interpolate(logits_gpu.float(), size=(height, width),
                mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
            
            prob = torch.sigmoid(logits_resized)
            final_masks.append((prob > 0.5).to(torch.uint8))

        self.predictor.handle_request(dict(type="close_session", session_id=sid_inline))

        print(f"[DEBUG] Final masks sums for first 5 frames: {[torch.sum(final_masks[i]).item() for i in range(min(5, B))]}")
        torch.cuda.empty_cache()
        return final_masks

class AlphaCornerPacker:
    def __init__(self, scale_factor=0.40, padding=0, device="cuda"):
        self.scale = scale_factor
        self.padding = padding
        self.device = torch.device(device)
        self.vignette_cache = None

    def _get_circular_vignette(self, w, h):
        if self.vignette_cache is not None and self.vignette_cache.shape == (h, w):
            return self.vignette_cache

        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=torch.float32),
            torch.arange(w, device=self.device, dtype=torch.float32),
            indexing='ij')

        cy, cx = h / 2.0 - 0.5, w / 2.0 - 0.5
        r = torch.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)
        max_r = min(w, h) / 2.0 - 2.0
        t = torch.clamp((max_r - r) / 15.0 + 0.5, 0.0, 1.0)
        vignette = t * t * (3.0 - 2.0 * t)
        self.vignette_cache = vignette
        return self.vignette_cache

    def pack_batch(self, sbs_rgb_tensor, masks_l_tensor, masks_r_tensor):
        B, H, SBS_W, C = sbs_rgb_tensor.shape
        W = SBS_W // 2

        target_w = int(W * self.scale)
        target_h = int(H * self.scale)

        if masks_l_tensor.shape[1] != target_h or masks_l_tensor.shape[2] != target_w:
            masks_l_tensor = F.interpolate(
                masks_l_tensor.unsqueeze(1), size=(target_h, target_w), mode='area'
            ).squeeze(1)
            masks_r_tensor = F.interpolate(
                masks_r_tensor.unsqueeze(1), size=(target_h, target_w), mode='area'
            ).squeeze(1)

        vignette = self._get_circular_vignette(target_w, target_h)
        
        mask_l_vignette = masks_l_tensor * vignette
        mask_r_vignette = masks_r_tensor * vignette

        packed_batch = sbs_rgb_tensor.clone()

        h_half = target_h // 2
        top_half_mask = mask_l_vignette[:, :h_half, :]
        bottom_half_mask = mask_l_vignette[:, h_half:h_half*2, :]

        w_half = target_w // 2
        q_tl_mask = mask_r_vignette[:, :h_half, :w_half]
        q_tr_mask = mask_r_vignette[:, :h_half, w_half:w_half*2]
        q_bl_mask = mask_r_vignette[:, h_half:h_half*2, :w_half]
        q_br_mask = mask_r_vignette[:, h_half:h_half*2, w_half:w_half*2]

        def blend_red_mask_torch(roi, mask_1ch):
            alpha = mask_1ch.unsqueeze(-1)
            red_color = torch.tensor([255.0, 0.0, 0.0], dtype=torch.float32, device=roi.device)
            return ((1.0 - alpha) * roi.float() + alpha * red_color).to(torch.uint8)

        y1_top = self.padding
        y2_top = y1_top + h_half
        x1_mid = (SBS_W // 2) - (target_w // 2)
        x2_mid = x1_mid + target_w
        
        packed_batch[:, y1_top:y2_top, x1_mid:x2_mid] = blend_red_mask_torch(
            packed_batch[:, y1_top:y2_top, x1_mid:x2_mid], bottom_half_mask
        )

        y1_bot = H - self.padding - h_half
        y2_bot = y1_bot + h_half
        packed_batch[:, y1_bot:y2_bot, x1_mid:x2_mid] = blend_red_mask_torch(
            packed_batch[:, y1_bot:y2_bot, x1_mid:x2_mid], top_half_mask
        )

        y1_tr = self.padding
        y2_tr = y1_tr + h_half
        x1_tr = SBS_W - self.padding - w_half
        x2_tr = SBS_W - self.padding
        packed_batch[:, y1_tr:y2_tr, x1_tr:x2_tr] = blend_red_mask_torch(
            packed_batch[:, y1_tr:y2_tr, x1_tr:x2_tr], q_bl_mask
        )

        y1_tl_l = self.padding
        y2_tl_l = y1_tl_l + h_half
        x1_tl_l = self.padding
        x2_tl_l = self.padding + w_half
        packed_batch[:, y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l] = blend_red_mask_torch(
            packed_batch[:, y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l], q_br_mask
        )

        y1_br_r = H - self.padding - h_half
        y2_br_r = y1_br_r + h_half
        x1_br_r = SBS_W - self.padding - w_half
        x2_br_r = SBS_W - self.padding
        packed_batch[:, y1_br_r:y2_br_r, x1_br_r:x2_br_r] = blend_red_mask_torch(
            packed_batch[:, y1_br_r:y2_br_r, x1_br_r:x2_br_r], q_tl_mask
        )

        y1_bl_l = H - self.padding - h_half
        y2_bl_l = y1_bl_l + h_half
        x1_bl_l = self.padding
        x2_bl_l = self.padding + w_half
        packed_batch[:, y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l] = blend_red_mask_torch(
            packed_batch[:, y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l], q_tr_mask
        )

        return packed_batch

    def pack_frame(self, sbs_rgb, mask_l, mask_r):
        is_numpy = isinstance(sbs_rgb, np.ndarray)
        if is_numpy:
            sbs_rgb_t = torch.from_numpy(sbs_rgb).to(self.device).unsqueeze(0)
            mask_l_t = torch.from_numpy(mask_l).to(self.device).float().unsqueeze(0)
            mask_r_t = torch.from_numpy(mask_r).to(self.device).float().unsqueeze(0)
            if mask_l_t.max() > 1.0:
                mask_l_t /= 255.0
                mask_r_t /= 255.0
        else:
            sbs_rgb_t = sbs_rgb.unsqueeze(0)
            mask_l_t = mask_l.unsqueeze(0)
            mask_r_t = mask_r.unsqueeze(0)

        packed_t = self.pack_batch(sbs_rgb_t, mask_l_t, mask_r_t).squeeze(0)
        
        if is_numpy:
            return packed_t.cpu().numpy()
        return packed_t

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

    predictor = build_sam3_video_predictor(offload_video_to_cpu=True, async_loading_frames=True)
    hybrid_loop = HybridSam3MotionLoop(video_predictor=predictor, raft_compensator=RaftMotionCompensator(device="cuda"))
    total_frames, num_keyframes, width, height, duration, fps = metadata(video_path)

    # Use ffmpeg pipe directly to get rgb24 bytes
    read_cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", 
        "-i", video_path, "-f", "image2pipe", "-pix_fmt", "rgb24", 
        "-vcodec", "rawvideo", "-"
    ]
    reader = subprocess.Popen(read_cmd, stdout=subprocess.PIPE, bufsize=width*height*3*batch_size)
    writer = ffmpeg_pipe(out_path, width, height, fps)
    packer = AlphaCornerPacker(scale_factor=matte_size, device=hybrid_loop.device)
    hybrid_loop.raft._load_model()
    
    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing SBS Batches")

    def propagate_eye(eye_frames_tensor_rgb, bbox):
        # eye_frames_tensor_rgb is float32 [B, 3, H, W] in [0, 1] on GPU
        B, C, h, w = eye_frames_tensor_rgb.shape
        eye_masks = []
        
        # SAM3 requires PIL Images for start_session initialization
        eye_frames_cpu = (eye_frames_tensor_rgb * 255.0).clamp(0, 255).to(torch.uint8).cpu()
        eye_frames_pil = [TVF.to_pil_image(eye_frames_cpu[i]) for i in range(B)]
        
        res = hybrid_loop.predictor.handle_request(dict(type="start_session", resource_path=eye_frames_pil))
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
        hybrid_loop.predictor.model.tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)
        
        prev_logits = tracker_state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(hybrid_loop.device).float()
        batch_size_obj = len(tracker_state["obj_ids"])
        
        out_f0 = tracker_state["output_dict"]["cond_frame_outputs"][0]
        logits_f0 = out_f0["pred_masks_high_res"] if "pred_masks_high_res" in out_f0 else out_f0["pred_masks"]

        if logits_f0.shape[0] > 0:
            logits_f0 = torch.max(logits_f0, dim=0, keepdim=True).values
        else:
            logits_f0 = torch.zeros((1, 1, h, w), device=hybrid_loop.device)

        f0_resized = torch.nn.functional.interpolate(
            logits_f0.to(hybrid_loop.device), size=(h, w), mode="bicubic", align_corners=False).squeeze(0).squeeze(0)

        prob_f0 = torch.sigmoid(f0_resized)
        eye_masks.append((prob_f0 > 0.5).to(torch.uint8))
        
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for frame_idx in range(1, B):
                prev_tensor = eye_frames_tensor_rgb[frame_idx - 1]
                curr_tensor = eye_frames_tensor_rgb[frame_idx]
                
                hybrid_loop.predictor.model._prepare_backbone_feats(inference_state=inference_state,
                    frame_idx=frame_idx, reverse=False)
                
                _, _, h_mask, w_mask = prev_logits.shape
                flow = hybrid_loop.raft._compute_raft_flow(prev_tensor.unsqueeze(0), curr_tensor.unsqueeze(0)).squeeze(0)
                flow_downscaled = torch.nn.functional.interpolate(flow.unsqueeze(0), size=(h_mask, w_mask), mode="bicubic", align_corners=False).squeeze(0)
                flow_downscaled[0] *= (w_mask / w)
                flow_downscaled[1] *= (h_mask / h)
                
                warped_logits = hybrid_loop.raft._warp_frame(prev_logits.bfloat16(), flow_downscaled).float()
                
                dummy_point_inputs = {
                    "point_coords": torch.zeros(batch_size_obj, 1, 2, device=hybrid_loop.device),
                    "point_labels": -torch.ones(batch_size_obj, 1, dtype=torch.int32, device=hybrid_loop.device)
                }
                
                current_out, _ = hybrid_loop.predictor.model.tracker._run_single_frame_inference(
                    inference_state=tracker_state,
                    output_dict=tracker_state["output_dict"],
                    frame_idx=frame_idx,
                    batch_size=batch_size_obj,
                    is_init_cond_frame=False,
                    point_inputs=dummy_point_inputs,
                    mask_inputs=None,
                    reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=warped_logits,
                )
                
                tracker_state["output_dict"]["non_cond_frame_outputs"][frame_idx] = current_out
                hybrid_loop.predictor.model.tracker._add_output_per_object(tracker_state, frame_idx, current_out, "non_cond_frame_outputs")
                tracker_state["frames_already_tracked"][frame_idx] = {"reverse": False}
                
                prev_logits = current_out["pred_masks"].to(hybrid_loop.device).float()
                logits_gpu = current_out["pred_masks_high_res"] if "pred_masks_high_res" in current_out else current_out["pred_masks"]
                
                if logits_gpu.shape[0] > 0:
                    logits_gpu = torch.max(logits_gpu, dim=0, keepdim=True).values
                else:
                    logits_gpu = torch.zeros((1, 1, h, w), device=hybrid_loop.device)
                
                logits_resized = torch.nn.functional.interpolate(logits_gpu.to(hybrid_loop.device).float(), size=(h, w), mode="bilinear", align_corners=False).squeeze(0).squeeze(0)
                prob = torch.sigmoid(logits_resized)
                eye_masks.append((prob > 0.5).to(torch.uint8))
                
        hybrid_loop.predictor.handle_request(dict(type="close_session", session_id=sid))
        torch.cuda.empty_cache()
        return eye_masks

    frame_bytes = width * height * 3
    
    while frame_count < total_frames:
        raw_data = reader.stdout.read(frame_bytes * batch_size)
        if not raw_data: 
            break
            
        chunk_length = len(raw_data) // frame_bytes
        if chunk_length == 0:
            break
            
        # Parse straight to tensor on GPU
        frames_tensor = torch.frombuffer(raw_data, dtype=torch.uint8).reshape(chunk_length, height, width, 3).to(hybrid_loop.device)
        
        half_w = width // 2
        # shapes: [B, H, W, 3] -> [B, 3, H, W]
        frames_l_tensor = frames_tensor[:, :, :half_w, :].permute(0, 3, 1, 2)
        frames_r_tensor = frames_tensor[:, :, half_w:, :].permute(0, 3, 1, 2)

        target_w = int(half_w * matte_size)
        target_h = int(height * matte_size)
        
        # interpolate expects float [B, C, H, W]
        frames_l_small = torch.nn.functional.interpolate(frames_l_tensor.float() / 255.0, size=(target_h, target_w), mode='area')
        frames_r_small = torch.nn.functional.interpolate(frames_r_tensor.float() / 255.0, size=(target_h, target_w), mode='area')
        
        left_bbox_small = [
            left_bbox[0] * matte_size, left_bbox[1] * matte_size,
            left_bbox[2] * matte_size, left_bbox[3] * matte_size
        ] if left_bbox is not None else None
        
        right_bbox_small = [
            right_bbox[0] * matte_size, right_bbox[1] * matte_size,
            right_bbox[2] * matte_size, right_bbox[3] * matte_size
        ] if right_bbox is not None else None
        
        print(f"\n[SBS] Tracking Left Eye Batch (Frames {frame_count} to {frame_count+chunk_length-1}) at {int(matte_size*100)}% scale...")
        masks_l = propagate_eye(frames_l_small, left_bbox_small)

        print(f"[SBS] Tracking Right Eye Batch (Frames {frame_count} to {frame_count+chunk_length-1}) creating mattes at {int(matte_size*100)}% scale...")
        masks_r = propagate_eye(frames_r_small, right_bbox_small)

        masks_l_tensor = torch.stack(masks_l)
        masks_r_tensor = torch.stack(masks_r)

        # packer.pack_batch expects [B, H, W, 3]
        packed_batch = packer.pack_batch(frames_tensor, masks_l_tensor.float(), masks_r_tensor.float())
        packed_bytes = packed_batch.cpu().numpy().tobytes()
        writer.stdin.write(packed_bytes)
            
        del frames_tensor, frames_l_tensor, frames_r_tensor, frames_l_small, frames_r_small
        del masks_l_tensor, masks_r_tensor, packed_batch
        del masks_l, masks_r
        torch.cuda.empty_cache()
        
        frame_count += chunk_length
        pbar.update(chunk_length)
        
    reader.stdout.close()
    reader.wait()
    writer.stdin.close()
    writer.wait()
    print("Stereoscopic chunked processing complete!")

process_video_in_batches(
    video_path="test.mp4",
    out_path="test_o.mp4",
    prompt_text="One girl",
    batch_size=50,
    matte_size=0.4,
    use_class_process_batch=False
)
