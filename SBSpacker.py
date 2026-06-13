import cv2, torch, subprocess, numpy as np, json, logging, os, glob, gc
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from model_builder import build_sam3_video_predictor
import torchvision.transforms.functional as V
try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    HAS_RAFT = True
except ImportError:
    HAS_RAFT = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float16
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def target_shape(img_shape, target_size: int):
    h, w = img_shape[:2]
    new_w, new_h = (int(target_size * w / h), target_size) if h > w else (target_size, int(target_size * h / w))
    return new_h, new_w

def have(a):
    return a is not None  

def aorb(a, b):
    return a if have(a) else b

def aborc(a, b, c):
    return aorb(a, aorb(b, c))

def abcord(a, b, c, d):
    return aorb(a, aborc(b, c, d))

def no_none(x):
    return x.apply(lambda tensor: tensor if tensor is not None else None)

def denormalize(tensor, target_w, target_h):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).to(device)
    
    img_float = (tensor.float() * 0.5 + 0.5) * 255.0
    if img_float.shape[2] != target_w or img_float.shape[1] != target_h:
        img_float = torch.nn.functional.interpolate(img_float.unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False).squeeze(0)
    img = img_float.permute(1, 2, 0).to(torch.uint8)
    return img

def soft_matte(outputs, frames=None, width=None, height=None, fg_color=(255, 0, 0), bg_color=(0, 0, 0), dilation=0, feather_radius=0, smooth_edges=0):
    
    if frames is not None:
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames).to(device)

        if frames.ndim == 3 and frames.shape[0] in [1, 3]:
            frames = frames.permute(1, 2, 0)
            if frames.min() < -0.1:
                frames = (frames * 0.5) + 0.5

        if frames.dtype in [torch.bfloat16, torch.float16, torch.float32] or frames.max() <= 1.0:
            frames = (frames * 255)

        frames = frames[..., :3].to(device, dtype=dtype)
        height, width = frames.shape[:2]

    combined_mask = torch.zeros((height, width), dtype=dtype).to(device)
    if "out_binary_masks" in outputs:
        for i in range(len(outputs["out_obj_ids"])):
            mask = outputs["out_binary_masks"][i]

            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).to(device)

            if dilation != 0:
                if dilation > 0:
                    mask = morph3x3(mask, dilation)
                else:
                    mask = morph3x3(mask, dilation)

            if mask.shape != (height, width):
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(height, width), mode='bicubic', align_corners=True, antialias=True).squeeze(0).squeeze(0)

            if smooth_edges > 0:
                mask = mask_edges(mask, kernel_size=smooth_edges)
        
            combined_mask = torch.maximum(combined_mask, mask)

        if feather_radius > 0:
            combined_mask = feather_mask(combined_mask, blur_radius=feather_radius)

    mask_3d = combined_mask[:, :, None]
    fg_array = torch.tensor(fg_color, dtype=dtype).to(device)
    bg_array = torch.tensor(bg_color, dtype=dtype).to(device)
    
    blended = (fg_array * mask_3d + bg_array * (1.0 - mask_3d)).to(torch.uint8)
    return blended

def feather_mask(mask: torch.Tensor, blur_radius: float = 1.5, iterations: int = 3) -> torch.Tensor:

    if blur_radius <= 0:
        return mask

    orig_shape = mask.shape
    if mask.ndim == 2:
        x = mask[None, None, ...]
    elif mask.ndim == 3:
        x = mask[None, ...]
    else:
        x = mask
        
    x = x.float()
    k_size = int(blur_radius * 2) + 1
    if k_size % 2 == 0:
        k_size += 1

    for _ in range(iterations):
        x = V.gaussian_blur(x, kernel_size=k_size, sigma=float(blur_radius))

    return x.view(orig_shape) 

def morph3x3(mask: torch.Tensor, dilation: int) -> torch.Tensor:
    
    if dilation == 0: return mask
    x = mask.float().view(1, 1, *mask.shape) if mask.ndim == 2 else mask
    k_size = 2 * abs(dilation) + 1
    padding = abs(dilation)
    
    if dilation > 0:
        x = torch.nn.functional.max_pool2d(x, kernel_size=k_size, stride=1, padding=padding)
    else:
        x = -torch.nn.functional.max_pool2d(-x, kernel_size=k_size, stride=1, padding=padding)
    return (x > 0.5).to(mask.dtype).view(mask.shape)

def mask_edges(mask: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
    if kernel_size <= 0:
        return mask
    
    orig_shape = mask.shape
    x = mask.float()
    if x.ndim == 2:
        x = x[None, None, ...]
    elif x.ndim == 3:
        x = x[None, ...]
        
    x = (x > 0.5).float()
    pad = kernel_size // 2
    k_size = pad * 2 + 1
    
    if pad > 0:
        x = torch.nn.functional.max_pool2d(x, kernel_size=k_size, stride=1, padding=pad)
        x = -torch.nn.functional.max_pool2d(-x, kernel_size=k_size, stride=1, padding=pad)
        
        x = -torch.nn.functional.max_pool2d(-x, kernel_size=k_size, stride=1, padding=pad)
        x = torch.nn.functional.max_pool2d(x, kernel_size=k_size, stride=1, padding=pad)
    
    blur_k = k_size + 2
    if blur_k % 2 == 0:
        blur_k += 1
    
    sigma = float(blur_k) / 3.0
    x = V.gaussian_blur(x, kernel_size=blur_k, sigma=sigma)
    
    x = (x > 0.5).to(mask.dtype)
    
    return x.view(orig_shape)

def apply_effects(masks, dilation, feather_radius, smooth_edges):
    out_masks = []
    for m in masks:
        t = torch.from_numpy(m).to(device).float() / 255.0
        if dilation != 0:
            t = morph3x3(t, dilation)
        if smooth_edges > 0:
            t = mask_edges(t, kernel_size=smooth_edges)
        if feather_radius > 0:
            t = feather_mask(t, blur_radius=feather_radius)
        out_masks.append((t.cpu().numpy() * 255).astype(np.uint8))
    return out_masks

def metadata(path):
    cmd_key = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'frame=pict_type',
        '-of', 'csv=p=0', '-skip_frame', 'nokey', path]

    res_key = subprocess.run(cmd_key, capture_output=True, text=True)
    lines = res_key.stdout.strip().split('\n')
    num_keyframes = len(lines)

    cmd_stream = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json', 
        '-show_streams', '-select_streams', 'v:0', path]

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
        '-i', '-', '-c:v', 'hevc_qsv', '-profile:v', 'main10', '-pix_fmt', 'p010le', '-tag:v', 'hvc1', '-g', '30', '-b:v', '100M', '-preset', 'veryslow',
        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-fps_mode', 'cfr', '-r', str(fps), '-movflags', '+faststart+write_colr+use_metadata_tags',
        '-metadata:s:v:0', 'stereo_mode=left_right', '-color_trc', 'bt709', '-color_range', 'pc', out_path
    ]
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

# class RaftMotionCompensator:
#     def __init__(self, device=None, max_size=256, scale_factor=0.5, interp_mode="bilinear"):
#         self.device = torch.device(device) if isinstance(device, str) else (device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#         self.max_size = max_size
#         self.scale_factor = scale_factor
#         self.interp_mode = interp_mode
#         self.model = None
#         self.transforms = None

#     def load_model(self):
#         if self.model is None:
#             if not HAS_RAFT:
#                 raise ImportError("torchvision.models.optical_flow is required for ")
#             weights = Raft_Small_Weights.DEFAULT
#             self.transforms = weights.transforms()
#             self.model = raft_small(weights=weights, progress=False).to(self.device).eval()

#     def compute_raft_flow(self, img1, img2, max_size=256, scale_factor=0.5, interp_mode="bilinear"):
#         orig_H, orig_W = img1.shape[2], img1.shape[3]
#         scale_factor = self.scale_factor
#         current_H, current_W = orig_H * scale_factor, orig_W * scale_factor

#         if max(current_H, current_W) > self.max_size:
#             scale_factor = scale_factor * (self.max_size / float(max(current_H, current_W)))

#         if scale_factor != 1.0:
#             new_H, new_W = int(orig_H * scale_factor), int(orig_W * scale_factor)
#             img1_s = F.interpolate(img1, size=(new_H, new_W), mode=self.interp_mode, antialias=True)
#             img2_s = F.interpolate(img2, size=(new_H, new_W), mode=self.interp_mode, antialias=True)
#         else:
#             img1_s, img2_s = img1, img2
            
#         img1_t, img2_t = self.transforms(img1_s, img2_s)
#         _, _, H_s, W_s = img1_t.shape
#         pad_h, pad_w = (8 - H_s % 8) % 8, (8 - W_s % 8) % 8

#         if pad_h > 0 or pad_w > 0:
#             img1_t = F.pad(img1_t, (0, pad_w, 0, pad_h))
#             img2_t = F.pad(img2_t, (0, pad_w, 0, pad_h))
            
#         with torch.autocast(device_type=self.device.type, dtype=torch.float16 if self.device.type == 'cuda' else torch.float32):
#             flow = self.model(img1_t, img2_t)[-1].float()

#         flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
#         if pad_h > 0 or pad_w > 0:
#             flow = flow[:, :, :H_s, :W_s]

#         if scale_factor != 1.0:
#             flow = F.interpolate(flow, size=(orig_H, orig_W), mode=self.interp_mode)
#             flow = flow / scale_factor

#         return flow

#     def warp_frame(self, pt_frame, flow, t=1.0, max_size=256, scale_factor=0.5, interp_mode="bilinear"):

#         if pt_frame.ndim == 3:
#             C, H, W = pt_frame.shape
#             scale_factord = flow * t
#             y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
#             x_norm = 2.0 * (x + scale_factord[0]) / max(W - 1, 1) - 1.0
#             y_norm = 2.0 * (y + scale_factord[1]) / max(H - 1, 1) - 1.0
#             grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
#             align_corners = True if self.interp_mode != 'nearest' else None

#             if align_corners is None:
#                 return F.grid_sample(pt_frame.unsqueeze(0), grid, mode=self.interp_mode, padding_mode='border').squeeze(0)
#             else:
#                 return F.grid_sample(pt_frame.unsqueeze(0), grid, mode=self.interp_mode, padding_mode='border', align_corners=align_corners).squeeze(0)

#         elif pt_frame.ndim == 4:
#             N, C, H, W = pt_frame.shape
#             scale_factord = flow * t
#             y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
#             x_norm = 2.0 * (x + scale_factord[0]) / max(W - 1, 1) - 1.0
#             y_norm = 2.0 * (y + scale_factord[1]) / max(H - 1, 1) - 1.0
#             grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
#             grid = grid.expand(N, -1, -1, -1)
#             align_corners = True if self.interp_mode != 'nearest' else None

#             if align_corners is None:
#                 return F.grid_sample(pt_frame, grid, mode=self.interp_mode, padding_mode='border')
#             else:
#                 return F.grid_sample(pt_frame, grid, mode=self.interp_mode, padding_mode='border', align_corners=align_corners)
#         else:
#             raise ValueError(f"Unexpected pt_frame dimensions: {pt_frame.ndim}")

#     def stabilize_alpha(self, rgb_frames, alpha_masks, blend_weights=(0.2, 0.6, 0.2)):

#         self.load_model()
#         is_numpy = isinstance(rgb_frames, np.ndarray)

#         if is_numpy:
#             t_rgb = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
#             if alpha_masks.ndim == 3:
#                 t_alpha = torch.from_numpy(alpha_masks).unsqueeze(1).float().div(255.0).to(self.device)
#             else:
#                 t_alpha = torch.from_numpy(alpha_masks).permute(0, 3, 1, 2).float().div(255.0).to(self.device)
#         else:
#             t_rgb = rgb_frames.to(self.device).float()
#             t_alpha = alpha_masks.to(self.device).float()

#         num_frames = t_rgb.shape[0]
#         if num_frames < 3: return alpha_masks
#         stabilized_alphas = torch.zeros_like(t_alpha)
#         stabilized_alphas[0] = t_alpha[0]
#         stabilized_alphas[-1] = t_alpha[-1]
#         w_prev, w_curr, w_next = blend_weights

#         with torch.no_grad():
#             for i in tqdm(range(1, num_frames - 1)):
#                 prev_rgb, curr_rgb, next_rgb = t_rgb[i-1:i+2]
#                 prev_alpha, curr_alpha, next_alpha = t_alpha[i-1:i+2]
#                 flow_forward = self.compute_raft_flow(prev_rgb.unsqueeze(0), curr_rgb.unsqueeze(0)).squeeze(0)
#                 flow_backward = self.compute_raft_flow(next_rgb.unsqueeze(0), curr_rgb.unsqueeze(0)).squeeze(0)
#                 alpha_prev_warped = self.warp_frame(prev_alpha, flow_forward, t=1.0)
#                 alpha_next_warped = self.warp_frame(next_alpha, flow_backward, t=1.0)
#                 merged_alpha = (w_prev * alpha_prev_warped) + (w_curr * curr_alpha) + (w_next * alpha_next_warped)
#                 stabilized_alphas[i] = torch.clamp(merged_alpha, 0.0, 1.0)

#         if is_numpy:
#             if alpha_masks.ndim == 3:
#                 return (stabilized_alphas.squeeze(1).cpu().numpy() * 255.0).astype(np.uint8)
#             return (stabilized_alphas.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
#         return stabilized_alphas

#     def interpolate_frame_cached(self, frame_a, frame_b, t, flow_fwd, flow_bwd, max_size=256, scale_factor=0.5, interp_mode="bilinear"):
#         warp_a =  self.warp_frame(frame_a, flow_fwd, t, interp_mode=interp_mode)
#         warp_b =  self.warp_frame(frame_b, flow_bwd, 1.0 - t, interp_mode=interp_mode)
        
#         mag_fwd = torch.norm(flow_fwd, dim=0, keepdim=True)
#         mag_bwd = torch.norm(flow_bwd, dim=0, keepdim=True)
        
#         weight_a = torch.exp(-mag_fwd)
#         weight_b = torch.exp(-mag_bwd)
        
#         weights = weight_a + weight_b + 1e-6
#         weight_a = weight_a / weights
#         weight_b = weight_b / weights
        
#         blended = warp_a * weight_a + warp_b * weight_b
#         return torch.clamp(blended, 0.0, 1.0)

#     def motion_compensated(self, pt_frames, indices, threads=0, threshold=0.05, chunk_size=16, max_size=256, scale_factor=0.5, interp_mode="bilinear"):
#         if not HAS_RAFT:
#             raise ImportError("torchvision.models.optical_flow is required for  Please upgrade torchvision.")
            
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         num_frames = len(pt_frames)
        
#         if num_frames < 2:
#             return pt_frames

#         cuts =  self.detect_scene_cuts(pt_frames, threshold=threshold, chunk_size=chunk_size, interp_mode=interp_mode)

#         print("[Motion‑compensated] Loading RAFT model...")
#         weights = Raft_Small_Weights.DEFAULT
#         transforms = weights.transforms()
#         model = raft_small(weights=weights, progress=False).to(device).eval()

#         flow_fwd_cache = {}
#         flow_bwd_cache = {}

#         print("[Motion‑compensated] Precomputing RAFT optical flow (forward and backward)...")
#         with torch.no_grad():
#             for i in tqdm(range(num_frames - 1), desc="RAFT Flow Cache", colour="yellow", unit="pair"):
#                 if cuts[i]:
#                     continue

#                 img1 = pt_frames[i:i+1].to(device)
#                 img2 = pt_frames[i+1:i+2].to(device)

#                 flow_fwd =  self.compute_raft_flow(model, transforms, img1, img2, device, max_size=max_size, interp_mode=interp_mode, scale_factor=scale_factor)
#                 flow_bwd =  self.compute_raft_flow(model, transforms, img2, img1, device, max_size=max_size, interp_mode=interp_mode, scale_factor=scale_factor)

#                 flow_fwd_cache[i] = flow_fwd.squeeze(0).cpu()
#                 flow_bwd_cache[i] = flow_bwd.squeeze(0).cpu()

#                 del img1, img2, flow_fwd, flow_bwd
#                 if i % 5 == 0:
#                     torch.cuda.empty_cache()

#         del model
#         torch.cuda.empty_cache()

#         selected = []
        
#         for idx in tqdm(indices, desc="Motion‑compensated (RAFT, cached)", colour="blue", unit="frame"):
#             if idx <= 0.0:
#                 selected.append(pt_frames[0])
#                 continue
#             if idx >= num_frames - 1:
#                 selected.append(pt_frames[-1])
#                 continue

#             base = int(torch.floor(torch.tensor(idx)).item())
#             t = float(idx - base)
#             is_cut = cuts[base] if 0 <= base < len(cuts) else False

#             frame_a = pt_frames[base]
#             frame_b = pt_frames[base + 1]

#             if is_cut:
#                 selected.append(frame_a if t < 0.5 else frame_b)
#                 continue

#             if t <= 1e-6:
#                 selected.append(frame_a)
#                 continue
#             if t >= 1.0 - 1e-6:
#                 selected.append(frame_b)
#                 continue

#             flow_fwd = flow_fwd_cache.get(base)
#             flow_bwd = flow_bwd_cache.get(base)

#             if flow_fwd is None or flow_bwd is None:
#                 selected.append(frame_a if t < 0.5 else frame_b)
#                 continue

#             out = self.interpolate_frame_cached(frame_a.to(device), frame_b.to(device), t, flow_fwd.to(device), flow_bwd.to(device), interp_mode=interp_mode)
#             selected.append(out.cpu())

#         return torch.stack(selected)

#     def process_batch(self, frames_pil, frames_bgr, prompt_text=None, bbox=None, prior_mask=None):

#         self.load_model()
#         height, width = frames_bgr[0].shape[:2]
#         chunk = len(frames_pil)

#         res_inline = self.predictor.handle_request(dict(
#             type="start_session",
#             resource_path=frames_pil))

#         sid_inline = res_inline["session_id"]
#         if prior_mask is not None:
#             self.predictor.handle_request(dict(
#                 type="add_new_mask",
#                 session_id=sid_inline,
#                 frame_index=0,
#                 obj_id=0,
#                 mask=prior_mask))

#         prompt_req_inline = dict(type="add_prompt", session_id=sid_inline, frame_index=0, obj_id=0)

#         if prompt_text is not None:
#             prompt_req_inline["text"] = prompt_text

#         if bbox is not None:
#             prompt_req_inline["bounding_boxes"] = [bbox]
#             prompt_req_inline["bounding_box_labels"] = [1]

#         if prompt_text is not None or bbox is not None or prior_mask is None:
#             self.predictor.handle_request(prompt_req_inline)

#         session_inline = self.predictor._get_session(sid_inline)
#         inference_state = session_inline["state"]
#         tracker_states = inference_state["tracker_inference_states"]

#         if len(tracker_states) == 0:
#             print(f"[WARNING] Prompt '{prompt_text}' found no objects in this chunk. Generating empty masks.")
#             self.predictor.handle_request(dict(type="close_session", session_id=sid_inline))
#             return [np.zeros((height, width), dtype=np.uint8) for _ in range(chunk)]

#         tracker_state = tracker_states[0]
#         tensors_rgb = []

#         for f_bgr in frames_bgr:
#             f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
#             t_rgb = torch.from_numpy(f_rgb).permute(2, 0, 1).float().div(255.0).to(self.device)
#             tensors_rgb.append(t_rgb)

#         prev_logits = tracker_state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(self.device).float()
#         batch_size = len(tracker_state["obj_ids"])
#         self.predictor.model.tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)

#         with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
#             for frame_idx in range(1, chunk):
#                 prev_tensor = tensors_rgb[frame_idx - 1]
#                 curr_tensor = tensors_rgb[frame_idx]
#                 self.predictor.model._prepare_backbone_feats(inference_state=inference_state, frame_idx=frame_idx, reverse=False)

#                 _, _, h_mask, w_mask = prev_logits.shape

#                 flow = self.compute_raft_flow(curr_tensor.unsqueeze(0), prev_tensor.unsqueeze(0)).squeeze(0)
#                 # flow = self.compute_raft_flow(prev_tensor.unsqueeze(0), curr_tensor.unsqueeze(0)).squeeze(0)
#                 flow_downscaled = torch.nn.functional.interpolate(flow.unsqueeze(0), size=(h_mask, w_mask), mode="bilinear", align_corners=False).squeeze(0)
#                 flow_downscaled[0] *= (w_mask / width)
#                 flow_downscaled[1] *= (h_mask / height)

#                 warped_logits = self.warp_frame(prev_logits, flow_downscaled)

#                 dummy_point_inputs = {
#                     "point_coords": torch.zeros(batch_size, 1, 2, device=self.device),
#                     "point_labels": -torch.ones(batch_size, 1, dtype=torch.int32, device=self.device)
#                 }

#                 current_out, _ = self.predictor.model.tracker._run_single_frame_inference(
#                     inference_state=tracker_state,
#                     output_dict=tracker_state["output_dict"],
#                     frame_idx=frame_idx,
#                     batch_size=batch_size,
#                     is_init_cond_frame=False,
#                     point_inputs=dummy_point_inputs,
#                     mask_inputs=None,
#                     reverse=False,
#                     run_mem_encoder=True,
#                     prev_sam_mask_logits=warped_logits,
#                 )

#                 tracker_state["output_dict"]["non_cond_frame_outputs"][frame_idx] = current_out
#                 self.predictor.model.tracker._add_output_per_object(tracker_state, frame_idx, current_out, "non_cond_frame_outputs")

#                 tracker_state["frames_already_tracked"][frame_idx] = {"reverse": False}
#                 prev_logits = current_out["pred_masks"].to(self.device).float()

#         final_masks = []

#         for i in range(chunk):
#             storage_key = "cond_frame_outputs" if i == 0 else "non_cond_frame_outputs"
#             out = tracker_state["output_dict"][storage_key][i]
#             logits_gpu = out["pred_masks_high_res"].to(self.device) if "pred_masks_high_res" in out else out["pred_masks"].to(self.device)

#             if logits_gpu.shape[0] > 0:
#                 logits_gpu = torch.max(logits_gpu, dim=0, keepdim=True).values
#             else:
#                 logits_gpu = torch.zeros((1, 1, height, width), device=self.device)

#             logits_resized = torch.nn.functional.interpolate(
#                 logits_gpu,
#                 size=(height, width),
#                 mode="bilinear",
#                 align_corners=False).squeeze(0).squeeze(0)

#             prob = torch.sigmoid(logits_resized)
#             final_masks.append(((prob > 0.5) * 255).to(torch.uint8).cpu().numpy())

#         self.predictor.handle_request(dict(type="close_session", session_id=sid_inline))
#         return final_masks

# class HybridSam3MotionLoop:
#     def __init__(self, video_predictor=None, raft_compensator=None, target_res=(256, 256)):

#         self.predictor = video_predictor
#         self.raft = raft_compensator
#         self.device = self.device
#         self.target_res = target_res

# def load_model():
#     if model is None:
#         if not HAS_RAFT:
#             raise ImportError("torchvision.models.optical_flow is required for ")
#         weights = Raft_Small_Weights.DEFAULT
#         transforms = weights.transforms()
#         model = raft_small(weights=weights, progress=False).to(device).eval()
    # return model, transforms, weights

def compute_raft_flow(img1, img2, max_size=256, scale_factor=1.0, interp_mode="bilinear", target_size=None):

    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights, progress=False).to(device).eval()
    orig_H, orig_W =  V.get_image_size(img1)

    current_H, current_W = orig_H * scale_factor, orig_W * scale_factor 

    if max(current_H, current_W) > max_size:
        scale_factor = scale_factor * (max_size / float(max(current_H, current_W)))

    if scale_factor != 1.0:
        new_H, new_W = int(orig_H * scale_factor), int(orig_W * scale_factor)
        img1_s = F.interpolate(img1, size=(new_H, new_W), mode=interp_mode, antialias=True)
        img2_s = F.interpolate(img2, size=(new_H, new_W), mode=interp_mode, antialias=True)
    else:
        new_H, new_W = orig_H, orig_W
        img1_s, img2_s = img1, img2

    img1_t, img2_t = transforms(img1_s, img2_s)
    _, _, H_s, W_s = img1_t.shape
    pad_h, pad_w = (8 - H_s % 8) % 8, (8 - W_s % 8) % 8

    if pad_h > 0 or pad_w > 0:
        img1_t = F.pad(img1_t, (0, pad_w, 0, pad_h))
        img2_t = F.pad(img2_t, (0, pad_w, 0, pad_h))

    # h, w =  V.get_image_size(img1_t)
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16 if device.type == 'cuda' else torch.float32):

        flow = model(img1_t, img2_t)[-1].float()
        flow = torch.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :H_s, :W_s]

        out_H, out_W = target_size if target_size else (orig_H, orig_W)
        if out_H != H_s or out_W != W_s:
            flow = F.interpolate(flow, size=(out_H, out_W), mode=interp_mode)
            flow[:, 0] *= (out_W / W_s)
            flow[:, 1] *= (out_H / H_s)
            
    return flow

def warp_frame(pt_frame, flow, t=1.0, max_size=256, scale_factor=0.5, interp_mode="bilinear"):

    if pt_frame.ndim == 3:
        C, H, W = pt_frame.shape
        scale_factord = flow * t
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        x_norm = 2.0 * (x + scale_factord[0]) / max(W - 1, 1) - 1.0
        y_norm = 2.0 * (y + scale_factord[1]) / max(H - 1, 1) - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
        align_corners = True if interp_mode != 'nearest' else None

        if align_corners is None:
            return F.grid_sample(pt_frame.unsqueeze(0), grid, mode=interp_mode, padding_mode='border').squeeze(0)
        else:
            return F.grid_sample(pt_frame.unsqueeze(0), grid, mode=interp_mode, padding_mode='border', align_corners=align_corners).squeeze(0)

    elif pt_frame.ndim == 4:
        N, C, H, W = pt_frame.shape
        scale_factord = flow * t
        y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        x_norm = 2.0 * (x + scale_factord[0]) / max(W - 1, 1) - 1.0
        y_norm = 2.0 * (y + scale_factord[1]) / max(H - 1, 1) - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
        grid = grid.expand(N, -1, -1, -1)
        align_corners = True if interp_mode != 'nearest' else None

        if align_corners is None:
            return F.grid_sample(pt_frame, grid, mode=interp_mode, padding_mode='border')
        else:
            return F.grid_sample(pt_frame, grid, mode=interp_mode, padding_mode='border', align_corners=align_corners)
    else:
        raise ValueError(f"Unexpected pt_frame dimensions: {pt_frame.ndim}")

def stabilize_alpha(rgb_frames, alpha_masks, blend_weights=(0.3, 0.4, 0.3)):
    is_numpy = isinstance(rgb_frames, np.ndarray)

    if is_numpy:
        t_rgb = torch.from_numpy(rgb_frames).permute(0, 3, 1, 2).float().div(255.0).to(device)
        if alpha_masks.ndim == 3:
            t_alpha = torch.from_numpy(alpha_masks).unsqueeze(1).float().div(255.0).to(device)
        else:
            t_alpha = torch.from_numpy(alpha_masks).permute(0, 3, 1, 2).float().div(255.0).to(device)
    else:
        t_rgb = rgb_frames.to(device).float()
        t_alpha = alpha_masks.to(device).float()

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
            flow_forward = compute_raft_flow(prev_rgb.unsqueeze(0), curr_rgb.unsqueeze(0)).squeeze(0)
            flow_backward = compute_raft_flow(next_rgb.unsqueeze(0), curr_rgb.unsqueeze(0)).squeeze(0)
            alpha_prev_warped = warp_frame(prev_alpha, flow_forward, t=1.0)
            alpha_next_warped = warp_frame(next_alpha, flow_backward, t=1.0)
            merged_alpha = (w_prev * alpha_prev_warped) + (w_curr * curr_alpha) + (w_next * alpha_next_warped)
            stabilized_alphas[i] = torch.clamp(merged_alpha, 0.0, 1.0)

    if is_numpy:
        if alpha_masks.ndim == 3:
            return (stabilized_alphas.squeeze(1).cpu().numpy() * 255.0).astype(np.uint8)
        return (stabilized_alphas.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
    return stabilized_alphas

def interpolate_frame_cached(frame_a, frame_b, t, flow_fwd, flow_bwd, max_size=256, scale_factor=0.5, interp_mode="bilinear"):
    warp_a =  warp_frame(frame_a, flow_fwd, t, interp_mode=interp_mode)
    warp_b =  warp_frame(frame_b, flow_bwd, 1.0 - t, interp_mode=interp_mode)
    
    mag_fwd = torch.norm(flow_fwd, dim=0, keepdim=True)
    mag_bwd = torch.norm(flow_bwd, dim=0, keepdim=True)
    
    weight_a = torch.exp(-mag_fwd)
    weight_b = torch.exp(-mag_bwd)
    
    weights = weight_a + weight_b + 1e-6
    weight_a = weight_a / weights
    weight_b = weight_b / weights
    
    blended = warp_a * weight_a + warp_b * weight_b
    return torch.clamp(blended, 0.0, 1.0)

def detect_scene_cuts(pt_frames, threshold=0.05, chunk_size=32, interp_mode="bilinear"):
    n = len(pt_frames)

    if n < 2:
        return torch.zeros(0, dtype=torch.bool, device=torch.device('cpu'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    small_frames = []
    
    for i in range(0, n, chunk_size):
        chunk = pt_frames[i:i+chunk_size].to(device)
        align_corners = False if interp_mode != 'nearest' else None
        recompute_scale_factor = False
        antialias = True if interp_mode != 'nearest' else False
        small_chunk = F.interpolate(chunk, size=(64, 64), mode=interp_mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor, antialias=antialias)
        small_frames.append(small_chunk.cpu())
        
    small_frames = torch.cat(small_frames, dim=0)
    diff = small_frames[1:] - small_frames[:-1]
    mse = (diff ** 2).mean(dim=[1, 2, 3])
    
    cuts = mse > threshold
    return cuts

def motion_compensated(pt_frames, indices, threads=0, threshold=0.05, chunk_size=16, max_size=256, scale_factor=0.5, interp_mode="bilinear"):
    if not HAS_RAFT:
        raise ImportError("torchvision.models.optical_flow is required for  Please upgrade torchvision.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_frames = len(pt_frames)
    
    if num_frames < 2:
        return pt_frames

    cuts =  detect_scene_cuts(pt_frames, threshold=threshold, chunk_size=chunk_size, interp_mode=interp_mode)

    print("[Motion‑compensated] Loading RAFT model...")
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights, progress=False).to(device).eval()

    flow_fwd_cache = {}
    flow_bwd_cache = {}

    print("[Motion‑compensated] Precomputing RAFT optical flow (forward and backward)...")
    with torch.no_grad():
        for i in tqdm(range(num_frames - 1), desc="RAFT Flow Cache", colour="yellow", unit="pair"):
            if cuts[i]:
                continue

            img1 = pt_frames[i:i+1].to(device)
            img2 = pt_frames[i+1:i+2].to(device)

            flow_fwd =  compute_raft_flow(model, transforms, img1, img2, device, max_size=max_size, interp_mode=interp_mode, scale_factor=scale_factor)
            flow_bwd =  compute_raft_flow(model, transforms, img2, img1, device, max_size=max_size, interp_mode=interp_mode, scale_factor=scale_factor)

            flow_fwd_cache[i] = flow_fwd.squeeze(0).cpu()
            flow_bwd_cache[i] = flow_bwd.squeeze(0).cpu()

            del img1, img2, flow_fwd, flow_bwd
            if i % 5 == 0:
                torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    selected = []
    
    for idx in tqdm(indices, desc="Motion‑compensated (RAFT, cached)", colour="blue", unit="frame"):
        if idx <= 0.0:
            selected.append(pt_frames[0])
            continue
        if idx >= num_frames - 1:
            selected.append(pt_frames[-1])
            continue

        base = int(torch.floor(torch.tensor(idx)).item())
        t = float(idx - base)
        is_cut = cuts[base] if 0 <= base < len(cuts) else False

        frame_a = pt_frames[base]
        frame_b = pt_frames[base + 1]

        if is_cut:
            selected.append(frame_a if t < 0.5 else frame_b)
            continue

        if t <= 1e-6:
            selected.append(frame_a)
            continue
        if t >= 1.0 - 1e-6:
            selected.append(frame_b)
            continue

        flow_fwd = flow_fwd_cache.get(base)
        flow_bwd = flow_bwd_cache.get(base)

        if flow_fwd is None or flow_bwd is None:
            selected.append(frame_a if t < 0.5 else frame_b)
            continue

        out = interpolate_frame_cached(frame_a.to(device), frame_b.to(device), t, flow_fwd.to(device), flow_bwd.to(device), interp_mode=interp_mode)
        selected.append(out.cpu())

    return torch.stack(selected)

class AlphaCornerPacker:
    def __init__(self, scale_factor=0.40, padding=0):

        self.scale = scale_factor
        self.padding = padding
        self.vignette_cache = None

    def get_circular_vignette(self, w, h):
   
        if self.vignette_cache is not None and self.vignette_cache.shape == (h, w):
            return self.vignette_cache

        vignette = np.zeros((h, w), dtype=np.uint16)
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 2 
        cv2.circle(vignette, center, radius, 1.0, -1, cv2.LINE_AA)
        self.vignette_cache = cv2.GaussianBlur(vignette, (15, 15), 0)
        return self.vignette_cache

    def pack_frame(self, sbs_rgb, mask_l, mask_r):

        dilation = 0
        feather = 1
        smooth = 1
        fcolor = (0, 0, 255)  
        bcolor = (0, 0, 0)

        H, SBS_W, C = sbs_rgb.shape
        W = SBS_W // 2

        if mask_l.dtype != np.uint8:
            mask_l = (mask_l * 255).astype(np.uint8)
            mask_r = (mask_r * 255).astype(np.uint8)

        target_w = int(W * self.scale)
        target_h = int(H * self.scale)
        
        if mask_l.shape[:2] != (target_h, target_w):
            l_small = cv2.resize(mask_l, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            l_small = mask_l

        if mask_r.shape[:2] != (target_h, target_w):
            r_small = cv2.resize(mask_r, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            r_small = mask_r

        vignette = self.get_circular_vignette(target_w, target_h)
        mask_l_vignette = (l_small * vignette).astype(np.uint8)
        mask_r_vignette = (r_small * vignette).astype(np.uint8)
 
        mask_l_vignette = l_small.astype(np.uint8) 
        mask_r_vignette = r_small.astype(np.uint8) 

        packed_frame = sbs_rgb

        h_half = target_h // 2
        top_half_mask = mask_l_vignette[:h_half, :]
        bottom_half_mask = mask_l_vignette[h_half:h_half*2, :]

        w_half = target_w // 2
        q_tl_mask = mask_r_vignette[:h_half, :w_half]
        q_tr_mask = mask_r_vignette[:h_half, w_half:w_half*2]
        q_bl_mask = mask_r_vignette[h_half:h_half*2, :w_half]
        q_br_mask = mask_r_vignette[h_half:h_half*2, w_half:w_half*2]

        def blend_red_mask(roi, mask_1ch):
            inv_mask_3d = (255 - mask_1ch)[..., np.newaxis]
            blended = (roi.astype(np.uint16) * inv_mask_3d) // 255
            blended[..., 2] += mask_1ch  # BGR: channel 2 is Red
            return blended.astype(np.uint8)

        y1_top = self.padding
        y2_top = y1_top + h_half
        x1_mid = (SBS_W // 2) - (target_w // 2)
        x2_mid = x1_mid + target_w
        
        packed_frame[y1_top:y2_top, x1_mid:x2_mid] = blend_red_mask(packed_frame[y1_top:y2_top, x1_mid:x2_mid], bottom_half_mask)

        y1_bot = H - self.padding - h_half
        y2_bot = y1_bot + h_half
        packed_frame[y1_bot:y2_bot, x1_mid:x2_mid] = blend_red_mask(packed_frame[y1_bot:y2_bot, x1_mid:x2_mid], top_half_mask)

        y1_tr = self.padding
        y2_tr = y1_tr + h_half
        x1_tr = SBS_W - self.padding - w_half
        x2_tr = SBS_W - self.padding
        packed_frame[y1_tr:y2_tr, x1_tr:x2_tr] = blend_red_mask(packed_frame[y1_tr:y2_tr, x1_tr:x2_tr], q_bl_mask)

        y1_tl_l = self.padding
        y2_tl_l = y1_tl_l + h_half
        x1_tl_l = self.padding
        x2_tl_l = self.padding + w_half
        packed_frame[y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l] = blend_red_mask(packed_frame[y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l], q_br_mask)

        y1_br_r = H - self.padding - h_half
        y2_br_r = y1_br_r + h_half
        x1_br_r = SBS_W - self.padding - w_half
        x2_br_r = SBS_W - self.padding
        packed_frame[y1_br_r:y2_br_r, x1_br_r:x2_br_r] = blend_red_mask(packed_frame[y1_br_r:y2_br_r, x1_br_r:x2_br_r], q_tl_mask)

        y1_bl_l = H - self.padding - h_half
        y2_bl_l = y1_bl_l + h_half
        x1_bl_l = self.padding
        x2_bl_l = self.padding + w_half
        packed_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l] = blend_red_mask(
            packed_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l], q_tr_mask
        )

        return packed_frame

def propagate_eye(predictor, frames_pil, frames_bgr, prompt_text, bbox, prior_mask,
                magnitude=False, meld=False, allthethings=False):

    chunk_length = len(frames_bgr)
    height, width = frames_bgr[0].shape[:2]

    res = predictor.handle_request(dict(
        type="start_session",
        resource_path=frames_pil))
   
    sid = res["session_id"]
    
    if prior_mask is not None:
        predictor.handle_request(dict(
            type="add_new_mask",
            session_id=sid,
            frame_index=0,
            obj_id=0,
            mask=prior_mask))
        
    prompt_req = dict(type="add_prompt", session_id=sid, frame_index=0, obj_id=0)

    if prompt_text is not None:
        prompt_req["text"] = prompt_text

    if bbox is not None:
        prompt_req["bounding_boxes"] = [bbox]
        prompt_req["bounding_box_labels"] = [1]
        
    if prompt_text is not None or bbox is not None or prior_mask is None:
        predictor.handle_request(prompt_req)
    
    # --- Phase 1: Native SAM3 Tracking ---
    out_buffer = []
    for st in predictor.handle_stream_request(dict(
        type="propagate_in_video",
        session_id=sid,
        propagation_direction="forward",
        start_frame_index=0,
        max_frame_num_to_track=chunk_length,
    )):
        out_buffer.append(st["outputs"])

    predictor.handle_request(dict(type="close_session", session_id=sid))

    sam_masks = []
    for i in range(chunk_length):
        if i < len(out_buffer):
            outputs = out_buffer[i]
            mask = np.zeros((height, width), dtype=np.float32)
            if "out_binary_masks" in outputs:
                for m in outputs["out_binary_masks"]:
                    if isinstance(m, torch.Tensor):
                        m = m.cpu().numpy()
                    if m.shape != (height, width):
                        m = cv2.resize(m.astype(np.float32), (width, height),
                                       interpolation=cv2.INTER_NEAREST)
                    mask = np.maximum(mask, m.astype(np.float32))
            sam_masks.append(mask)
        else:
            sam_masks.append(np.zeros((height, width), dtype=np.float32))

    if chunk_length >= 3:
        rgb_np = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr])
        alpha_np = np.stack(sam_masks) * 255.0
        stabilized = stabilize_alpha(
            rgb_np, alpha_np, blend_weights=(0.15, 0.70, 0.15)
        )
        
        final_masks = []
        for i in range(chunk_length):
            m = stabilized[i] if isinstance(stabilized, np.ndarray) else stabilized[i].cpu().numpy()
            if m.ndim == 3:
                m = m.squeeze()
            final_masks.append((m > 127).astype(np.uint8) * 255)
    else:
        final_masks = [(m > 0.5).astype(np.uint8) * 255 for m in sam_masks]
        
    return final_masks

def process_videos(video_path, out_path, out_mask_path=None, left_bbox=None, right_bbox=None, prompt_text=None,
    batch_size=100, matte_size=0.4, motion_guided_prompt=False, magnitude=False, meld=False, allthethings=False):

    predictor = build_sam3_video_predictor(
        has_presence_token=False,
        geo_encoder_use_img_cross_attn=True,
        strict_state_dict_loading=False,
        async_loading_frames=True,
        video_loader_type="ffmpeg",
        offload_video_to_cpu = True,
        apply_temporal_disambiguation = True,
        compile = False,
    )

    total_frames, num_keyframes, width, height, duration, fps = metadata(video_path)
    print(f'width', width)
    print(f'height', height)
    cap = cv2.VideoCapture(video_path)
    writer = ffmpeg_pipe(out_path, width, height, fps)
    mask_writer = ffmpeg_pipe(out_mask_path, width, height, fps) if out_mask_path else None
    half_w = width // 2
    packer = AlphaCornerPacker(scale_factor=matte_size)
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights, progress=False).to(device).eval()
    frame_count = 0
    pbar = tqdm(total=total_frames, desc="Processing SBS Batches")

    last_mask_l = None
    last_mask_r = None

    if mask_writer is not None:
        red_sbs = np.zeros((height, width, 3), dtype=np.uint8)
        full_mask_l = np.empty((height, half_w), dtype=np.uint8)
        full_mask_r = np.empty((height, half_w), dtype=np.uint8)

    while frame_count < total_frames:
        frames_bgr = []
        for _ in range(batch_size):
            ret, frame = cap.read()
            if not ret: break
            frames_bgr.append(frame)
            
        if not frames_bgr:
            break
            
        chunk = len(frames_bgr)
        frames_l_bgr = [f[:, :half_w] for f in frames_bgr]
        frames_r_bgr = [f[:, half_w:] for f in frames_bgr]

        track_size = 1008
        track_h, track_w = target_shape((height, half_w), track_size)
        
        frames_l = [cv2.resize(f, (track_w, track_h), interpolation=cv2.INTER_AREA) for f in frames_l_bgr]
        frames_r = [cv2.resize(f, (track_w, track_h), interpolation=cv2.INTER_AREA) for f in frames_r_bgr]
        
        scale_w = track_w / half_w
        scale_h = track_h / height
        
        left_bbox_small = [
            left_bbox[0] * scale_w,
            left_bbox[1] * scale_h,
            left_bbox[2] * scale_w,
            left_bbox[3] * scale_h
        ] if left_bbox is not None else None
        
        right_bbox_small = [
            right_bbox[0] * scale_w,
            right_bbox[1] * scale_h,
            right_bbox[2] * scale_w,
            right_bbox[3] * scale_h
        ] if right_bbox is not None else None
        
        valid_prior_l = last_mask_l if (last_mask_l is not None and np.sum(last_mask_l) > 0) else None
        eye_frames_pil_l = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_l]
        
        if motion_guided_prompt:
            
            masks_l = process_batch(
                predictor=predictor,
                frames_pil=eye_frames_pil_l,
                frames_bgr=frames_l,
                prompt_text=prompt_text,
                bbox=left_bbox_small,
                prior_mask=valid_prior_l
            )
        else:
            masks_l = propagate_eye(                           
                predictor=predictor,
                frames_pil=eye_frames_pil_l,
                frames_bgr=frames_l,
                prompt_text=prompt_text,
                bbox=left_bbox_small,
                prior_mask=valid_prior_l
                )
        
        torch.cuda.empty_cache()

        valid_prior_r = last_mask_r if (last_mask_r is not None and np.sum(last_mask_r) > 0) else None
        eye_frames_pil_r = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_r]

        if motion_guided_prompt:
            
            masks_r = process_batch(
                predictor=predictor,
                frames_pil=eye_frames_pil_r,
                frames_bgr=frames_r,
                prompt_text=prompt_text,
                bbox=right_bbox_small,
                prior_mask=valid_prior_r
            )
        else:
            masks_r = propagate_eye(                
                predictor=predictor,
                frames_pil=eye_frames_pil_r,
                frames_bgr=frames_r,
                prompt_text=prompt_text,
                bbox=right_bbox_small,
                prior_mask=valid_prior_r
                )
        torch.cuda.empty_cache()
        
        last_mask_l = masks_l[-1]
        last_mask_r = masks_r[-1]

        masks_l = apply_effects(masks_l, dilation=0, feather_radius=0.5, smooth_edges=1)
        masks_r = apply_effects(masks_r, dilation=0, feather_radius=0.5, smooth_edges=1)

        for i in range(chunk):
            packed_frame = packer.pack_frame(frames_bgr[i], masks_l[i],masks_r[i])
            writer.stdin.write(packed_frame.astype(np.uint8).tobytes())
            
            if mask_writer is not None:
                cv2.resize(masks_l[i], (half_w, height), dst=full_mask_l, interpolation=cv2.INTER_LINEAR)
                cv2.resize(masks_r[i], (half_w, height), dst=full_mask_r, interpolation=cv2.INTER_LINEAR)
                red_sbs[:, :half_w, 2] = full_mask_l
                red_sbs[:, half_w:, 2] = full_mask_r
                mask_writer.stdin.write(red_sbs.tobytes())
            
        frame_count += chunk
        pbar.update(chunk)
        
    cap.release()
    writer.stdin.close()
    writer.wait()

    if mask_writer is not None:
        mask_writer.stdin.close()
        mask_writer.wait()
    print("Stereoscopic chunked processing complete!")

def process_batch(predictor, frames_pil, frames_bgr, prompt_text=None, bbox=None, prior_mask=None):
    
    weights = Raft_Small_Weights.DEFAULT
    transforms = weights.transforms()
    model = raft_small(weights=weights, progress=False).to(device).eval()
    height, width = frames_bgr[0].shape[:2]
    chunk = len(frames_pil)

    res_inline = predictor.handle_request(dict(
        type="start_session",
        resource_path=frames_pil))

    sid_inline = res_inline["session_id"]
    if prior_mask is not None:
        predictor.handle_request(dict(
            type="add_new_mask",
            session_id=sid_inline,
            frame_index=0,
            obj_id=0,
            mask=prior_mask))
        
    prompt_req_inline = dict(type="add_prompt", session_id=sid_inline, frame_index=0, obj_id=0)

    if prompt_text is not None:
        prompt_req_inline["text"] = prompt_text

    if bbox is not None:
        prompt_req_inline["bounding_boxes"] = [bbox]
        prompt_req_inline["bounding_box_labels"] = [1]

    if prompt_text is not None or bbox is not None or prior_mask is None:
        predictor.handle_request(prompt_req_inline)

    session_inline = predictor._get_session(sid_inline)
    inference_state = session_inline["state"]
    tracker_states = inference_state["tracker_inference_states"]

    if len(tracker_states) == 0:
        print(f"[WARNING] Prompt '{prompt_text}' found no objects in this chunk. Generating empty masks.")
        predictor.handle_request(dict(type="close_session", session_id=sid_inline))
        return [np.zeros((height, width), dtype=np.uint8) for _ in range(chunk)]

    tracker_state = tracker_states[0]
    tensors_rgb = []

    for f_bgr in frames_bgr:
        f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
        t_rgb = torch.from_numpy(f_rgb).permute(2, 0, 1).float().div(255.0).to(device)
        tensors_rgb.append(t_rgb)

    prev_logits = tracker_state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(device).float()
    batch_size = len(tracker_state["obj_ids"])
    predictor.model.tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        for frame_idx in range(1, chunk):
            prev_tensor = tensors_rgb[frame_idx - 1]
            curr_tensor = tensors_rgb[frame_idx]
            predictor.model._prepare_backbone_feats(inference_state=inference_state, frame_idx=frame_idx, reverse=False)

            _, _, h_mask, w_mask = prev_logits.shape

            flow_downscaled = compute_raft_flow(
                curr_tensor.unsqueeze(0), 
                prev_tensor.unsqueeze(0), 
                target_size=(h_mask, w_mask)
            ).squeeze(0)
            
            warped_logits = warp_frame(prev_logits, flow_downscaled)

            dummy_point_inputs = {
                "point_coords": torch.zeros(batch_size, 1, 2, device=device),
                "point_labels": -torch.ones(batch_size, 1, dtype=torch.int32, device=device)
            }

            current_out, _ = predictor.model.tracker._run_single_frame_inference(
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
            predictor.model.tracker._add_output_per_object(tracker_state, frame_idx, current_out, "non_cond_frame_outputs")

            tracker_state["frames_already_tracked"][frame_idx] = {"reverse": False}
            prev_logits = current_out["pred_masks"].to(device).float()

    final_masks = []

    for i in range(chunk):
        storage_key = "cond_frame_outputs" if i == 0 else "non_cond_frame_outputs"
        out = tracker_state["output_dict"][storage_key][i]
        logits_gpu = out["pred_masks_high_res"].to(device) if "pred_masks_high_res" in out else out["pred_masks"].to(device)

        if logits_gpu.shape[0] > 0:
            logits_gpu = torch.max(logits_gpu, dim=0, keepdim=True).values
        else:
            logits_gpu = torch.zeros((1, 1, height, width), device=device)

        logits_resized = torch.nn.functional.interpolate(
            logits_gpu,
            size=(height, width),
            mode="bilinear",
            align_corners=False).squeeze(0).squeeze(0)
        
        prob = torch.sigmoid(logits_resized)
        final_masks.append(((prob > 0.5) * 255).to(torch.uint8).cpu().numpy())

    predictor.handle_request(dict(type="close_session", session_id=sid_inline))
    return final_masks

def process_directory(input_dir, output_dir, **kwargs):
    os.makedirs(output_dir, exist_ok=True)

    video_files = []
    for ext in ["*.mp4", "*.mkv", "*.mov", "*.avi"]:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not video_files:
        print(f"No videos found in {input_dir}")
        return
        
    print(f"Found {len(video_files)} videos in {input_dir}")
    
    for i, video_path in enumerate(video_files):
        filename = os.path.basename(video_path)
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{base_name}_masked.mp4")
        out_mask_path = os.path.join(output_dir, f"{base_name}_redmask.mp4")
        
        print(f"\n=======================================================")
        print(f"[{i+1}/{len(video_files)}] Processing: {filename}")
        print(f"=======================================================")
        
        if os.path.exists(out_path) and os.path.exists(out_mask_path):
            print(f"Skipping {filename}, outputs already exist.")
            continue
            
        try:
            process_videos(
                video_path=video_path,
                out_path=out_path,
                out_mask_path=out_mask_path,
                **kwargs
            )
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    INPUT_FOLDER = "assets/video_segments"
    OUTPUT_FOLDER = "assets/matted_segments"
    
    process_directory(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        prompt_text="One girl",
        batch_size=50,
        matte_size=0.4,
        motion_guided_prompt=True
    )

    # if apply_temporal_disambiguation:
    #     model = Sam3VideoInferenceWithInstanceInteractivity(
    #         detector=detector,
    #         tracker=tracker,
    #         score_threshold_detection=0.65,
    #         assoc_iou_thresh=0.3,
    #         det_nms_thresh=0.1,
    #         new_det_thresh=0.99,
    #         hotstart_delay=8,
    #         hotstart_unmatch_thresh=5,
    #         hotstart_dup_thresh=5,
    #         suppress_unmatched_only_within_hotstart=False,
    #         min_trk_keep_alive=-1,
    #         max_trk_keep_alive=100,
    #         init_trk_keep_alive=5,
    #         suppress_overlapping_based_on_recent_occlusion_threshold=0.9,
    #         suppress_det_close_to_boundary=True,
    #         fill_hole_area=4,
    #         recondition_every_nth_frame=64,
    #         masklet_confirmation_enable=True,
    #         decrease_trk_keep_alive_for_empty_masklets=True,
    #         image_size=1008,
    #         image_mean=(0.5, 0.5, 0.5),
    #         image_std=(0.5, 0.5, 0.5),
    #         compile_model=compile,
    #     )
