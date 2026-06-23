import logging, os, glob, json
import torch.nn.functional as F
from tqdm import tqdm
from imagemask import *
from model_builder import build_sam3_video_predictor
import torchvision.transforms.functional as V
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def _setup_tf32() -> None:
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

_setup_tf32()

def have(a):
    return a is not None  

def aorb(a, b):
    return a if have(a) else b

def aborc(a, b, c):
    return aorb(a, aorb(b, c))

def abcord(a, b, c, d):
    return aorb(a, aborc(b, c, d))

def metadata(path):
    cmd_key = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'frame=pict_type', '-of', 'csv=p=0', '-skip_frame', 'nokey', path]
    res_key = subprocess.run(cmd_key, capture_output=True, text=True)
    lines = res_key.stdout.strip().split('\n')
    keyframes = len(lines)
    cmd_stream = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', '-select_streams', 'v:0', path]
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
        frames = int(f_tot)
    else:
        frames = int(duration * fps) if duration > 0 else 0
        
    return frames, keyframes, width, height, duration, fps

def ffmpeg_pipe(out_path, width, height, fps):

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-', '-c:v', 'hevc_qsv', '-profile:v', 'main10', '-pix_fmt', 'p010le', '-tag:v', 'hvc1', '-g', '100', '-b:v', '100M', '-preset', 'fast',
        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-fps_mode', 'cfr', '-r', str(fps), '-movflags', '+faststart+write_colr+use_metadata_tags',
        '-metadata:s:v:0', 'stereo_mode=left_right', '-color_trc', 'bt709', out_path
    ]
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

class raft_flow:
    def __init__(self, device, max_size=1008, flow_scale=1.0, interp_mode="bicubic"):

        try:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights # from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            HAS_RAFT = True
        except ImportError:
            HAS_RAFT = False

        self.device = torch.device(device) if isinstance(device, str) else (device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.max_size = max_size
        self.flow_scale = flow_scale
        self.interp_mode = interp_mode
        self.weights = Raft_Small_Weights.DEFAULT
        self.model = raft_small(weights=self.weights, progress=False).to(self.device).eval()
        self.transforms = self.weights.transforms()     
 
    def compute_raft_flow(self, img1a, img2a, max_size, scale, interp_mode, target_size):

        origH, origW =  V.get_image_size(img1a)
        current_H, current_W = origH * scale, origW * scale 

        if max(current_H, current_W) > max_size:
            scale = scale * (max_size / float(max(current_H, current_W)))

        if scale != 1.0:
            newH, newW = int(origH * scale), int(origW * scale)
            img1b = F.interpolate(img1a, size=(newH, newW), mode=interp_mode, antialias=True)
            img2b = F.interpolate(img2a, size=(newH, newW), mode=interp_mode, antialias=True)
        else:
            newH, newW = origH, origW
            img1b, img2b = img1a, img2a

        img1c, img2c = self.transforms(img1b, img2b)
        _, _, H_s, W_s = img1c.shape
        padh, padw = (8 - H_s % 8) % 8, (8 - W_s % 8) % 8

        if padh > 0 or padw > 0:
            img1c = F.pad(img1c, (0, padw, 0, padh))
            img2c = F.pad(img2c, (0, padw, 0, padh))

        flow = self.model(img1c, img2c)[-1].float()
        if padh > 0 or padw > 0:
            flow = flow[:, :, :H_s, :W_s]

        out_H, out_W = target_size if target_size else (origH, origW)
        if out_H != H_s or out_W != W_s:
            flow = F.interpolate(flow, size=(out_H, out_W), mode=interp_mode, antialias=True)
            flow[:, 0] *= (out_W / W_s)
            flow[:, 1] *= (out_H / H_s)
                
        return flow

    def warp_frame(self, a, b, scale=1.0, interp_mode="bicubic", N=None):
    
        if a.ndim == 3: 
            C, H, W = a.shape 
        if a.ndim == 4: 
            N, C, H, W = a.shape
        
        scaled = b * scale
        y, x = torch.meshgrid(torch.arange(H, device=self.device), torch.arange(W, device=self.device), indexing='ij')
        x_norm = 2.0 * (x + scaled[0]) / max(W - 1, 1) - 1.0
        y_norm = 2.0 * (y + scaled[1]) / max(H - 1, 1) - 1.0
        grid = torch.stack((x_norm, y_norm), dim=-1).unsqueeze(0)
        grid = grid.expand(N, -1, -1, -1) if have(N) else grid
        return F.grid_sample(a, grid, mode=interp_mode, padding_mode='border', align_corners=True) if have(N) else F.grid_sample(a.unsqueeze(0), grid, mode=interp_mode, padding_mode='border', align_corners=True).squeeze(0)

class AlphaPacker:
    def __init__(self, scale=0.40, padding=0):

        self.scale = scale
        self.padding = padding
        self.vignette_cache = None

    def pack_frame(self, frames, mask_l = None, mask_r = None, sbs=False):

        H, SBS_W, C = frames.shape
        half_W = SBS_W // 2

        if mask_l.dtype != np.uint8:
            mask_l = (mask_l * 255).astype(np.uint8)
            mask_r = (mask_r * 255).astype(np.uint8)

        target_w = int(half_W * self.scale)
        target_h = int(H * self.scale)
        
        if mask_l.shape[:2] != (target_h, target_w):
            l_small = cv2.resize(mask_l, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            l_small = mask_l

        if mask_r.shape[:2] != (target_h, target_w):
            r_small = cv2.resize(mask_r, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            r_small = mask_r

        mask_l_vignette = l_small.astype(np.uint8) 
        mask_r_vignette = r_small.astype(np.uint8) 

        packed_frame = frames
        h_half = target_h // 2
        top_half_mask = mask_l_vignette[:h_half, :]
        bottom_half_mask = mask_l_vignette[h_half:h_half*2, :]

        w_half = target_w // 2
        q_tl_mask = mask_r_vignette[:h_half, :w_half]
        q_tr_mask = mask_r_vignette[:h_half, w_half:w_half*2]
        q_bl_mask = mask_r_vignette[h_half:h_half*2, :w_half]
        q_br_mask = mask_r_vignette[h_half:h_half*2, w_half:w_half*2]

        def blend_white_mask(roi, mask_1ch):
            inv_mask_3d = (255 - mask_1ch)[..., np.newaxis]
            blended = (roi.astype(np.uint16) * inv_mask_3d) // 255
            blended += mask_1ch[..., np.newaxis]
            return blended.astype(np.uint8)

        y1_top = self.padding
        y2_top = y1_top + h_half
        x1_mid = (SBS_W // 2) - (target_w // 2)
        x2_mid = x1_mid + target_w
        
        packed_frame[y1_top:y2_top, x1_mid:x2_mid] = blend_white_mask(packed_frame[y1_top:y2_top, x1_mid:x2_mid], bottom_half_mask)
        y1_bot = H - self.padding - h_half
        y2_bot = y1_bot + h_half
        packed_frame[y1_bot:y2_bot, x1_mid:x2_mid] = blend_white_mask(packed_frame[y1_bot:y2_bot, x1_mid:x2_mid], top_half_mask)

        y1_tr = self.padding
        y2_tr = y1_tr + h_half
        x1_tr = SBS_W - self.padding - w_half
        x2_tr = SBS_W - self.padding
        packed_frame[y1_tr:y2_tr, x1_tr:x2_tr] = blend_white_mask(packed_frame[y1_tr:y2_tr, x1_tr:x2_tr], q_bl_mask)

        y1_tl_l = self.padding
        y2_tl_l = y1_tl_l + h_half
        x1_tl_l = self.padding
        x2_tl_l = self.padding + w_half
        packed_frame[y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l] = blend_white_mask(packed_frame[y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l], q_br_mask)

        y1_br_r = H - self.padding - h_half
        y2_br_r = y1_br_r + h_half
        x1_br_r = SBS_W - self.padding - w_half
        x2_br_r = SBS_W - self.padding
        packed_frame[y1_br_r:y2_br_r, x1_br_r:x2_br_r] = blend_white_mask(packed_frame[y1_br_r:y2_br_r, x1_br_r:x2_br_r], q_tl_mask)

        y1_bl_l = H - self.padding - h_half
        y2_bl_l = y1_bl_l + h_half
        x1_bl_l = self.padding
        x2_bl_l = self.padding + w_half
        packed_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l] = blend_white_mask(packed_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l], q_tr_mask)
        return packed_frame

def process_frames(predictor, raft, frames_pil=None, resized_frames=None, full_frames=None, prompt_text=None, prior_mask=None, warp=False):

    chunk = len(frames_pil)
    height, width, _ = full_frames[0].shape

    res = predictor.handle_request(dict(
        type="start_session",
        resource_path=frames_pil
        ))

    sid = res["session_id"]
    
    if prior_mask is not None:
        predictor.handle_request(dict(
            type="add_new_mask",
            session_id=sid,
            frame_index=0,
            obj_id=0,
            mask=prior_mask
            ))
 
    prompt_req = dict(type="add_prompt", session_id=sid, frame_index=0, obj_id=0)

    prompt_req["text"] = prompt_text
    predictor.handle_request(prompt_req)
    session = predictor._get_session(sid)
    inference_state = session["state"]
    states = inference_state["tracker_inference_states"]

    if len(states) == 0:
        print(f"[GASP!] OH NO! Prompt '{prompt_text}' found no chunky objects..")
        predictor.handle_request(dict(type="close_session", session_id=sid))
        empty = [np.zeros((height, width), dtype=np.uint8) for _ in range(chunk)]
        return empty, empty

    state = states[0]

    if warp:
        frames_stack = np.stack(resized_frames) # [B, H, W, 3] BGR
        frames_rgb = frames_stack[..., ::-1].transpose(0, 3, 1, 2) # [B, 3, H, W] RGB
        tensors_rgb = torch.from_numpy(np.ascontiguousarray(frames_rgb)).to(device, dtype=torch.float32) / 255.0
    
    prev_logits = state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(device).float()
    batch_size = len(state["obj_ids"])
    predictor.model.tracker.propagate_in_video_preflight(state, run_mem_encoder=True)

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for frame_idx in range(1, chunk):
            if warp:
                prev_tensor = tensors_rgb[frame_idx - 1]
                curr_tensor = tensors_rgb[frame_idx]

            predictor.model._prepare_backbone_feats(
                inference_state=inference_state,
                frame_idx=frame_idx,
                reverse=False)

            _, _, h_mask, w_mask = prev_logits.shape

            if warp:
                _, _, frame_H, frame_W = prev_tensor.unsqueeze(0).shape

                flow = raft.compute_raft_flow(
                    prev_tensor.unsqueeze(0),
                    curr_tensor.unsqueeze(0),
                    max_size=max(frame_H, frame_W),
                    scale=1.0,
                    interp_mode="bicubic",
                    target_size=(h_mask, w_mask)).squeeze(0)

                flow[0] *= (w_mask / height)
                flow[1] *= (h_mask / height)

                warped_logits = raft.warp_frame(prev_logits, flow)
            else:
                warped_logits = prev_logits

            dummy_point_inputs = {
                "point_coords": torch.zeros(batch_size, 1, 2, device=device),
                "point_labels": -torch.ones(batch_size, 1, dtype=torch.int32, device=device)
            }

            current_out, _ = predictor.model.tracker._run_single_frame_inference(
                inference_state=state,
                output_dict=state["output_dict"],
                frame_idx=frame_idx,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=dummy_point_inputs,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=warped_logits,
            )

            if warp:
                if current_out["pred_masks"].max() < 0.0:
                    if warped_logits.max() > 0.0:
                        current_out["pred_masks"] = warped_logits
                    else:
                        current_out["pred_masks"] = prev_logits
                    if "pred_masks_high_res" in current_out:
                        del current_out["pred_masks_high_res"]

            state["output_dict"]["non_cond_frame_outputs"][frame_idx] = current_out
            predictor.model.tracker._add_output_per_object(state, frame_idx, current_out, "non_cond_frame_outputs")
            state["frames_already_tracked"][frame_idx] = {"reverse": False}
            prev_logits = current_out["pred_masks"].to(device).float()

    hard_masks = []
    soft_masks = []

    for i in range(chunk):
        storage_key = "cond_frame_outputs" if i == 0 else "non_cond_frame_outputs"
        out = state["output_dict"][storage_key][i]
        logits_gpu = out["pred_masks_high_res"].to(device) if "pred_masks_high_res" in out else out["pred_masks"].to(device)

        if logits_gpu.shape[0] > 0:
            logits_gpu = torch.max(logits_gpu, dim=0, keepdim=True).values
        else:
            logits_gpu = torch.zeros((1, 1, height, width), device=device)

        logits_resized = torch.nn.functional.interpolate(
            logits_gpu,
            size=(height, width),
            mode="bicubic",
            align_corners=False).squeeze(0).squeeze(0)
        
        prob = torch.sigmoid(logits_resized)
        soft_masks.append(prob)
        hard_masks.append(((prob > 0.5) * 255).to(torch.uint8).cpu().numpy())

    if warp:
        hard_masks = [m.copy() for m in hard_masks]
        valid_idx = [i for i, m in enumerate(hard_masks) if np.sum(m) > 0]
        
        if 0 < len(valid_idx) < chunk:
            for i in range(chunk):
                if np.sum(hard_masks[i]) == 0:
                    print(f'No Mask!', i)
                    prev_i = next((j for j in reversed(valid_idx) if j < i), None)
                    next_i = next((j for j in valid_idx if j > i), None)
                    if prev_i is not None and next_i is not None:
                        dist_prev = i - prev_i
                        dist_next = next_i - i
                        w_prev = dist_next / (dist_prev + dist_next)
                        w_next = dist_prev / (dist_prev + dist_next)
                        blended = (hard_masks[prev_i].astype(np.float32) * w_prev + 
                                hard_masks[next_i].astype(np.float32) * w_next)
                        hard_masks[i] = (blended > 127).astype(np.uint8) * 255
                    elif prev_i is not None:
                        hard_masks[i] = hard_masks[prev_i] 
                    elif next_i is not None:
                        hard_masks[i] = hard_masks[next_i] 

    predictor.handle_request(dict(type="close_session", session_id=sid))
    return hard_masks, soft_masks

def process_videos(video_path, out_path, out_mask_path, prompt_text, batch_size, matte_size, warp, full_sbs=False, debug=None):  
    bbox=None

    predictor = build_sam3_video_predictor(
        gpus_to_use=None,
        has_presence_token=False,
        geo_encoder_use_img_cross_attn=False,
        strict_state_dict_loading=False,
        async_loading_frames=True,
        video_loader_type="ffmpeg",
        offload_video_to_cpu = True,
        apply_temporal_disambiguation = True,
        compile = False,
    )

    raft = raft_flow(device="cuda") if warp else None
    frames_tot, keyframes, width, height, duration, fps = metadata(video_path)
    gen = video_frame_generator(video_path, force_rate=0, frame_load_cap=debug or 0, skip_first_frames=0, select_every_nth=1, output_format="bgr24")
    _ = next(gen) 

    writer = ffmpeg_pipe(out_path, width, height, fps) if out_path else None
    mask_writer = ffmpeg_pipe(out_mask_path, width, height, fps) if out_mask_path else None
    half_w = width // 2
    packer = AlphaPacker(scale=matte_size)

    frame_count = 0
    frames_tot = frames_tot if debug is None else debug
    pbar = tqdm(total=frames_tot, desc="Processing .. beep.boop.bop.. beep.")

    last_l = None
    last_r = None

    while frame_count < frames_tot:
        frames = []
      
        for _ in range(keyframes if batch_size is None else batch_size):
            try:
                frame_bgr = next(gen)
                frames.append(frame_bgr)
            except StopIteration:
                break
        if not frames:
            break
        
        chunk = len(frames) 

        if full_sbs:

            prior_frames = last_l if (last_l is not None and np.sum(last_l) > 0) else None
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in  frames]       

            masks, _ = process_frames(
                predictor=predictor,
                raft=raft,
                frames_pil=pil_frames,
                resized_frames=frames,
                full_frames=frames,
                prompt_text=prompt_text,
                prior_mask=prior_frames,
                warp=warp
                )

            last_l = masks[-1] if masks is not None else None
            masks = process_mask(masks, sensitivity=1.0, mask_blur=0, mask_offset=0, smooth=3, 
                            fill_holes=False, invert_output=False, dilation=0, feather_radius=1.1, smooth_edges=0)

            masks_r = [f[:, half_w:] for f in masks]
            masks_l = [f[:, :half_w] for f in masks]

        else:
            frames_l = [f[:, :half_w] for f in frames]
            prior_l = last_l if (last_l is not None and np.sum(last_l) > 0) else None
            pil_l = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_l]       

            masks_l, soft_l = process_frames(
                predictor=predictor,
                raft=raft,
                frames_pil=pil_l,
                resized_frames=frames_l,
                full_frames=frames_l,
                prompt_text=prompt_text,
                prior_mask=prior_l,
                warp=warp
                )
            
            frames_r = [f[:, half_w:] for f in frames]
            prior_r = last_r if (last_r is not None and np.sum(last_r) > 0) else None
            pil_r = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_r]

            masks_r, soft_r = process_frames(
                predictor=predictor,
                raft=raft,
                frames_pil=pil_r,
                resized_frames=frames_r,
                full_frames=frames_r,
                prompt_text=prompt_text,
                prior_mask=prior_r,
                warp=warp
                )

            last_l = masks_l[-1] if masks_l is not None else None
            last_r = masks_r[-1] if masks_r is not None else None

        for i in range(chunk):
            if full_sbs:
                packed_frame = packer.pack_frame(frames[i], masks_l[i], masks_r[i], sbs=full_sbs)
                writer.stdin.write(packed_frame.astype(np.uint8).tobytes())         
                full_sbs_img = resize_mask(torch.from_numpy(masks[i]).unsqueeze(0).unsqueeze(0), (height, width)).squeeze().numpy().astype(np.uint8)
                sbs_out = np.stack([full_sbs_img, full_sbs_img, full_sbs_img], axis=-1)
                if mask_writer is not None:
                    mask_writer.stdin.write(sbs_out.tobytes())
            else:
                packed_frame = packer.pack_frame(frames[i], masks_l[i], masks_r[i], sbs=full_sbs)
                writer.stdin.write(packed_frame.astype(np.uint8).tobytes())
                full_mask_l = resize_mask(torch.from_numpy(masks_l[i]).unsqueeze(0).unsqueeze(0), (height, half_w)).squeeze().numpy().astype(np.uint8)
                full_mask_r = resize_mask(torch.from_numpy(masks_r[i]).unsqueeze(0).unsqueeze(0), (height, half_w)).squeeze().numpy().astype(np.uint8)
                red_sbs = np.zeros((height, width, 3), dtype=np.uint8)
                red_sbs[:, :half_w, 2] = full_mask_l
                red_sbs[:, half_w:, 2] = full_mask_r
                mask_writer.stdin.write(red_sbs.tobytes())

        frame_count += chunk
        pbar.update(chunk)
        
    if gen: gen.close()
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    if mask_writer is not None:
        mask_writer.stdin.close()
        mask_writer.wait()
        
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
        out_path = os.path.join(output_dir, f"{base_name}_ALPHA.mp4")
        out_mask_path = os.path.join(output_dir, f"{base_name}_mask.mp4")
        
        print(f"\n=======================================================")
        print(f"[{i+1}/{len(video_files)}] Processing: {filename}")
        print(f"=======================================================")
        
        if os.path.exists(out_path) and os.path.exists(out_mask_path):
            print(f"Skipping {filename}, outputs already exist.")
            continue
            
        process_videos(
            video_path=video_path,
            out_path=out_path,
            out_mask_path=out_mask_path,
            **kwargs
        )

if __name__ == "__main__":
    INPUT_FOLDER = "assets/video_segments"
    OUTPUT_FOLDER = "assets/matted_segments"
    
    process_directory(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        prompt_text="One girl",
        batch_size=100,
        matte_size=0.4,
        warp=False,
        full_sbs=True,
        debug=None,
    )
