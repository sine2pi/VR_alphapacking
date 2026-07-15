
from torch import set_default_dtype
from masksandthings import *
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
set_default_dtype(dtype)
gpus_to_use = [torch.cuda.current_device()]

def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print(device_props)

_setup_tf32()

def build_sam3_video_predictor(*model_args, checkpoint_path=None, bpe_path=None, gpus_to_use=None, is_sbs=False,  max_num_objects=1, num_obj_for_compile=1, strict_state_dict_loading=False, **model_kwargs):
    from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
    predictor = Sam3VideoPredictorMultiGPU(*model_args, checkpoint_path=checkpoint_path, gpus_to_use=gpus_to_use, is_sbs=is_sbs, max_num_objects= max_num_objects, num_obj_for_compile=num_obj_for_compile, strict_state_dict_loading=strict_state_dict_loading, **model_kwargs)
    return predictor

def ffmpeg_pipe(out_path, width, height, fps, audio_source=None):

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-']
    
    if audio_source:
        ffmpeg_cmd.extend(['-i', audio_source, '-map', '0:v', '-map', '1:a?'])
        
    ffmpeg_cmd.extend([
        '-sws_flags', 'lanczos+full_chroma_int+accurate_rnd+full_chroma_inp', '-c:v', 'hevc_qsv', '-profile:v', 'main10', '-pix_fmt', 'p010le', '-tag:v', 'hvc1', '-g', '100', '-b:v', '100M', '-preset', 'medium', '-aspect', '2:1', '-copyts', '-start_at_zero', '-bitexact', '-c:a', 'aac', '-b:a', '256k', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-fps_mode', 'cfr', '-r', str(fps), '-movflags', '+faststart+write_colr+use_metadata_tags', '-metadata:s:v:0', 'stereo_mode=left_right', '-color_trc', 'bt709', out_path])

    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

class ToddPacker:
    def __init__(n, scale=0.40, padding=0, circle=False):

        n.scale = scale
        n.padding = padding
        n.circle = circle
        n._cache = None

    def pack_frame(n, frames, mask_l = None, mask_r = None):

        H, SBS_W, C = frames.shape
        W = SBS_W // 2

        target_w = int(W * n.scale)
        target_h = int(H * n.scale)

        if mask_l.shape[1] != target_h or mask_l.shape[2] != target_w:
            mask_l = F.interpolate(mask_l.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode='area').squeeze(0).squeeze(0)
            mask_r = F.interpolate(mask_r.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode='area').squeeze(0).squeeze(0)

        p_frame = frames
        h_half = target_h // 2
        top_half_mask = mask_l[:h_half, :]
        bottom_half_mask = mask_l[h_half:h_half*2, :]
        w_half = target_w // 2

        q_tl_mask = mask_r[:h_half, :w_half]
        q_tr_mask = mask_r[:h_half, w_half:w_half*2]
        q_bl_mask = mask_r[h_half:h_half*2, :w_half]
        q_br_mask = mask_r[h_half:h_half*2, w_half:w_half*2]

        q_tl_circle = None
        q_tr_circle = None
        q_bl_circle = None 
        q_br_circle = None

        def blend_white_mask(roi, mask_1ch, red=True):
            if red:
                alpha = mask_1ch.unsqueeze(-1)
                red_color = torch.tensor([255.0, 0.0, 0.0], dtype=torch.float32, device=roi.device)
                x = ((1.0 - alpha) * roi.float() + alpha * red_color).to(torch.uint8)
            else:
                alpha = (255 - mask_1ch)[..., None]
                blend = (roi.to(torch.int32) * alpha) // 255
                blend += mask_1ch[..., None]
                x = blend.to(torch.uint8)
            return x

        y1_top = n.padding
        y2_top = y1_top + h_half
        x1_mid = (SBS_W // 2) - (target_w // 2)
        x2_mid = x1_mid + target_w

        p_frame[y1_top:y2_top, x1_mid:x2_mid] = blend_white_mask(p_frame[y1_top:y2_top, x1_mid:x2_mid], bottom_half_mask)
        y1_bot = H - n.padding - h_half
        y2_bot = y1_bot + h_half
        p_frame[y1_bot:y2_bot, x1_mid:x2_mid] = blend_white_mask(p_frame[y1_bot:y2_bot, x1_mid:x2_mid], top_half_mask)

        y1_tr = n.padding
        y2_tr = y1_tr + h_half
        x1_tr = SBS_W - n.padding - w_half
        x2_tr = SBS_W - n.padding
        p_frame[y1_tr:y2_tr, x1_tr:x2_tr] = blend_white_mask(p_frame[y1_tr:y2_tr, x1_tr:x2_tr], q_bl_mask, q_bl_circle)

        y1_tl_l = n.padding
        y2_tl_l = y1_tl_l + h_half
        x1_tl_l = n.padding
        x2_tl_l = n.padding + w_half
        p_frame[y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l] = blend_white_mask(p_frame[y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l], q_br_mask, q_br_circle)

        y1_br_r = H - n.padding - h_half
        y2_br_r = y1_br_r + h_half
        x1_br_r = SBS_W - n.padding - w_half
        x2_br_r = SBS_W - n.padding
        p_frame[y1_br_r:y2_br_r, x1_br_r:x2_br_r] = blend_white_mask(p_frame[y1_br_r:y2_br_r, x1_br_r:x2_br_r], q_tl_mask, q_tl_circle)

        y1_bl_l = H - n.padding - h_half
        y2_bl_l = y1_bl_l + h_half
        x1_bl_l = n.padding
        x2_bl_l = n.padding + w_half
        p_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l] = blend_white_mask(p_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l], q_tr_mask, q_tr_circle)

        return p_frame

def process_frames(predictor, frames, frames_pil=None, prompt_text=None, frame_idx=0, object_id=1, start_frame_idx=0,
 max_frames_to_track=-1, close_after_propagation=True, keep_model_loaded=True, session_id=None, prev_mask=None, positive_coords=None, 
 negative_coords=None, bbox=None, propagation_direction="forward", sam31=False, warp=None, prev_frame=None, matte_size=None, prev_flow=None, max_side=1.0, objects_out=False):

    H, W, C = int(frames[0].shape[0]), int(frames[0].shape[1]), int(frames[0].shape[2])  

    if max_side is not None:
        if min(H, W) < 4096:
            max_side = max_side
        else:
            max_side = 0.5

        if isinstance(max_side, int): 
            H, W = int(H * (max_side / float(min(H, W)))), int(W * (max_side / float(min(H, W))))
        else:
            H, W = int(max_side * H), int(max_side * W)
    else:
         H, W = H, W

    if isinstance(frames, np.ndarray):
        frames = [torch.from_numpy(f).permute(2, 0, 1).float().unsqueeze(0) for f in frames]
    else: 
        frames = [(f).permute(2, 0, 1).float().unsqueeze(0) for f in frames]

    frames = [F.interpolate(f, size=(H, W), mode='bicubic', antialias=True) for f in frames] if aorb(max_side, 1.0) else frames
    frames_pil = [Image.fromarray((f.squeeze(0).permute(1, 2, 0).cpu().numpy() * (255 if f.max() <= 1.0 else 1)).astype('uint8')) for f in frames]
    B, C, H, W = frames[0].shape

    chunk = len(frames_pil)
    if frame_idx > chunk - 1:
        frame_idx = 0

    print(f"Processing frames of size: {W}x{H} ")
    print(f"frame_idx: {frame_idx} batch: {chunk}")
    print(f"Frame shape {frames[0].shape}")

    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=frames_pil,
            session_id=session_id,
            start_frame_idx=start_frame_idx,
            offload_video_to_cpu = True,
            offload_state_to_cpu = True))

    session_id = response.get("session_id", None)
    if session_id is None:
        raise ValueError("Failed to start video prediction session")

    if prev_mask is not None:
        predictor.handle_request(dict(
            type="add_new_mask",
            session_id=session_id,
            frame_idx=frame_idx,
            obj_id=object_id,
            mask=prev_mask))

        s_idx = set()
        sid_inline = response["session_id"]
        session_inline = predictor._get_session(sid_inline)
        inference_state = session_inline["state"]
        tracker_states = inference_state["tracker_inference_states"]

        for state_idx, inference_state in enumerate(tracker_states):
            if (object_id in inference_state["obj_ids"] and frame_idx in inference_state["frames_already_tracked"]):
    
                predictor.model.tracker.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=object_id,
                    mask=prev_mask)
                s_idx.add(state_idx)

        for idx in s_idx:
            predictor.model.tracker.propagate_in_video_preflight(
                tracker_states[idx], run_mem_encoder=True)
        return tracker_states

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pos_points, pos_count, pos_errors = parse_points(positive_coords, frames[0].shape)
        neg_points, neg_count, neg_errors = parse_points(negative_coords, frames[0].shape)
        points = None
        point_labels = None

        if pos_points is not None and neg_points is not None:
            points = pos_points + neg_points
            point_labels = [1] * pos_count + [0] * neg_count
            
        elif pos_points is not None:
            points = pos_points
            point_labels = [1] * pos_count

        elif neg_points is not None:
            points = neg_points
            point_labels = [0] * neg_count

        bounding_boxes = None
        bounding_box_labels = None

        if bbox is not None:
            bbox_coords, bbox_count = parse_bbox(bbox, frames[0].shape)

            if bbox_coords is not None:
                bounding_boxes = bbox_coords
                bounding_box_labels = [1] * bbox_count

        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_idx=frame_idx,
                text=prompt_text if prompt_text else None,
                bounding_boxes=bounding_boxes,
                bounding_box_labels=bounding_box_labels,
                points=points,
                point_labels=point_labels,
                obj_id=object_id))

        hard_masks = []
        objects = {}
        processed_frames = 0

        output = torch.zeros((chunk, H, W), dtype=torch.float32)
        object_outputs = {"obj_ids":None, "obj_masks":[]}
        session = predictor._get_session(session_id)
        inference_state = session["state"]
        num_frames = inference_state["num_frames"]        

        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=propagation_direction,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=num_frames)):

            frame_idx = response.get("frame_idx", 0)
            outputs = response.get("outputs", {})
            obj_ids = outputs.get("out_obj_ids", None)
            
            if obj_ids is not None:
                object_outputs["obj_ids"] = obj_ids

            if warp is not None:
                session = predictor._get_session(session_id)
                inference_state = session["state"]
                states = inference_state["tracker_inference_states"]

                state = states[0]
                tensors_rgb = []

                for f in frames:
                    tensors_rgb.append(f)

                prev_logits = state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(device).float()
                batch_size = len(state["obj_ids"])
                predictor.model.tracker.propagate_in_video_preflight(state, run_mem_encoder=True)
            
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for frame_idx in range(1, chunk):
                        prev_tensor = tensors_rgb[frame_idx - 1]
                        curr_tensor = tensors_rgb[frame_idx]

                        predictor.model._prepare_backbone_feats(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            reverse=False)

                        _, _, h_mask, w_mask = prev_logits.shape
                        _, _, frame_H, frame_W = prev_tensor.unsqueeze(0).shape

                        flow = warp.compute_raft_flow(
                            prev_tensor.unsqueeze(0),
                            curr_tensor.unsqueeze(0),
                            max_size=max(frame_H, frame_W),
                            scale=1.0,
                            interp_mode="bicubic",
                            target_size=(h_mask, w_mask)
                            ).squeeze(0)

                        flow[0] *= (w_mask / H)
                        flow[1] *= (h_mask / W)

                        warped_logits = raft.warp_frame(prev_logits, flow)

                        dummy_point_inputs = {
                            "point_coords": torch.zeros(batch_size, 1, 2, device=device),
                            "point_labels": -torch.ones(batch_size, 1, dtype=torch.int32, device=device)}

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
                            prev_sam_mask_logits=warped_logits)

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
                        logits = current_out["pred_masks"].to(device).float()

            if outputs:
                if "out_mask_logits" in outputs:
                    logits = outputs["out_mask_logits"]

                    if logits.shape[0] > 0:
                        tensor = torch.from_numpy(logits) if isinstance(logits, np.ndarray) else logits
                        prob = torch.sigmoid(tensor)
                        prob_expanded = prob.unsqueeze(0) if prob.dim() == 3 else prob
                        
                        smooth_prob = torch.nn.functional.interpolate(
                            prob_expanded, size=(H, W), mode='bicubic', align_corners=False, antialias=True).squeeze(0)
                        
                        objects[frame_idx] = smooth_prob.to(dtype=torch.float32)
                        merged_prob = torch.max(smooth_prob, dim=0).values
                        merged = (merged_prob * 255).to(dtype=torch.uint8)
                        output[frame_idx] = merged
                        hard_masks.append(output)

                    else:
                        objects[frame_idx] = torch.zeros((1, H, W), dtype=torch.float32, device=tensor.device if 'tensor' in locals() else None)
                        
                elif "out_binary_masks" in outputs:
                    mask = outputs["out_binary_masks"]

                    if mask.shape[0] > 0:
                        tensor_mask = torch.from_numpy(mask).float() if isinstance(mask, np.ndarray) else mask.float()
                        mask_expanded = tensor_mask.unsqueeze(0) if tensor_mask.dim() == 3 else tensor_mask
                        
                        smooth_mask = torch.nn.functional.interpolate(
                            mask_expanded, size=(H, W), mode='bicubic', align_corners=False, antialias=True).squeeze(0)
                        
                        objects[frame_idx] = smooth_mask
                        merged_prob = torch.max(smooth_mask, dim=0).values
                        output[frame_idx] = (merged_prob * 255).to(dtype=torch.uint8)
                        hard_masks.append(output)
                    else:
                        objects[frame_idx] = torch.zeros((1, H, W), dtype=torch.float32)
                else:
                    objects[frame_idx] = torch.zeros((1, H, W), dtype=torch.float32)

            if len(objects) > 0 and objects_out:
                max_objects = max(mask.shape[0] for mask in objects.values())

                ordered = []
                padded = []

                for frame_idx in range(chunk):
                    if frame_idx in objects:
                        mask = objects[frame_idx]  
                        num_objects = mask.shape[0]

                        if num_objects < max_objects:
                            padding = np.zeros((max_objects - num_objects, H, W), dtype=np.float32)
                            padded_mask = np.concatenate([mask, padding], axis=0)
                            ordered.append(padded_mask)
                            padded.append(torch.from_numpy(padded_mask))
                        else:
                            ordered.append(mask)
                            padded.append(torch.from_numpy(mask))
                    else:
                        empty = np.zeros((max_objects, H, W), dtype=np.float32)
                        ordered.append(empty)
                        padded.append(torch.zeros((max_objects, H, W)))

                object_masks = torch.stack(padded, dim=0)
                object_outputs["obj_masks"] = ordered
            else:
                object_masks = torch.zeros((C, H, W))
                object_outputs["obj_masks"] = []
            
            processed_frames += 1

        if close_after_propagation:
            predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id))

        if not keep_model_loaded and close_after_propagation:
            predictor.shutdown()

    return output

def process_allthethings(video_path1, video_path2, out_path, mask_path, prompt_text=None, 
batch_size=None, matte_size=None, warp=False, debug=None, sam31=False):  

    if sam31:
        origin_ckpt = torch.load("assets/sam3.1_multiplex.pt")
        mapped_ckpt = {}

        for k, v in origin_ckpt.items():
            if k.startswith("tracker.model."):
                mapped_ckpt[k.replace("tracker.model.", "")] = v
            elif k.startswith("detector."):
                mapped_ckpt[k.replace("detector.", "")] = v
            mapped_ckpt[k] = v

        torch.save(mapped_ckpt, "assets/sam3.1_multiplex_mapped.pt")
        checkpoint_path = "assets/sam3.1_multiplex_mapped.pt"

        from sam3.model_builder import build_sam3_multiplex_video_predictor
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
            bpe_path="assets/bpe_simple_vocab_16e6.txt.gz",
            max_num_objects = 2,
            multiplex_count = 1,
            use_fa3 = False,
            use_rope_real = False,
            compile = False,
            warm_up = False,
            session_expiration_sec = 5000,
            default_output_prob_thresh = 0.5,
            async_loading_frames = True)

    else:
        predictor = build_sam3_video_predictor(
            checkpoint_path = "assets/sam3.pt",
            bpe_path="assets/bpe_simple_vocab_16e6.txt.gz",
            gpus_to_use = None,
            has_presence_token = False,
            geo_encoder_use_img_cross_attn = False,
            strict_state_dict_loading = False,
            async_loading_frames = True,
            video_loader_type = "cv2",
            apply_temporal_disambiguation = True,
            compile = False,
            max_num_objects=1,
            num_obj_for_compile=1) 
    
    decoder = VFG(video_path1, force_rate=0, frame_load_cap=debug or 0, skip_first_frames=0, select_every_nth=1, output_format="rgb24")
    width, height, fps, duration, total_frames, target_frame_time, out_frame = next(decoder)  
    half_w = width // 2
    raft = raft_flow(device="cuda") if warp else None

    writer = ffmpeg_pipe(out_path, width, height, fps, audio_source=video_path1) if out_path else None
    mask_writer = ffmpeg_pipe(mask_path, width, height, fps) if mask_path else None
    packer = ToddPacker(scale=matte_size, circle=False)

    frame_count = 0
    total_frames = total_frames if debug is None else debug
    batch_size = total_frames if batch_size == 0 else batch_size
    pbar = tqdm(total=total_frames, desc="Processing .. beep.boop.bop.. beep.")

    while frame_count < total_frames:
        frames = []
        mattes = []
        for _ in range(keyframes if batch_size is None else batch_size):
            try:
                out_frame = next(decoder)
                frames.append(out_frame)
            except StopIteration:
                break
        if not frames:
            break

        chunk = len(frames) 
        print(f"Processing batch of {chunk} frames, total processed: {frame_count}/{total_frames} Frame shape {frames[0].shape}")
        max_track = total_frames - frame_count
        
        masks_l = process_frames(predictor, frames=[f[:, :half_w] for f in frames], prompt_text=prompt_text, max_frames_to_track=max_track, frame_idx=frame_count, warp=raft)
        masks_r = process_frames(predictor, frames=[f[:, half_w:] for f in frames], prompt_text=prompt_text, max_frames_to_track=max_track, frame_idx=frame_count, warp=raft)
            
        for i in range(chunk):
            if frames[i].max() <= 1.0:
                raw_frame = (frames[i] * 255).to(dtype=torch.uint8)
            else:
                raw_frame = frames[i].to(dtype=torch.uint8)

            if raw_frame.shape[-1] == 3:
                frames[i] = raw_frame[:, :, [2, 1, 0]]
            else:
                frames[i] = raw_frame

            p_frame = packer.pack_frame(frames[i], masks_l[i], masks_r[i])
            writer.stdin.write(p_frame.to(dtype=torch.uint8).cpu().contiguous().numpy().tobytes())
            mask_h, mask_w = masks_l[i].shape
            ml_4d = masks_l[i].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
            mr_4d = masks_r[i].unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)
            smooth_ml = torch.nn.functional.interpolate(ml_4d, size=(height, height), mode='bicubic', antialias=True).squeeze()
            smooth_mr = torch.nn.functional.interpolate(mr_4d, size=(height, height), mode='bicubic', antialias=True).squeeze()
            gray_sbs = torch.cat([smooth_ml, smooth_mr], dim=1).to(dtype=torch.uint8)
            white_sbs = torch.stack([gray_sbs, gray_sbs, gray_sbs], dim=-1)
            mask_writer.stdin.write(white_sbs.cpu().contiguous().numpy().tobytes())

        frame_count += chunk
        pbar.update(chunk)

    if writer is not None:
        writer.stdin.close()
        writer.wait()

    if mask_writer is not None:
        mask_writer.stdin.close()
        mask_writer.wait()
        
def process_directory(video_path1, video_path2, output_dir, **kwargs):
    os.makedirs(output_dir, exist_ok=True)
    import glob

    video_files = []
    for ext in ["*.mp4", "*.mov", "*.mkv", "*.avi"]: 
        video_files.extend(glob.glob(os.path.join(video_path1, ext)))
    
    if not video_files:
        print(f"No videos found in {video_path1}")
        return
        
    print(f"Found {len(video_files)} videos in {video_path1}")
        
    for i, v_path1 in enumerate(video_files):
        filename = os.path.basename(v_path1)
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{base_name}_ALPHA.mp4")
        mask_path = os.path.join(output_dir, f"{base_name}_mask.mp4")
        packed = os.path.join(output_dir, f"{base_name}_XALPHA.mp4")
        v_path2 = os.path.join(video_path2, filename) if video_path2 else None
        
        print(f"\n=======================================================")
        print(f"[{i+1}/{len(video_files)}] Processing: {filename}")
        print(f"=======================================================")

        if os.path.exists(out_path) and os.path.exists(mask_path):
            print(f"Skipping {filename}, outputs already exist.")
            continue
        if os.path.exists(out_path) and os.path.exists(packed):
            print(f"Skipping {filename}, outputs already exist.")
            continue      

        process_allthethings(
            video_path1=v_path1,
            video_path2=v_path2,
            out_path=out_path,
            mask_path=mask_path,
            **kwargs)

if __name__ == "__main__":

    INPUT_FOLDER = "assets/video_segments"
    OUTPUT_FOLDER = "assets/out_segments"
 
    video_path1=INPUT_FOLDER
    video_path2=None

    process_directory(
        video_path1=video_path1,
        video_path2=video_path2,
        output_dir=OUTPUT_FOLDER,        
        prompt_text="One girl",
        batch_size=100,
        matte_size=0.4,
        warp=False,
        debug=None,
    )
