
from huggingface_hub import hf_hub_download
# import pkg_resources
from torch import set_default_dtype
from masksandthings import *
from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
set_default_dtype(dtype)
gpus_to_use = [torch.cuda.current_device()]

def build_sam3_video_predictor(*model_args, checkpoint_path=None, bpe_path=None, gpus_to_use=None, **model_kwargs):
    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )

def download_ckpt_from_hf(version="sam3"):

    if version == "sam3.1":
        repo_id = "facebook/sam3.1"
        ckpt_name = "sam3.1_multiplex.pt"
        cfg_name = "config.json"

    elif version == "sam3.1_fp16":
        repo_id = "strangervisionhf/sam3.1-st-bf16"
        ckpt_name = "sam3.1_multiplex.pt"
        cfg_name = "config.json"

    elif version == "sam3_fp16":
        repo_id = "mlx-community/sam3-bf16"
        ckpt_name = "model.safetensors"
        cfg_name = "config.json"
    else:
        repo_id = "facebook/sam3"
        ckpt_name = "sam3.pt"
        cfg_name = "config.json"

            #if you dont have the model force_download=True local_files_only=False and enter a token for hf.. the fp16s are fine. sam3 is good. they kind of messed 3.1 up.
    _ = hf_hub_download(repo_id=repo_id, filename=cfg_name, force_download=False, local_files_only=False, token="")
    checkpoint_path = hf_hub_download(repo_id=repo_id, filename=ckpt_name, force_download=False, local_files_only=False, token="")
    return checkpoint_path

def ffmpeg_pipe(out_path, width, height, fps, audio_source=None):

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-'
    ]
    
    if audio_source:
        ffmpeg_cmd.extend(['-i', audio_source, '-map', '0:v', '-map', '1:a?'])
        
    ffmpeg_cmd.extend([
        '-sws_flags', 'lanczos+full_chroma_int+accurate_rnd+full_chroma_inp', '-c:v', 'hevc_qsv', '-profile:v', 'main10', '-pix_fmt', 'p010le', '-tag:v', 'hvc1', '-g', '100', '-b:v', '100M', '-preset', 'medium', '-aspect', '2:1', '-copyts', '-start_at_zero', '-bitexact', '-c:a', 'aac', '-b:a', '256k', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-fps_mode', 'cfr', '-r', str(fps), '-movflags', '+faststart+write_colr+use_metadata_tags', '-metadata:s:v:0', 'stereo_mode=left_right', '-color_trc', 'bt709', out_path
    ])
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

class AlphaPacker:
    def __init__(n, scale=0.40, padding=0, circle=False):

        n.scale = scale
        n.padding = padding
        n.circle = circle
        n._cache = None

    def _circle(n, w, h):
        if n._cache is not None and n._cache.shape == (h, w):
            return n._cache

        y = np.arange(h, dtype=np.float32)
        x = np.arange(w, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(y, x, indexing='ij')

        cy, cx = h / 2.0 - 0.5, w / 2.0 - 0.5
        r = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)

        max_r = min(w, h) / 2.0
        outer_r = max_r * 0.55  
        inner_r = max_r * 0.45  
        
        t = np.clip((outer_r - r) / (outer_r - inner_r), 0.0, 1.0) 
        cache = t * t * (3.0 - 2.0 * t)
        n._cache = cache
        return n._cache

    def pack_frame(n, frames, mask_l = None, mask_r = None):

        H, SBS_W, C = frames.shape
        half_W = SBS_W // 2

        if mask_l.dtype != np.uint8:
            mask_l = (mask_l * 255).astype(np.uint8)
            mask_r = (mask_r * 255).astype(np.uint8)

        target_w = int(half_W * n.scale)
        target_h = int(H * n.scale)
      
        if mask_l.shape[:2] != (target_h, target_w):
            l_small = cv2.resize(mask_l, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            l_small = mask_l

        if mask_r.shape[:2] != (target_h, target_w):
            r_small = cv2.resize(mask_r, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            r_small = mask_r

        mask_l = l_small.astype(np.uint8) 
        mask_r = r_small.astype(np.uint8) 

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

        if n.circle:
            circle = n._circle(target_w, target_h) 
            inv_circle_3d = (1.0 - circle)[..., np.newaxis].astype(np.float32)
            q_tl_circle = inv_circle_3d[:h_half, :w_half]
            q_tr_circle = inv_circle_3d[:h_half, w_half:w_half*2]
            q_bl_circle = inv_circle_3d[h_half:h_half*2, :w_half]
            q_br_circle = inv_circle_3d[h_half:h_half*2, w_half:w_half*2]
               
        def blend_white_mask(roi, mask_1ch, inv_circle_slice=None):
            if inv_circle_slice is None:
                inv_mask_3d = (255 - mask_1ch)[..., np.newaxis]
                blended = (roi.astype(np.uint16) * inv_mask_3d) // 255
                blended += mask_1ch[..., np.newaxis]
                x = blended.astype(np.uint8)
            else:
                blended = roi.astype(np.float32) * inv_circle_slice
                blended += mask_1ch[..., np.newaxis]
                x = np.clip(blended, 0, 255).astype(np.uint8)
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

class otherAlphaPacker:
    def __init__(n, scale=0.25, padding=0):

        n.scale = scale
        n.padding = padding
        n._cache = None

    def _circle(n, w, h):
        if n._cache is not None and n._cache.shape == (h, w):
            return n._cache

        y = np.arange(h, dtype=np.float32)
        x = np.arange(w, dtype=np.float32)
        grid_y, grid_x = np.meshgrid(y, x, indexing='ij')

        cy, cx = h / 2.0 - 0.5, w / 2.0 - 0.5
        r = np.sqrt((grid_x - cx)**2 + (grid_y - cy)**2)

        max_r = min(w, h) / 2.0
        outer_r = max_r * 0.55  
        inner_r = max_r * 0.45  
        
        t = np.clip((outer_r - r) / (outer_r - inner_r), 0.0, 1.0) 
        cache = t * t * (3.0 - 2.0 * t)
        n._cache = cache
        return n._cache

    def pack_frame(n, frames, mask_l = None, mask_r = None, sbs=False):

        H, SBS_W, C = frames.shape
        half_W = SBS_W // 2

        if mask_l.dtype != np.uint8:
            mask_l = (mask_l * 255).astype(np.uint8)
            mask_r = (mask_r * 255).astype(np.uint8)

        target_w = int(half_W * n.scale)
        target_h = int(H * n.scale)
      
        circle = n._circle(target_w, target_h)

        if mask_l.shape[:2] != (target_h, target_w):
            l_small = cv2.resize(mask_l, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            l_small = mask_l

        if mask_r.shape[:2] != (target_h, target_w):
            r_small = cv2.resize(mask_r, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            r_small = mask_r

        mask_l_circle = l_small
        mask_r_circle = r_small

        p_frame = frames
        h_half = target_h // 2
        w_half = target_w // 2

        inv_circle_3d = (1.0 - circle)[..., np.newaxis].astype(np.float32)

        def blend_white_mask(roi, mask_1ch, inv_circle_slice):
            blended = roi.astype(np.float32) * inv_circle_slice
            blended += mask_1ch[..., np.newaxis]
            return np.clip(blended, 0, 255).astype(np.uint8)

        sh, sw = r_small.shape[:2]
     
        y1 = ((H - target_h) // 2) - sh // 8 
        y2 = y1 + target_h
        
        x1_center = half_W - w_half
        x2_center = x1_center + target_w
        p_frame[y1:y2, x1_center:x2_center] = blend_white_mask(p_frame[y1:y2, x1_center:x2_center], mask_l_circle, inv_circle_3d)

        left_half_mask = mask_r_circle[:, :w_half]
        right_half_mask = mask_r_circle[:, w_half:]
        inv_circle_left = inv_circle_3d[:, :w_half]
        inv_circle_right = inv_circle_3d[:, w_half:]
        w_remain = target_w - w_half
        
        p_frame[y1:y2, 0:w_remain] = blend_white_mask(
            p_frame[y1:y2, 0:w_remain], right_half_mask, inv_circle_right)

        p_frame[y1:y2, SBS_W - w_half:SBS_W] = blend_white_mask(
            p_frame[y1:y2, SBS_W - w_half:SBS_W], left_half_mask, inv_circle_left)

        return p_frame

def process_frames(predictor, frames, frames_pil=None, prompt_text=None, frame_idx=0, object_id=1, start_frame_idx=0, max_frames_to_track=-1, close_after_propagation=True, keep_model_loaded=True, session_id=None, prev_mask=None, positive_coords=None, negative_coords=None, bbox=None, propagation_direction="forward", sam31=False, warp=None, prev_frame=None, matte_size=None, prev_flow=None, max_size=1008):

    frames = [cv2.resize(f, (max_size, max_size), interpolation=cv2.INTER_LANCZOS4) for f in frames] if max_size is not None else frames
    frames_pil = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames] 
    H, W, C = int(frames[0].shape[0]), int(frames[0].shape[1]), int(frames[0].shape[2])
    chunk = len(frames_pil)

    if frame_idx > chunk - 1:
        frame_idx = 0

    print(f"Processing frames of size: {W}x{H} ")
    print(f"frame_idx: {frame_idx} batch: {chunk}")

    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=frames_pil,
            session_id=session_id,
            start_frame_idx=start_frame_idx,
            offload_video_to_cpu = True,
            offload_state_to_cpu = False))

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
        output = np.zeros((chunk, H, W), dtype=np.uint8)
        processed_frames = 0
        object_outputs = {
            "obj_ids":None,
            "obj_masks":[]
        }
        objects = {}

        session = predictor._get_session(session_id)
        inference_state = session["state"]
        num_frames = inference_state["num_frames"]        

        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=propagation_direction,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=num_frames#max_frames_to_track if max_frames_to_track != -1 else None,
            )
        ):
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

                for f_bgr in frames:
                    f_rgb = cv2.cvtColor(f_bgr, cv2.COLOR_BGR2RGB)
                    t_rgb = torch.from_numpy(f_rgb).permute(2, 0, 1).float().div(255.0).to(device)
                    tensors_rgb.append(t_rgb)

                prev_logits = state["output_dict"]["cond_frame_outputs"][0]["pred_masks"].to(device).float()
                batch_size = len(state["obj_ids"])
                predictor.model.tracker.propagate_in_video_preflight(state, run_mem_encoder=True)
            
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for frame_idx in range(1, chunk):
                        prev_tensor = tensors_rgb[frame_idx - 1]
                        curr_tensor = tensors_rgb[frame_idx]
                        # print(f'Frame index 3: {frame_idx}')

                    # if prev_frame is not None:
                    #     flow = get_flow_from_images(frames_pil, prev_frame, method="DIS Medium", prev_flow=prev_flow)
                    #     prev_flow = flow[-1] if flow is not None else None

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
                        merged_prob = torch.max(prob, dim=0).values.cpu().numpy()
                        merged = (merged_prob * 255).astype(np.uint8)
                        
                        objects[frame_idx] = (logits > 0).astype(np.float32) if isinstance(logits, np.ndarray) else (logits > 0).cpu().numpy().astype(np.float32)
                        output[frame_idx] = merged
                        hard_masks.append(output)
                    else:
                        objects[frame_idx] = np.zeros((1, H, W), dtype=np.float32)
                
                elif "out_binary_masks" in outputs:
                    mask = outputs["out_binary_masks"]
                    
                    if mask.shape[0] > 0:
                        objects[frame_idx] = mask
                        merged = (np.any(mask, axis=0) * 255).astype(np.uint8)
                        output[frame_idx] = merged
                        hard_masks.append(output)
                    else:
                        objects[frame_idx] = np.zeros((1, H, W), dtype=np.float32)
                else:
                    objects[frame_idx] = np.zeros((1, H, W), dtype=np.float32)
                    
            if len(objects) > 0:
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
                    session_id=session_id,
                )
            )

        if not keep_model_loaded and close_after_propagation:
            predictor.shutdown()

    return output

def process_allthethings(video_path1, video_path2, out_path, mask_path, prompt_text=None, batch_size=None, matte_size=None, warp=False, full_sbs=False, alpha_pack=False, left_right=False, debug=None, bbox=None, overlay=False, track_prev=False, sam31=False):  

    raft = raft_flow(device="cuda") if warp else None

    predictor = build_sam3_video_predictor(
        checkpoint_path = download_ckpt_from_hf(version="sam3_fp16"),
        gpus_to_use = None,
        has_presence_token = False,
        geo_encoder_use_img_cross_attn = False,
        strict_state_dict_loading = False,
        async_loading_frames = True,
        video_loader_type = "cv2",
        apply_temporal_disambiguation = True,
        compile = False
    ) if not sam31 else None
    
    predictor = build_sam3_multiplex_video_predictor(
        checkpoint_path = download_ckpt_from_hf(version="sam3.1"),
        max_num_objects = 1,
        multiplex_count = 1,
        use_fa3 = False,
        use_rope_real = False,
        compile = False,
        warm_up = False,
        session_expiration_sec = 5000,
        default_output_prob_thresh = 0.5,
        async_loading_frames = True,
    ) if sam31 else predictor           
     
    frames_tot, keyframes, width, height, duration, fps = metadata(video_path1)
    half_w = width // 2

    if alpha_pack:
        frames_tot2, keyframes2, width2, height2, duration2, fps2 = metadata(video_path2)
        half_w2 = width2 // 2
        #i wrote a better function will remove this hacky-ness
        sbs = video_frame_generator(video_path1, force_rate=0, frame_load_cap=debug or 0, skip_first_frames=0, select_every_nth=1, output_format="bgr24")
        _ = next(sbs)         
        matte = video_frame_generator(video_path2, force_rate=0, frame_load_cap=debug or 0, skip_first_frames=0, select_every_nth=1, output_format="bgr24")
        _ = next(matte)         
    else:
        sbs = video_frame_generator(video_path1, force_rate=0, frame_load_cap=debug or 0, skip_first_frames=0, select_every_nth=1, output_format="bgr24")
        _ = next(sbs) 

        # with VideoFrameGenerator(video_path1, force_rate=0, frame_load_cap=debug or 0, skip_first_frames=0, select_every_nth=1, output_format="bgr24") as sbs:
        #     width, height, fps = sbs.width, sbs.height, sbs.fps
        #     for frame in sbs:
        #         frames.append(frame)

    writer = ffmpeg_pipe(out_path, width, height, fps, audio_source=video_path1) if out_path else None
    mask_writer = ffmpeg_pipe(mask_path, width, height, fps) if mask_path else None
    packer = AlphaPacker(scale=matte_size, circle=False)

    frame_count = 0
    frames_tot = frames_tot if debug is None else debug
    batch_size = frames_tot if batch_size == 0 else batch_size
    pbar = tqdm(total=frames_tot, desc="Processing .. beep.boop.bop.. beep.")

    while frame_count < frames_tot:
        frames = []
        mattes = []
      
        for _ in range(keyframes if batch_size is None else batch_size):
            try:
                if alpha_pack:        
                    frame_bgr = next(sbs)
                    frames.append(frame_bgr)        
                    matte_bgr = next(matte)
                    mattes.append(matte_bgr)                      
                else:
                    frame_bgr = next(sbs)
                    frames.append(frame_bgr)

            except StopIteration:
                break

        if not frames:
            break
        
        chunk = len(frames) 
        print(f"Processing chunk of {chunk} frames, total processed: {frame_count}/{frames_tot}")
        max_track = frames_tot - frame_count
        
        if full_sbs: # sam3 seems to be able to segment side by side as if it were a single image. You text prompt "one girl" and it will segment both left and right as if it see "one girl". Nothing special about the attention or rope..
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in  frames]     
            masks = process_frames(predictor, frames, frames_pil=pil_frames, prompt_text=prompt_text, max_frames_to_track=max_track, frame_idx=frame_count, warp=raft)
            masks_r = [f[:, half_w:] for f in masks]
            masks_l = [f[:, :half_w] for f in masks]
            masks_l = process_mask(masks_l, sensitivity=1.0, mask_blur=0, mask_offset=-2, fill_holes=False, invert_output=False, dilation=0, feather_radius=2.0, smooth_edges=1, davinci=True)
            masks_r = process_mask(masks_r, sensitivity=1.0, mask_blur=0, mask_offset=-2, fill_holes=False, invert_output=False, dilation=0, feather_radius=2.0, smooth_edges=1, davinci=True)

        elif alpha_pack:
            masks_l = [cv2.cvtColor(f[:, :half_w2], cv2.COLOR_BGR2GRAY) for f in mattes]
            masks_r = [cv2.cvtColor(f[:, half_w2:], cv2.COLOR_BGR2GRAY) for f in mattes]
            masks_l = process_mask(masks_l, sensitivity=1.0, mask_blur=0, mask_offset=-2, fill_holes=False, invert_output=False, dilation=0, feather_radius=2.0, smooth_edges=1, davinci=True)
            masks_r = process_mask(masks_r, sensitivity=1.0, mask_blur=0, mask_offset=-2, fill_holes=False, invert_output=False, dilation=0, feather_radius=2.0, smooth_edges=1, davinci=True)

        else:
            masks_l = process_frames(predictor, frames=[f[:, :half_w] for f in frames], prompt_text=prompt_text, max_frames_to_track=max_track, frame_idx=frame_count, warp=raft)
            masks_r = process_frames(predictor, frames=[f[:, half_w:] for f in frames], prompt_text=prompt_text, max_frames_to_track=max_track, frame_idx=frame_count, warp=raft)
            masks_l = process_mask(masks_l, sensitivity=1.0, mask_blur=0, mask_offset=-1, fill_holes=False, invert_output=False, dilation=2, feather_radius=2.0, smooth_edges=3, davinci=True)
            masks_r = process_mask(masks_r, sensitivity=1.0, mask_blur=0, mask_offset=-1, fill_holes=False, invert_output=False, dilation=2, feather_radius=2.0, smooth_edges=3, davinci=True)
            
        for i in range(chunk):
            
            if alpha_pack:
                p_frame = packer.pack_frame(frames[i], masks_l[i], masks_r[i])
                writer.stdin.write(p_frame.astype(np.uint8).tobytes())

            elif full_sbs:
                p_frame = packer.pack_frame(frames[i], masks_l[i], masks_r[i])
                writer.stdin.write(p_frame.astype(np.uint8).tobytes())         
                sbs_img = cv2.resize(masks[i], (width, height), interpolation=cv2.INTER_CUBIC)

                if overlay:
                    mask_3d = np.zeros_like(frames[i])
                    mask_3d[:, :, 1] = sbs_img  
                    sbs_out = cv2.addWeighted(frames[i], 0.6, mask_3d, 0.6, 0)           
                    mask_writer.stdin.write(sbs_out.tobytes())        
                else:
                    sbs_out = np.stack([sbs_img, sbs_img, sbs_img], axis=-1)
                    mask_writer.stdin.write(sbs_out.tobytes())

            else:  # standard:
                p_frame = packer.pack_frame(frames[i], masks_l[i], masks_r[i])
                writer.stdin.write(p_frame.astype(np.uint8).tobytes())
                gray_sbs = np.zeros((height, width), dtype=np.uint8) #w:h = 2:1
                gray_sbs[:, :height] = cv2.resize(masks_l[i], (height, height), interpolation=cv2.INTER_CUBIC)
                gray_sbs[:, height:] = cv2.resize(masks_r[i], (height, height), interpolation=cv2.INTER_CUBIC)

                if overlay:
                    mask_3d = np.zeros_like(frames[i])
                    mask_3d[:, :, 1] = gray_sbs  
                    sbs_out = cv2.addWeighted(frames[i], 0.3, mask_3d, 0.7, 0)
                    mask_writer.stdin.write(sbs_out.tobytes())      
                else:
                    white_sbs = np.stack([gray_sbs, gray_sbs, gray_sbs], axis=-1)
                    mask_writer.stdin.write(white_sbs.tobytes())

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
    INPUT_FOLDER2 = "assets/video_segments2"
    OUTPUT_FOLDER = "assets/out_segments"

    left_right=False 
    alpha_pack=False

    if alpha_pack:
        video_path1=INPUT_FOLDER
        video_path2=INPUT_FOLDER2
    else:
        video_path1=INPUT_FOLDER
        video_path2=None

    process_directory(
        video_path1=video_path1,
        video_path2=video_path2,
        output_dir=OUTPUT_FOLDER,        
        prompt_text="One girl",
        batch_size=5,
        matte_size=0.4,
        warp=False,
        full_sbs=False,
        alpha_pack=alpha_pack,
        left_right=left_right,
        debug=5,
        bbox=None,
    )
