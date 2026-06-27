import logging, os, glob, json
from tqdm import tqdm
from imagemask import *
from sam3.model_builder import build_sam3_predictor
from model_builder import build_sam3_video_predictor
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

def ffmpeg_pipe(out_path, width, height, fps, audio_source=None):

    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
        '-i', '-'
    ]
    
    if audio_source:
        ffmpeg_cmd.extend(['-i', audio_source, '-map', '0:v', '-map', '1:a?'])
        
    ffmpeg_cmd.extend([
        '-sws_flags', 'bicubic+full_chroma_int+accurate_rnd+full_chroma_inp', '-c:v', 'hevc_qsv', '-profile:v', 'main10', '-pix_fmt', 'p010le', '-tag:v', 'hvc1', '-g', '100', '-b:v', '100M', '-preset', 'medium', '-c:a', 'aac', '-b:a', '256k', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-fps_mode', 'cfr', '-r', str(fps), '-movflags', '+faststart+write_colr+use_metadata_tags', '-metadata:s:v:0', 'stereo_mode=left_right', '-color_trc', 'bt709', '-color_range', 'pc', out_path
    ])
    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

class AlphaPacker:
    def __init__(self, scale=0.40, padding=0, circle=False):

        self.scale = scale
        self.padding = padding
        self.circle = circle
        self._cache = None

    def _circle(self, w, h):
        if self._cache is not None and self._cache.shape == (h, w):
            return self._cache

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
        self._cache = cache
        return self._cache

    def pack_frame(self, frames, mask_l = None, mask_r = None):

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

        mask_l = l_small.astype(np.uint8) 
        mask_r = r_small.astype(np.uint8) 

        packed_frame = frames
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

        if self.circle:
            circle = self._circle(target_w, target_h) 
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
        packed_frame[y1_tr:y2_tr, x1_tr:x2_tr] = blend_white_mask(packed_frame[y1_tr:y2_tr, x1_tr:x2_tr], q_bl_mask, q_bl_circle)

        y1_tl_l = self.padding
        y2_tl_l = y1_tl_l + h_half
        x1_tl_l = self.padding
        x2_tl_l = self.padding + w_half
        packed_frame[y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l] = blend_white_mask(packed_frame[y1_tl_l:y2_tl_l, x1_tl_l:x2_tl_l], q_br_mask, q_br_circle)

        y1_br_r = H - self.padding - h_half
        y2_br_r = y1_br_r + h_half
        x1_br_r = SBS_W - self.padding - w_half
        x2_br_r = SBS_W - self.padding
        packed_frame[y1_br_r:y2_br_r, x1_br_r:x2_br_r] = blend_white_mask(packed_frame[y1_br_r:y2_br_r, x1_br_r:x2_br_r], q_tl_mask, q_tl_circle)

        y1_bl_l = H - self.padding - h_half
        y2_bl_l = y1_bl_l + h_half
        x1_bl_l = self.padding
        x2_bl_l = self.padding + w_half
        packed_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l] = blend_white_mask(packed_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l], q_tr_mask, q_tr_circle)
        return packed_frame

class AlphaPackerB:
    def __init__(self, scale=0.25, padding=0):

        self.scale = scale
        self.padding = padding
        self._cache = None

    def _circle(self, w, h):
        if self._cache is not None and self._cache.shape == (h, w):
            return self._cache

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
        self._cache = cache
        return self._cache

    def pack_frame(self, frames, mask_l = None, mask_r = None, sbs=False):

        H, SBS_W, C = frames.shape
        half_W = SBS_W // 2

        if mask_l.dtype != np.uint8:
            mask_l = (mask_l * 255).astype(np.uint8)
            mask_r = (mask_r * 255).astype(np.uint8)

        target_w = int(half_W * self.scale)
        target_h = int(H * self.scale)
      
        circle = self._circle(target_w, target_h)

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

        packed_frame = frames
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
        packed_frame[y1:y2, x1_center:x2_center] = blend_white_mask(packed_frame[y1:y2, x1_center:x2_center], mask_l_circle, inv_circle_3d)

        left_half_mask = mask_r_circle[:, :w_half]
        right_half_mask = mask_r_circle[:, w_half:]
        inv_circle_left = inv_circle_3d[:, :w_half]
        inv_circle_right = inv_circle_3d[:, w_half:]
        w_remain = target_w - w_half
        
        packed_frame[y1:y2, 0:w_remain] = blend_white_mask(
            packed_frame[y1:y2, 0:w_remain], right_half_mask, inv_circle_right)

        packed_frame[y1:y2, SBS_W - w_half:SBS_W] = blend_white_mask(
            packed_frame[y1:y2, SBS_W - w_half:SBS_W], left_half_mask, inv_circle_left)

        return packed_frame

def process_frames(video_frames, frames_pil, prompt, frame_index=0, object_id=1, start_frame_index=0, max_frames_to_track=-1, close_after_propagation=True,  keep_model_loaded=False, session_id=None, prior_mask=None, positive_coords=None, negative_coords=None, bbox=None, propagation_direction="forward", sam31=False, warp=False, prior_frame=None):
    
    raft = raft_flow(device="cuda") if warp else None

    predictor =  build_sam3_predictor(
    checkpoint_path = None,
    bpe_path = None,
    version= "sam3.1",
    compile= False,
    warm_up= False,
    max_num_objects = 2,
    multiplex_count = 16,   
    use_fa3 = False,
    use_rope_real = False,
    async_loading_frames = True,
    default_output_prob_thresh=0.55) if sam31 else None
  
    predictor = build_sam3_video_predictor(
    gpus_to_use=None,
    has_presence_token=False,
    geo_encoder_use_img_cross_attn=False,
    strict_state_dict_loading=False,
    async_loading_frames=True,
    video_loader_type="torchcodec",
    apply_temporal_disambiguation = True,
    compile = False)

    chunk = len(frames_pil)
    H, W, C = video_frames[0].shape 

    if frame_index > chunk - 1:
        logger.info(f"Frame index {frame_index} is out of bounds, setting to last frame {chunk - 1}")
        frame_index = chunk - 1

    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=frames_pil,
            session_id=session_id
        )
    )

    session_id = response.get("session_id", None)
    if session_id is None:
        raise ValueError("Failed to start video prediction session")

    if prior_mask is not None:
        predictor.handle_request(dict(
            type="add_new_mask",
            session_id=session_id,
            frame_index=frame_index,
            obj_id=object_id,
            mask=prior_mask,
            ))

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pos_points, pos_count, pos_errors = parse_points(positive_coords, video_frames[0].shape)
        neg_points, neg_count, neg_errors = parse_points(negative_coords, video_frames[0].shape)
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
            bbox_coords, bbox_count = parse_bbox(bbox, video_frames[0].shape)
            if bbox_coords is not None:
                bounding_boxes = bbox_coords
                bounding_box_labels = [1] * bbox_count

        response = predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=prompt if prompt else None,
                bounding_boxes=bounding_boxes,
                bounding_box_labels=bounding_box_labels,
                points=points,
                point_labels=point_labels,
                obj_id=object_id
            )
        )

        hard_masks = []
        output = np.zeros((chunk, H, W), dtype=np.uint8)
        processed_frames = 0
        object_outputs = {
            "obj_ids":None,
            "obj_masks":[]
        }
        objects = {}

        session = predictor._get_session(session_id)
        for response in predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
                propagation_direction=propagation_direction,
                start_frame_index=start_frame_index,
                max_frame_num_to_track=max_frames_to_track if max_frames_to_track != -1 else None,
            )
        ):
            frame_idx = response.get("frame_index", 0)
            outputs = response.get("outputs", {})
            obj_ids = outputs.get("out_obj_ids", None)
            
            if obj_ids is not None:
                object_outputs["obj_ids"] = obj_ids

            if warp:
                session = predictor._get_session(session_id)
                inference_state = session["state"]
                states = inference_state["tracker_inference_states"]

                state = states[0]
                tensors_rgb = []

                for f_bgr in video_frames:
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

                    if prior_frame is not None:
                        flow = get_flow_from_images(frames_pil, prior_frame, method="DIS Medium", prev_flow=prev_flow)
                        prev_flow = flow[-1] if flow is not None else None

                        predictor.model._prepare_backbone_feats(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            reverse=False)

                        _, _, h_mask, w_mask = prev_logits.shape
                        _, _, frame_H, frame_W = prev_tensor.unsqueeze(0).shape
                        flow = raft.compute_raft_flow(
                            prev_tensor.unsqueeze(0),
                            curr_tensor.unsqueeze(0),
                            max_size=max(frame_H, frame_W),
                            scale=1.0,
                            interp_mode="bicubic",
                            target_size=(h_mask, w_mask)).squeeze(0)

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

    if len(objects) > 0:
        max_num_objects = max(mask.shape[0] for mask in objects.values())

        ordered = []
        padded = []

        for frame_idx in range(C):
            if frame_idx in objects:
                mask = objects[frame_idx]  
                num_objects = mask.shape[0]
                
                if num_objects < max_num_objects:
                    padding = np.zeros((max_num_objects - num_objects, H, W), dtype=np.float32)
                    padded_mask = np.concatenate([mask, padding], axis=0)
                    ordered.append(padded_mask)
                    padded.append(torch.from_numpy(padded_mask))
                else:
                    ordered.append(mask)
                    padded.append(torch.from_numpy(mask))
            else:
                empty = np.zeros((max_num_objects, H, W), dtype=np.float32)
                ordered.append(empty)
                padded.append(torch.zeros((max_num_objects, H, W)))

        object_masks = torch.stack(padded, dim=0)
        object_outputs["obj_masks"] = ordered
    else:
        object_masks = torch.zeros((C, H, W))
        object_outputs["obj_masks"] = []
    return output, session_id, frame_idx

def process_videos(video_path1, video_path2, out_path, mask_path, prompt_text=None, batch_size=None, matte_size=None, warp=False, full_sbs=False, alpha_pack=False, left_right=False, debug=None, bbox=None, overlay=False):  

    frames_tot, keyframes, width, height, duration, fps = metadata(video_path1)
    half_w = width if left_right else width // 2
    if left_right or alpha_pack:
        frames_tot2, keyframes2, width2, height2, duration2, fps2 = metadata(video_path2)
        half_w2 = width2 // 2

    if left_right:
        left = video_frame_generator(video_path1, force_rate=0, frame_load_cap=debug or 0, skip_first_frames=0, select_every_nth=1, output_format="bgr24")
        _ = next(left)
        right = video_frame_generator(video_path2, force_rate=0, frame_load_cap=debug or 0, skip_first_frames=0, select_every_nth=1, output_format="bgr24")
        _ = next(right) 
    elif alpha_pack:
        sbs = video_frame_generator(video_path1, force_rate=0, frame_load_cap=debug or 0, skip_first_frames=0, select_every_nth=1, output_format="bgr24")
        _ = next(sbs)          
        matte = video_frame_generator(video_path2, force_rate=0, frame_load_cap=debug or 0, skip_first_frames=0, select_every_nth=1, output_format="bgr24")
        _ = next(matte)         
    else:
        sbs = video_frame_generator(video_path1, force_rate=0, frame_load_cap=debug or 0, skip_first_frames=0, select_every_nth=1, output_format="bgr24")
        _ = next(sbs) 

    out_width = width * 2 if left_right else width

    writer = ffmpeg_pipe(out_path, out_width, height, fps, audio_source=video_path1) if out_path else None
    mask_writer = ffmpeg_pipe(mask_path, out_width, height, fps) if mask_path else None
    packer = AlphaPacker(scale=matte_size, circle=False)

    frame_count = 0
    frames_tot = frames_tot if debug is None else debug
    batch_size = frames_tot if batch_size == 0 else batch_size
    pbar = tqdm(total=frames_tot, desc="Processing .. beep.boop.bop.. beep.")

    last_l = None
    last_r = None
    session_l = 0
    frame_idx_l = 0
    prev_flow = None
    prior_frame = None

    while frame_count < frames_tot:
        frames = []
        mattes = []
        l_frames = []
        r_frames = []
      
        for _ in range(keyframes if batch_size is None else batch_size):
            try:

                if left_right:
                    bgr_l = next(left)
                    l_frames.append(bgr_l)
                    bgr_r = next(right)
                    r_frames.append(bgr_r)
                    frames.append(np.concatenate((bgr_l, bgr_r), axis=1))
                
                elif alpha_pack:        
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

        if full_sbs:
            prior_mask = last_l if (last_l is not None and np.sum(last_l) > 0) else None
            pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in  frames]       
            masks, session_id, frame_idx = process_frames(frames, pil_frames, prompt_text, prior_mask=prior_mask, frame_index=frame_idx_l, prior_frame=prior_frame)
            
            last_l = masks[-1] if masks is not None else None
            session_l = session_id[-1] if session_id is not None else None
            frame_idx_l = frame_idx if frame_idx is not None else None
            prior_frame =  pil_frames[-1] if pil_frames is not None else None

            masks = process_mask(masks, sensitivity=1.0, mask_blur=2, mask_offset=-2, smooth=0, 
                            fill_holes=False, invert_output=False, dilation=0, feather_radius=4, smooth_edges=0, davinci=True)

            masks_r = [f[:, half_w:] for f in masks]
            masks_l = [f[:, :half_w] for f in masks]

        elif alpha_pack:
            masks_l = [cv2.cvtColor(f[:, :half_w2], cv2.COLOR_BGR2GRAY) for f in mattes]
            masks_r = [cv2.cvtColor(f[:, half_w2:], cv2.COLOR_BGR2GRAY) for f in mattes]

        else:
            frames_l = l_frames if left_right else [f[:, :half_w] for f in frames]
            prior_l = last_l if (last_l is not None and np.sum(last_l) > 0) else None
            pil_l = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_l]       
            masks_l, session_id, frame_idx_l = process_frames(frames_l, pil_l, prompt_text)
            masks_l = process_mask(masks_l, sensitivity=1.0, mask_blur=0, mask_offset=0, smooth=0, 
                            fill_holes=False, invert_output=False, dilation=5, feather_radius=0, smooth_edges=0, davinci=True)

            frames_r = r_frames if left_right else [f[:, half_w:] for f in frames]
            prior_r = last_r if (last_r is not None and np.sum(last_r) > 0) else None
            pil_r = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_r]
            masks_r, session_id, frame_idx_r = process_frames(frames_r, pil_r, prompt_text)
            masks_r = process_mask(masks_r, sensitivity=1.0, mask_blur=0, mask_offset=0, smooth=0, 
                            fill_holes=False, invert_output=False, dilation=5, feather_radius=0, smooth_edges=0, davinci=True)

            last_l = masks_l[-1] if masks_l is not None else None
            last_r = masks_r[-1] if masks_r is not None else None

        for i in range(chunk):
            if full_sbs:
                packed_frame = packer.pack_frame(frames[i], masks_l[i], masks_r[i])
                if writer: writer.stdin.write(packed_frame.astype(np.uint8).tobytes())        
                
                if mask_writer:
                    if overlay:
                        full_sbs_img = resize_mask(torch.from_numpy(masks[i]).unsqueeze(0).unsqueeze(0), (height, out_width)).squeeze().numpy().astype(np.uint8)
                        mask_3d = np.zeros_like(frames[i])
                        mask_3d[:, :, 1] = full_sbs_img  # Green channel
                        sbs_out = cv2.addWeighted(frames[i], 0.6, mask_3d, 0.6, 0)           
                        mask_writer.stdin.write(sbs_out.tobytes())
                    else:
                        full_mask_l = cv2.resize(masks_l[i], (half_w, height), interpolation=cv2.INTER_LINEAR)
                        full_mask_r = cv2.resize(masks_r[i], (half_w, height), interpolation=cv2.INTER_LINEAR)
                        gray_sbs = np.zeros((height, out_width), dtype=np.uint8)
                        gray_sbs[:, :half_w] = full_mask_l
                        gray_sbs[:, half_w:] = full_mask_r
                        mask_writer.stdin.write(gray_sbs.tobytes())
         
            elif alpha_pack:
                packed_frame = packer.pack_frame(frames[i], masks_l[i], masks_r[i])
                if writer: writer.stdin.write(packed_frame.astype(np.uint8).tobytes())
            
            else:
                packed_frame = packer.pack_frame(frames[i], masks_l[i], masks_r[i])
                if writer: writer.stdin.write(packed_frame.astype(np.uint8).tobytes())
                
                if mask_writer:
                    if overlay:
                        full_mask_l = resize_mask(torch.from_numpy(masks_l[i]).unsqueeze(0).unsqueeze(0), (height, half_w)).squeeze().numpy().astype(np.uint8)
                        full_mask_r = resize_mask(torch.from_numpy(masks_r[i]).unsqueeze(0).unsqueeze(0), (height, half_w)).squeeze().numpy().astype(np.uint8)
                        gray_sbs = np.zeros((height, out_width), dtype=np.uint8)
                        gray_sbs[:, :half_w] = full_mask_l
                        gray_sbs[:, half_w:] = full_mask_r
                        mask_3d = np.zeros_like(frames[i])
                        mask_3d[:, :, 1] = gray_sbs  
                        sbs_out = cv2.addWeighted(frames[i], 0.3, mask_3d, 0.6, 0)          
                        mask_writer.stdin.write(sbs_out.tobytes())
                    else:
                        full_mask_l = cv2.resize(masks_l[i], (half_w, height), interpolation=cv2.INTER_LINEAR)
                        full_mask_r = cv2.resize(masks_r[i], (half_w, height), interpolation=cv2.INTER_LINEAR)
                        gray_sbs = np.zeros((height, out_width), dtype=np.uint8)
                        gray_sbs[:, :half_w] = full_mask_l
                        gray_sbs[:, half_w:] = full_mask_r
                        mask_writer.stdin.write(gray_sbs.tobytes())

        frame_count += chunk
        pbar.update(chunk)
        
    if not left_right and 'sbs' in locals() and sbs: sbs.close()
    if left_right:
        if 'left' in locals() and left: left.close()
        if 'right' in locals() and right: right.close()
    if writer is not None:
        writer.stdin.close()
        writer.wait()
    if mask_writer is not None:
        mask_writer.stdin.close()
        mask_writer.wait()
        
def process_directory(video_path1, video_path2, output_dir, **kwargs):
    os.makedirs(output_dir, exist_ok=True)

    video_files = []
    for ext in ["*.mp4", "*.mkv", "*.mov", "*.avi"]:
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

        process_videos(
            video_path1=v_path1,
            video_path2=v_path2,
            out_path=out_path,
            mask_path=mask_path,
            **kwargs
        )

if __name__ == "__main__":
    INPUT_FOLDER = "assets/video_segments"
    INPUT_FOLDER2 = "assets/video_segments2"
    OUTPUT_FOLDER = "assets/out_segments"

    left_right=False 
    alpha_pack=True 

    if left_right or alpha_pack:
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
        batch_size=100,
        matte_size=0.4,
        warp=False,
        full_sbs=False,
        alpha_pack=alpha_pack,
        left_right=left_right,
        debug=None,
        bbox=None,

    )
