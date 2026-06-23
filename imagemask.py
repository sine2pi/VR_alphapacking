import torch, math, cv2, av, base64, subprocess, functools, os, re, logger, time, threading, numpy as np, torchvision
from io import BytesIO
from PIL import Image, ImageSequence, ImageOps, Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

def ensure_mask_type(func):
    @functools.wraps(func)
    def wrapper(mask, *args, **kwargs):
        if isinstance(mask, list):
            return [wrapper(m, *args, **kwargs) for m in mask]
            
        is_pil = isinstance(mask, Image.Image)
        is_numpy = isinstance(mask, np.ndarray)
        
        if is_pil:
            t = torch.from_numpy(np.array(mask.convert('L'))).float() / 255.0
        elif is_numpy:
            t = torch.from_numpy(mask).float()
            if mask.dtype == np.uint8 or mask.max() > 1.0: t /= 255.0
        else:
            t = mask.clone().float()
            if t.dtype == torch.uint8 or t.max() > 1.0: t /= 255.0
            
        out_t = func(t, *args, **kwargs)
        
        if is_pil:
            return Image.fromarray((out_t.cpu().numpy() * 255).astype(np.uint8))
        elif is_numpy:
            out_np = out_t.cpu().numpy()
            return (out_np * 255).astype(np.uint8) if mask.dtype == np.uint8 else out_np
        return out_t.to(mask.dtype)
    return wrapper

@ensure_mask_type
def fill_mask_region(mask: torch.Tensor) -> torch.Tensor:
    m_np = (mask.cpu().numpy() * 255).astype(np.uint8)
    orig_shape = m_np.shape
    if m_np.ndim == 2: m_np = m_np[None, ...]
    elif m_np.ndim == 4: m_np = m_np.squeeze(1)
    
    filled_batch = []
    for i in range(m_np.shape[0]):
        contours, _ = cv2.findContours(m_np[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled_mask = np.zeros_like(m_np[i])
        for contour in contours:
            cv2.drawContours(filled_mask, [contour], 0, 255, -1) 
        filled_batch.append(filled_mask)
        
    filled_np = np.stack(filled_batch, axis=0).reshape(orig_shape)
    return torch.from_numpy(filled_np).to(mask.device).float() / 255.0

@ensure_mask_type
def feather_mask(mask: torch.Tensor, blur_radius: float = 1.5, iterations: int = 3) -> torch.Tensor:
    if blur_radius <= 0: return mask
    orig_shape = mask.shape
    if mask.ndim == 2: x = mask[None, None, ...]
    elif mask.ndim == 3: x = mask[None, ...]
    else: x = mask
    x = x.float()
    k_size = int(blur_radius * 2) + 1
    if k_size % 2 == 0: k_size += 1
    for _ in range(iterations):
        x = torchvision.transforms.functional.gaussian_blur(x, kernel_size=k_size, sigma=float(blur_radius))
    return x.view(orig_shape) 

@ensure_mask_type
def morph3x3(mask: torch.Tensor, dilation: int) -> torch.Tensor:
    if dilation == 0: return mask
    orig_shape = mask.shape
    x = mask.float()
    if x.ndim == 2: x = x[None, None, ...]
    elif x.ndim == 3: x = x[None, ...]
    k_size = 2 * abs(dilation) + 1
    padding = abs(dilation)
    if dilation > 0:
        x = torch.nn.functional.max_pool2d(x, kernel_size=k_size, stride=1, padding=padding)
    else:
        x = -torch.nn.functional.max_pool2d(-x, kernel_size=k_size, stride=1, padding=padding)
    return (x > 0.5).view(orig_shape).float()

@ensure_mask_type
def mask_edges(mask: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
    if kernel_size <= 0: return mask
    orig_shape = mask.shape
    x = mask.float()
    if x.ndim == 2: x = x[None, None, ...]
    elif x.ndim == 3: x = x[None, ...]
    x = (x > 0.5).float()
    pad = kernel_size // 2
    k_size = pad * 2 + 1
    if pad > 0:
        x = torch.nn.functional.max_pool2d(x, kernel_size=k_size, stride=1, padding=pad)
        x = -torch.nn.functional.max_pool2d(-x, kernel_size=k_size, stride=1, padding=pad)
        x = -torch.nn.functional.max_pool2d(-x, kernel_size=k_size, stride=1, padding=pad)
        x = torch.nn.functional.max_pool2d(x, kernel_size=k_size, stride=1, padding=pad)
    blur_k = k_size + 2
    if blur_k % 2 == 0: blur_k += 1
    blur_radius = float(blur_k) / 3.0
    x = torchvision.transforms.functional.gaussian_blur(x, kernel_size=blur_k, sigma=float(blur_radius))
    return (x > 0.5).view(orig_shape).float()

@ensure_mask_type
def process_mask(mask: torch.Tensor, sensitivity=1.0, mask_blur=0, mask_offset=0, smooth=0.0, 
                 fill_holes=False, invert_output=False, 
                 dilation=0, feather_radius=0.0, smooth_edges=0):
    
    actual_offset = mask_offset if mask_offset != 0 else dilation
    actual_blur = mask_blur if mask_blur > 0 else feather_radius
    
    m = mask
    
    if sensitivity != 1.0:
        m = torch.clamp(m * (1 + (1 - sensitivity)), 0, 1)
        
    if smooth > 0:
        m_binary = (m > 0.5).float()
        k_size = int(smooth * 3)
        if k_size % 2 == 0: k_size += 1
        if k_size < 3: k_size = 3
        orig_shape = m_binary.shape
        if m_binary.ndim == 2: mb_view = m_binary[None, None, ...]
        elif m_binary.ndim == 3: mb_view = m_binary[None, ...]
        else: mb_view = m_binary
        blurred = torchvision.transforms.functional.gaussian_blur(mb_view, kernel_size=int(k_size), sigma=float(smooth))
        m = (blurred > 0.5).float().view(orig_shape)
        
    if fill_holes:
        m = fill_mask_region.__wrapped__(m)
        
    if actual_offset != 0:
        m = morph3x3.__wrapped__(m, actual_offset)
        
    if smooth_edges > 0:
        m = mask_edges.__wrapped__(m, kernel_size=smooth_edges)
        
    if actual_blur > 0:
        m = feather_mask.__wrapped__(m, blur_radius=actual_blur, iterations=1)
        
    if invert_output:
        m = 1.0 - m
        
    return m

apply_effects = process_mask

def tensor2pil(image):

    if isinstance(image, Image.Image):
        return image
        
    if isinstance(image, list):
        return [tensor2pil(img) for img in image]
        
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        img_np = image
    else:
        raise ValueError(f"Unsupported type: {type(image)}")
        
    dim = img_np.ndim
    is_batched = False
    
    if dim == 2:
        img_np = img_np[np.newaxis, ..., np.newaxis]
    elif dim == 3:
        if img_np.shape[0] in [1, 3, 4] and img_np.shape[2] not in [1, 3, 4]:
            img_np = np.transpose(img_np, (1, 2, 0))
        elif img_np.shape[0] in [1, 3, 4] and img_np.shape[2] in [1, 3, 4]:
             if img_np.shape[0] < img_np.shape[1]:
                img_np = np.transpose(img_np, (1, 2, 0))
        img_np = img_np[np.newaxis, ...]
    elif dim == 4:
        is_batched = True
        if img_np.shape[1] in [1, 3, 4] and img_np.shape[3] not in [1, 3, 4]:
            img_np = np.transpose(img_np, (0, 2, 3, 1))
    else:
        raise ValueError(f"Unsupported number of dimensions: {dim}")

    if img_np.max() <= 1.0 and img_np.dtype != np.uint8:
        img_np = img_np * 255.0
        
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    pil_images = []
    for img in img_np:
        channels = img.shape[-1]
        if channels == 1:
            pil_img = Image.fromarray(img.squeeze(-1), mode='L')
        elif channels == 3:
            pil_img = Image.fromarray(img, mode='RGB')
        elif channels == 4:
            pil_img = Image.fromarray(img, mode='RGBA')
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
        pil_images.append(pil_img)
        
    if is_batched:
        return pil_images
    else:
        return pil_images[0]

def pil2tensor(image):
    if image is None:
        return None
        
    if isinstance(image, list):
        tensors = [pil2tensor(img) for img in image if img is not None]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    output_images = []
    
    for frame in ImageSequence.Iterator(image):
        frame = ImageOps.exif_transpose(frame)
        
        if frame.mode == 'I':
            frame = frame.point(lambda i: i * (1 / 255)).convert('L')
        if frame.mode not in ['RGB', 'RGBA', 'L']:
            frame = frame.convert('RGB')
            
        img_np = np.array(frame).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)
        
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)
        else:
            img_tensor = img_tensor.permute(2, 0, 1)
            
        img_tensor = img_tensor.unsqueeze(0)
        
        output_images.append(img_tensor)

    if not output_images:
        return None
        
    return torch.cat(output_images, dim=0)

image_to_tensor = pil2tensor

def resize_image(img: Image.Image, width: int, height: int) -> Image.Image:
    return img.resize((width, height), resample=Image.LANCZOS)

def get_target_dimensions(orig_width, orig_height, custom_width=0, custom_height=0, megapixels=0.0, scale_by=1.0, size=0, resize_mode="longest_side", downscale_ratio=0):
    target_w, target_h = orig_width, orig_height

    if megapixels > 0:
        aspect_ratio = orig_width / orig_height
        target_pixels = int(megapixels * 1024 * 1024)
        target_h = int((target_pixels / aspect_ratio) ** 0.5)
        target_w = int(aspect_ratio * target_h)
    elif custom_width > 0 or custom_height > 0:
        if custom_width > 0 and custom_height == 0:
            target_h = int(orig_height * (custom_width / orig_width))
            target_w = custom_width
        elif custom_height > 0 and custom_width == 0:
            target_w = int(orig_width * (custom_height / orig_height))
            target_h = custom_height
        else:
            target_w = custom_width
            target_h = custom_height
    elif size > 0:
        if resize_mode == "longest_side":
            if orig_width >= orig_height:
                target_w = size
                target_h = int(orig_height * (size / orig_width))
            else:
                target_h = size
                target_w = int(orig_width * (size / orig_height))
        elif resize_mode == "shortest_side":
            if orig_width <= orig_height:
                target_w = size
                target_h = int(orig_height * (size / orig_width))
            else:
                target_h = size
                target_w = int(orig_width * (size / orig_height))
        elif resize_mode == "width":
            target_w = size
            target_h = int(orig_height * (size / orig_width))
        elif resize_mode == "height":
            target_h = size
            target_w = int(orig_width * (size / orig_height))

    if scale_by != 1.0:
        target_w = int(target_w * scale_by)
        target_h = int(target_h * scale_by)

    if downscale_ratio > 0:
        target_w = int(target_w / downscale_ratio + 0.5) * downscale_ratio
        target_h = int(target_h / downscale_ratio + 0.5) * downscale_ratio

    return int(target_w), int(target_h)

def target_shape(img_shape, target_size: int):
    h, w = img_shape[:2]
    new_w, new_h = get_target_dimensions(w, h, size=target_size, resize_mode="longest_side")
    return new_h, new_w

def _resize_image(image, megapixels=0.0, scale_by=1.0, size=0, resize_mode="longest_side", resampling=Image.LANCZOS):
    orig_w, orig_h = image.size
    target_w, target_h = get_target_dimensions(orig_w, orig_h, megapixels=megapixels, scale_by=scale_by, size=size, resize_mode=resize_mode)
    if target_w != orig_w or target_h != orig_h:
        image = image.resize((target_w, target_h), resampling)
    return image, target_w, target_h

def target_dimensions(orig_width, orig_height, megapixels=0.0, scale_by=1.0, size=0, resize_mode="longest_side"):
    return get_target_dimensions(orig_width, orig_height, megapixels=megapixels, scale_by=scale_by, size=size, resize_mode=resize_mode)

def resize_image(img, mask_channel="alpha", resampling=Image.LANCZOS, megapixels=0.0, scale_by=1.0, size=0, resize_mode="longest_side", advanced_mask=False):

    resized_img, width, height = _resize_image(img, megapixels=megapixels, scale_by=scale_by, size=size, resize_mode=resize_mode, resampling=resampling)
    img_rgb = resized_img.convert("RGB")
    mask = None
    if advanced_mask:
        if mask_channel == "alpha" and "A" in resized_img.getbands():
            mask = np.array(resized_img.getchannel("A")).astype(np.float32) / 255.0
        elif mask_channel == "red":
            mask = np.array(img_rgb.getchannel("R")).astype(np.float32) / 255.0
        elif mask_channel == "green":
            mask = np.array(img_rgb.getchannel("G")).astype(np.float32) / 255.0
        elif mask_channel == "blue":
            mask = np.array(img_rgb.getchannel("B")).astype(np.float32) / 255.0
    else:
        if "A" in resized_img.getbands():
            mask = np.array(resized_img.getchannel("A")).astype(np.float32) / 255.0

    if mask is None:
        mask = np.ones((height, width), dtype=np.float32)

    image_tensor = image_to_tensor(img_rgb)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0)

    if advanced_mask:
        mask_image = mask_tensor.reshape((-1, 1, height, width)).movedim(1, -1).expand(-1, -1, -1, 3)
    else:
        mask_image = None

    return image_tensor, mask_tensor, mask_image, width, height

def target_size(width, height, custom_width, custom_height, downscale_ratio=8) -> tuple[int, int]:
    if downscale_ratio is None:
        downscale_ratio = 8
    return get_target_dimensions(width, height, custom_width=custom_width, custom_height=custom_height, downscale_ratio=downscale_ratio)

def pil2mask(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image.convert("L")).astype(np.float32) / 255.0).unsqueeze(0)

def combine_masks(mask_1, mode="combine", mask_2=None, mask_3=None, mask_4=None):
    masks = [m for m in [mask_1, mask_2, mask_3, mask_4] if m is not None]
    if len(masks) <= 1:
        return (masks[0] if masks else torch.zeros((1, 64, 64), dtype=torch.float32),)
        
    ref_shape = masks[0].shape
    masks = [_resize_if_needed(m, ref_shape) for m in masks]
    
    if mode == "combine":
        result = torch.maximum(masks[0], masks[1])
        for mask in masks[2:]:
            result = torch.maximum(result, mask)
    elif mode == "intersection":
        result = torch.minimum(masks[0], masks[1])
    else:
        result = torch.abs(masks[0] - masks[1])
    return (torch.clamp(result, 0, 1),)

def resize_mask(mask, target_shape):
    if mask.shape == target_shape:
        return mask
        
    target_height = target_shape[-2] if len(target_shape) >= 2 else target_shape[0]
    target_width = target_shape[-1] if len(target_shape) >= 2 else target_shape[1]
    
    if mask.shape[-2] == target_height and mask.shape[-1] == target_width:
        return mask
        
    orig_shape = mask.shape
    
    if mask.ndim == 2:
        mask_view = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask_view = mask.unsqueeze(1)
    else:
        if mask.shape[-1] in [1, 3, 4] and mask.shape[-3] not in [1, 3, 4]:
            mask_view = mask.permute(0, 3, 1, 2)
        else:
            mask_view = mask
            
    resized = torch.nn.functional.interpolate(mask_view.float(), size=(target_height, target_width), mode="bilinear", align_corners=False)
    
    if len(orig_shape) == 2:
        return resized.squeeze(0).squeeze(0)
    elif len(orig_shape) == 3:
        return resized.squeeze(1)
    else:
        if mask.shape[-1] in [1, 3, 4] and mask.shape[-3] not in [1, 3, 4]:
            return resized.permute(0, 2, 3, 1)
        return resized

_resize_if_needed = resize_mask

def denormalize_and_resize(tensor, target_w, target_h):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor).to(device)
    img_float = (tensor.float() * 0.5 + 0.5) * 255.0
    if img_float.shape[2] != target_w or img_float.shape[1] != target_h:
        img_float = torch.nn.functional.interpolate(img_float.unsqueeze(0), size=(target_h, target_w), mode="bicubic", align_corners=False).squeeze(0)
    img = img_float.permute(1, 2, 0).to(torch.uint8)
    return img

def join_image_with_alpha(image: torch.Tensor, alpha: torch.Tensor, invert=False):
    batch_size = min(image.shape[0], alpha.shape[0])
    
    is_channel_first = image.shape[-3] in [1, 3, 4] and image.shape[-1] not in [1, 3, 4]
    spatial_shape = image.shape[-2:] if is_channel_first else image.shape[-3:-1]
    
    alpha_resized = resize_mask(alpha[:batch_size], spatial_shape)
    if invert: alpha_resized = 1.0 - alpha_resized
        
    image_batch = image[:batch_size]
    
    if is_channel_first:
        rgb = image_batch[:, :3, :, :]
        alpha_exp = alpha_resized.unsqueeze(1)
        out_images = torch.cat((rgb, alpha_exp), dim=1)
    else:
        rgb = image_batch[:, :, :, :3]
        alpha_exp = alpha_resized.unsqueeze(-1)
        out_images = torch.cat((rgb, alpha_exp), dim=-1)
        
    return (out_images,)

def convert(image=None, mask=None, mask_channel="alpha"):
    
    if image is None and mask is None:
        return (torch.zeros(1, 3, 64, 64), torch.zeros(1, 64, 64))
        
    if image is None and mask is not None:
        if mask.ndim == 4:
            if mask.shape[1] == 1:
                return (mask.expand(-1, 3, -1, -1), mask.squeeze(1))
            elif mask.shape[-1] == 1:
                return (mask.permute(0, 3, 1, 2).expand(-1, 3, -1, -1), mask.squeeze(-1))
            return (mask[:, :3, :, :] if mask.shape[1] in [3,4] else mask.permute(0, 3, 1, 2)[:, :3, :, :], mask.mean(dim=1 if mask.shape[1] in [3,4] else -1))
        elif mask.ndim == 3:
            return (mask.unsqueeze(1).expand(-1, 3, -1, -1), mask)
        elif mask.ndim == 2:
            return (mask.unsqueeze(0).unsqueeze(1).expand(-1, 3, -1, -1), mask.unsqueeze(0))
        else:
            print(f"Invalid mask shape: {mask.shape}")
            return (torch.zeros(1, 3, 64, 64), mask)
            
    if image is not None and mask is None:
        is_channel_first = image.ndim >= 3 and image.shape[-3] in [1, 3, 4] and image.shape[-1] not in [1, 3, 4]
        channels = image.shape[-3] if is_channel_first else image.shape[-1]
        
        channel_map = {"red": 0, "green": 1, "blue": 2, "alpha": 3}
        c_idx = channel_map.get(mask_channel, 3)
        
        if c_idx < channels:
            if is_channel_first:
                result_mask = image[..., c_idx, :, :]
            else:
                result_mask = image[..., c_idx]
        else:
            spatial = image.shape[-2:] if is_channel_first else image.shape[-3:-1]
            batch_size = image.shape[0] if image.ndim == 4 else 1
            result_mask = torch.ones((batch_size, *spatial), dtype=image.dtype, device=image.device)
            if image.ndim == 3: result_mask = result_mask.squeeze(0)
            
        return (image, result_mask)

    if image is not None and mask is not None:
        if mask.ndim == 4:
            mask = mask.squeeze(1 if mask.shape[1] == 1 else -1)
        return (image, mask)

def run_process(cmd, log_callback, process_callback=None):
    if log_callback:
        log_callback(f"Executing: {' '.join(cmd)}")
    else:
        print(f"Executing: {' '.join(cmd)}")
        
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, errors='replace')
    if process_callback: process_callback(process)
    
    for line in process.stdout:
        if log_callback: log_callback(line.strip())
    process.wait()
    if process.returncode != 0:
        raise Exception(f"Command failed with code {process.returncode}")

def video_frame_generator(video, force_rate=0, frame_load_cap=0, skip_first_frames=0, select_every_nth=1, start_time=0.0, output_format="tensor", **kwargs):
    try:
        container = av.open(video)
        video_stream = container.streams.video[0]
        video_stream.thread_type = "AUTO"
        
        fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
        width = video_stream.width
        height = video_stream.height
        duration = float(video_stream.duration * video_stream.time_base) if video_stream.duration else 0.0
        total_frames = video_stream.frames if video_stream.frames > 0 else int(duration * fps)
        
        target_fps = force_rate if force_rate > 0 else fps
        target_frame_time = 1.0 / target_fps
        
        yieldable_frames = total_frames
        if start_time > 0: yieldable_frames -= int(start_time * fps)
        if skip_first_frames > 0: yieldable_frames -= skip_first_frames
        if force_rate > 0: yieldable_frames = int(yieldable_frames * (force_rate / fps))
        if select_every_nth > 1: yieldable_frames //= select_every_nth
        if frame_load_cap > 0: yieldable_frames = min(yieldable_frames, frame_load_cap)
        
        yield (width, height, fps, duration, total_frames, target_frame_time, yieldable_frames)
        
        if start_time > 0:
            seek_pts = int(start_time / video_stream.time_base)
            container.seek(seek_pts, stream=video_stream)
            
        frames_yielded = 0
        frame_idx = -1
        current_time = 0.0
        
        for frame in container.decode(video_stream):
            frame_idx += 1
            if frame_idx < skip_first_frames:
                continue
                
            if frame_idx % select_every_nth != 0:
                continue
                
            if force_rate > 0:
                frame_time_sec = float(frame.pts * video_stream.time_base) if frame.pts else frame_idx / fps
                if frame_time_sec < current_time:
                    continue
                current_time += target_frame_time
                
            if output_format == "bgr24":
                out_frame = frame.to_ndarray(format='bgr24')
            else:
                img_np = frame.to_ndarray(format='rgb24')
                out_frame = torch.from_numpy(img_np).float() / 255.0
            
            inp = yield out_frame
            if inp is not None:
                return
            
            frames_yielded += 1
            if frame_load_cap > 0 and frames_yielded >= frame_load_cap:
                break
                
    finally:
        if 'container' in locals():
            container.close()
            
    if 'meta_batch' in kwargs and kwargs.get('meta_batch') is not None:
        unique_id = kwargs.get('unique_id')
        meta_batch = kwargs['meta_batch']
        if unique_id in meta_batch.inputs:
            meta_batch.inputs.pop(unique_id)
        meta_batch.has_closed_inputs = True

def bislerp(samples, width, height):
    def slerp(b1, b2, r):
        c = b1.shape[-1]
        b1_norms = torch.norm(b1, dim=-1, keepdim=True)
        b2_norms = torch.norm(b2, dim=-1, keepdim=True)

        b1_normalized = b1 / b1_norms
        b2_normalized = b2 / b2_norms
        b1_normalized[b1_norms.expand(-1,c) == 0.0] = 0.0
        b2_normalized[b2_norms.expand(-1,c) == 0.0] = 0.0
        dot = (b1_normalized*b2_normalized).sum(1)
        omega = torch.acos(dot)
        so = torch.sin(omega)

        res = (torch.sin((1.0-r.squeeze(1))*omega)/so).unsqueeze(1)*b1_normalized + (torch.sin(r.squeeze(1)*omega)/so).unsqueeze(1) * b2_normalized
        res *= (b1_norms * (1.0-r) + b2_norms * r).expand(-1,c)
        res[dot > 1 - 1e-5] = b1[dot > 1 - 1e-5]
        res[dot < 1e-5 - 1] = (b1 * (1.0-r) + b2 * r)[dot < 1e-5 - 1]
        return res

    def generate_bilinear_data(length_old, length_new, device):
        coords_1 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1))
        coords_1 = torch.nn.functional.interpolate(coords_1, size=(1, length_new), mode="bilinear")
        ratios = coords_1 - coords_1.floor()
        coords_1 = coords_1.to(torch.int64)

        coords_2 = torch.arange(length_old, dtype=torch.float32, device=device).reshape((1,1,1,-1)) + 1
        coords_2[:,:,:,-1] -= 1
        coords_2 = torch.nn.functional.interpolate(coords_2, size=(1, length_new), mode="bilinear")
        coords_2 = coords_2.to(torch.int64)
        return ratios, coords_1, coords_2

    orig_dtype = samples.dtype
    samples = samples.float()
    n,c,h,w = samples.shape
    h_new, w_new = (height, width)

    ratios, coords_1, coords_2 = generate_bilinear_data(w, w_new, samples.device)
    coords_1 = coords_1.expand((n, c, h, -1))
    coords_2 = coords_2.expand((n, c, h, -1))
    ratios = ratios.expand((n, 1, h, -1))

    pass_1 = samples.gather(-1,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = samples.gather(-1,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h, w_new, c).movedim(-1, 1)

    ratios, coords_1, coords_2 = generate_bilinear_data(h, h_new, samples.device)
    coords_1 = coords_1.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    coords_2 = coords_2.reshape((1,1,-1,1)).expand((n, c, -1, w_new))
    ratios = ratios.reshape((1,1,-1,1)).expand((n, 1, -1, w_new))

    pass_1 = result.gather(-2,coords_1).movedim(1, -1).reshape((-1,c))
    pass_2 = result.gather(-2,coords_2).movedim(1, -1).reshape((-1,c))
    ratios = ratios.movedim(1, -1).reshape((-1,1))

    result = slerp(pass_1, pass_2, ratios)
    result = result.reshape(n, h_new, w_new, c).movedim(-1, 1)
    return result.to(orig_dtype)

def lanczos(samples, width, height):
    if samples.ndim == 4:
        samples = samples.squeeze(1) if samples.shape[1] == 1 else samples.movedim(1, -1)
    images = [Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)) for image in samples]
    images = [image.resize((width, height), resample=Image.Resampling.LANCZOS) for image in images]
    images = [torch.from_numpy(t).movedim(-1, 0) if (t := np.array(image).astype(np.float32) / 255.0).ndim == 3 else torch.from_numpy(t) for image in images]
    result = torch.stack(images)
    return result.to(samples.device, samples.dtype)

def common_upscale(samples, width, height, upscale_method, crop):
        orig_shape = tuple(samples.shape)
        if len(orig_shape) > 4:
            samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
            samples = samples.movedim(2, 1)
            samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
        if crop == "center":
            old_width = samples.shape[-1]
            old_height = samples.shape[-2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        else:
            s = samples

        if upscale_method == "bislerp":
            out = bislerp(s, width, height)
        elif upscale_method == "lanczos":
            out = lanczos(s, width, height)
        else:
            out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

        if len(orig_shape) == 4:
            return out

        out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
        return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))

def bytesio_to_image_tensor(image_bytesio: BytesIO, mode: str = "RGBA") -> torch.Tensor:
    image = Image.open(image_bytesio)
    image = image.convert(mode)
    image_array = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)

def image_tensor_pair_to_batch(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:

    if image1.shape[1:] != image2.shape[1:]:
        image2 = common_upscale(
            image2.movedim(-1, 1),
            image1.shape[2],
            image1.shape[1],
            "bilinear",
            "center",
        ).movedim(1, -1)
    return torch.cat((image1, image2), dim=0)

def tensor_to_pil(image: torch.Tensor, total_pixels: int | None = 2048 * 2048) -> Image.Image:

    if len(image.shape) > 3:
        image = image[0]
    input_tensor = image.cpu()
    
    if total_pixels is not None:
        input_tensor = downscale_image_tensor(input_tensor.unsqueeze(0), total_pixels=total_pixels).squeeze()
    
    image_np = (input_tensor.numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    return img

def tensor_to_base64_string(image_tensor: torch.Tensor, total_pixels: int | None = 2048 * 2048, mime_type: str = "image/png") -> str:

    pil_image = tensor_to_pil(image_tensor, total_pixels)
    img_byte_arr = pil_to_bytesio(pil_image, mime_type=mime_type)
    img_bytes = img_byte_arr.getvalue()
    base64_encoded_string = base64.b64encode(img_bytes).decode("utf-8")
    return base64_encoded_string

def pil_to_bytesio(img: Image.Image, mime_type: str = "image/png") -> BytesIO:

    if not mime_type:
        mime_type = "image/png"

    img_byte_arr = BytesIO()
    pil_format = mime_type.split("/")[-1].upper()
    
    if pil_format == "JPG":
        pil_format = "JPEG"
   
    img.save(img_byte_arr, format=pil_format)
    img_byte_arr.seek(0)
    return img_byte_arr

def _compute_downscale_dims(src_w: int, src_h: int, total_pixels: int) -> tuple[int, int] | None:

    pixels = src_w * src_h
    if pixels <= total_pixels:
        return None
    
    scale = math.sqrt(total_pixels / pixels)
    new_w = max(2, int(src_w * scale))
    new_h = max(2, int(src_h * scale))
    new_w -= new_w % 2
    new_h -= new_h % 2
    return new_w, new_h

def downscale_image_tensor(image: torch.Tensor, total_pixels: int = 1536 * 1024) -> torch.Tensor:
    samples = image.movedim(-1, 1)
    dims = _compute_downscale_dims(samples.shape[3], samples.shape[2], int(total_pixels))

    if dims is None:
        return image
    
    new_w, new_h = dims
    return common_upscale(samples, new_w, new_h, "lanczos", "disabled").movedim(1, -1)

def downscale_image_tensor_by_max_side(image: torch.Tensor, *, max_side: int) -> torch.Tensor:

    samples = image.movedim(-1, 1)
    height, width = samples.shape[2], samples.shape[3]
    max_dim = max(width, height)

    if max_dim <= max_side:
        return image
    
    scale_by = max_side / max_dim
    new_width = round(width * scale_by)
    new_height = round(height * scale_by)
    s = common_upscale(samples, new_width, new_height, "lanczos", "disabled")
    s = s.movedim(1, -1)
    return s

def tensor_to_data_uri(image_tensor: torch.Tensor, total_pixels: int | None = 2048 * 2048, mime_type: str = "image/png") -> str:
    base64_string = tensor_to_base64_string(image_tensor, total_pixels, mime_type)
    return f"data:{mime_type};base64,{base64_string}"

def downscale_video_to_max_pixels(video, max_pixels: int):

    src_w, src_h = video.get_dimensions()
    scale_dims = _compute_downscale_dims(src_w, src_h, max_pixels)
    if scale_dims is None:
        return video
    return _apply_video_scale(video, scale_dims)

def _compute_upscale_dims(src_w: int, src_h: int, total_pixels: int) -> tuple[int, int] | None:
    pixels = src_w * src_h
    if pixels >= total_pixels:
        return None

    scale = math.sqrt(total_pixels / pixels)
    new_w = math.ceil(src_w * scale)
    new_h = math.ceil(src_h * scale)

    if new_w % 2:
        new_w += 1
    if new_h % 2:
        new_h += 1
    return new_w, new_h

def upscale_video_to_min_pixels(video, min_pixels: int):

    src_w, src_h = video.get_dimensions()
    scale_dims = _compute_upscale_dims(src_w, src_h, min_pixels)

    if scale_dims is None:
        return video
    return _apply_video_scale(video, scale_dims)

def _apply_video_scale(video, scale_dims: tuple[int, int]):

    out_w, out_h = scale_dims
    output_buffer = BytesIO()
    input_container = None
    output_container = None

    input_source = video.get_stream_source()
    input_container = av.open(input_source, mode="r")
    output_container = av.open(output_buffer, mode="w", format="mp4")

    video_stream = output_container.add_stream("h264", rate=video.get_frame_rate())
    video_stream.width = out_w
    video_stream.height = out_h
    video_stream.pix_fmt = "yuv420p"

    audio_stream = None
    for stream in input_container.streams:
        if isinstance(stream, av.AudioStream):
            audio_stream = output_container.add_stream("aac", rate=stream.sample_rate)
            audio_stream.sample_rate = stream.sample_rate
            audio_stream.layout = stream.layout
            break

    for frame in input_container.decode(video=0):
        frame = frame.reformat(width=out_w, height=out_h, format="yuv420p")
        for packet in video_stream.encode(frame):
            output_container.mux(packet)
    for packet in video_stream.encode():
        output_container.mux(packet)

    if audio_stream is not None:
        input_container.seek(0)
        for audio_frame in input_container.decode(audio=0):
            for packet in audio_stream.encode(audio_frame):
                output_container.mux(packet)
        for packet in audio_stream.encode():
            output_container.mux(packet)

    output_container.close()
    input_container.close()
    output_buffer.seek(0)
    return torch.Tensor.VideoFromFile(output_buffer)

def text_filepath_to_base64_string(filepath: str) -> str:
    with open(filepath, "rb") as f:
        file_content = f.read()
    return base64.b64encode(file_content).decode("utf-8")

def resize_mask_to_image(mask: torch.Tensor, image: torch.Tensor, upscale_method="nearest-exact", crop="disabled",
    allow_gradient=True, add_channel_dim=False):

    _, height, width, _ = image.shape
    mask = mask.unsqueeze(-1)
    mask = mask.movedim(-1, 1)
    mask = common_upscale(mask, width=width, height=height, upscale_method=upscale_method, crop=crop)
    mask = mask.movedim(1, -1)
    if not add_channel_dim:
        mask = mask.squeeze(-1)
    if not allow_gradient:
        mask = (mask > 0.5).float()
    return mask

def repeat_to_batch_size(tensor, batch_size, dim=0):
    if tensor.shape[dim] > batch_size:
        return tensor.narrow(dim, 0, batch_size)
    elif tensor.shape[dim] < batch_size:
        return tensor.repeat(dim * [1] + [math.ceil(batch_size / tensor.shape[dim])] + [1] * (len(tensor.shape) - 1 - dim)).narrow(dim, 0, batch_size)
    return tensor

def resize_to_batch_size(tensor, batch_size):
    in_batch_size = tensor.shape[0]
    if in_batch_size == batch_size:
        return tensor

    if batch_size <= 1:
        return tensor[:batch_size]

    output = torch.empty([batch_size] + list(tensor.shape)[1:], dtype=tensor.dtype, device=tensor.device)
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output[i] = tensor[min(round(i * scale), in_batch_size - 1)]
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
            output[i] = tensor[min(math.floor((i + 0.5) * scale), in_batch_size - 1)]

    return output

def resize_list_to_batch_size(l, batch_size):
    in_batch_size = len(l)
    if in_batch_size == batch_size or in_batch_size == 0:
        return l

    if batch_size <= 1:
        return l[:batch_size]

    output = []
    if batch_size < in_batch_size:
        scale = (in_batch_size - 1) / (batch_size - 1)
        for i in range(batch_size):
            output.append(l[min(round(i * scale), in_batch_size - 1)])
    else:
        scale = in_batch_size / batch_size
        for i in range(batch_size):
           output.append(l[min(math.floor((i + 0.5) * scale), in_batch_size - 1)])

    return output

def convert_mask_to_image(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1)
    return torch.cat([mask] * 3, dim=-1)

def load_resource_as_video_frames(
    resource_path, image_size, offload_video_to_cpu, img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False, video_loader_type="cv2", start_frame=0, max_frames=None
):
    if isinstance(resource_path, list):
        img_mean_t = torch.tensor(img_mean, dtype=torch.float16).view(3, 1, 1)
        img_std_t = torch.tensor(img_std, dtype=torch.float16).view(3, 1, 1)
        orig_height, orig_width = resource_path[0].size[1], resource_path[0].size[0]
        
        images = []
        for img_pil in resource_path:
            img_t = torch.from_numpy(np.array(img_pil.convert("RGB").resize((image_size, image_size)))).permute(2, 0, 1)
            img_t = img_t.to(dtype=torch.float16) / 255.0
            images.append((img_t - img_mean_t) / img_std_t)
            
        images = torch.stack(images)
        if not offload_video_to_cpu: images = images.cuda()
        return images, orig_height, orig_width

    return load_video_frames(
        video_path=resource_path, image_size=image_size, offload_video_to_cpu=offload_video_to_cpu,
        img_mean=img_mean, img_std=img_std, async_loading_frames=async_loading_frames,
        video_loader_type=video_loader_type, start_frame=start_frame, max_frames=max_frames
    )

def load_video_frames(
    video_path, image_size, offload_video_to_cpu, img_mean=(0.5, 0.5, 0.5), img_std=(0.5, 0.5, 0.5),
    async_loading_frames=False, video_loader_type="cv2", start_frame=0, max_frames=None
):
    IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]

    if video_path.startswith("<load-dummy-video"):
        num_frames = int(re.match(r"<load-dummy-video-(\d+)>", video_path).group(1)) if re.match(r"<load-dummy-video-(\d+)>", video_path) else 60
        images = torch.randn(num_frames, 3, image_size, image_size, dtype=torch.float16)
        if not offload_video_to_cpu: images = images.cuda()
        return images, 480, 640
        
    ext = os.path.splitext(video_path)[-1].lower()
    if ext not in VIDEO_EXTS and ext not in IMAGE_EXTS:
        raise NotImplementedError("Only video files and standard image formats are supported.")

    if video_loader_type == "cv2":
        return load_video_cv2(video_path, image_size, img_mean, img_std, offload_video_to_cpu, start_frame, max_frames)
    elif video_loader_type == "torchcodec":
        logger.info("Using ultra-fast TorchCodec video loader.")
        lazy_loader = TorchCodecVideoLoader(video_path, image_size, offload_video_to_cpu, img_mean, img_std)
        
        if not async_loading_frames:
            if lazy_loader.thread: lazy_loader.thread.join()
            return lazy_loader.get_all_frames(start_frame, max_frames), lazy_loader.video_height, lazy_loader.video_width
            
        return lazy_loader, lazy_loader.video_height, lazy_loader.video_width
    else:
        raise RuntimeError("video_loader_type must be either 'cv2' or 'torchcodec'")

def load_video_cv2(video_path, image_size, img_mean, img_std, offload_video_to_cpu, start_frame=0, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError(f"Could not open video: {video_path}")

    orig_height, orig_width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if start_frame > 0: cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
    frames = []
    pbar = tqdm(desc=f"frame loading (OpenCV)]", total=min(num_frames - start_frame, max_frames) if max_frames else num_frames - start_frame)
    
    count = 0
    while True:
        if max_frames and count >= max_frames: break
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (image_size, image_size), interpolation=cv2.INTER_CUBIC)
        frames.append(frame)
        pbar.update(1)
        count += 1
        
    cap.release()
    pbar.close()

    if not frames: raise RuntimeError(f"No frames decoded from {video_path}")

    video_tensor = torch.from_numpy(np.stack(frames, axis=0)).permute(0, 3, 1, 2).to(dtype=torch.float16) / 255.0
    mean_t = torch.tensor(img_mean, dtype=torch.float16).view(1, 3, 1, 1)
    std_t = torch.tensor(img_std, dtype=torch.float16).view(1, 3, 1, 1)
    
    if not offload_video_to_cpu:
        video_tensor, mean_t, std_t = video_tensor.cuda(), mean_t.cuda(), std_t.cuda()
        
    video_tensor = (video_tensor - mean_t) / std_t
    return video_tensor, orig_height, orig_width

class TorchCodecVideoLoader:

    def __init__(self, video_path, image_size, offload_video_to_cpu, img_mean, img_std, gpu_device=None):
        from torchcodec import _core as core
        
        self.image_size = image_size
        self.out_device = torch.device("cpu") if offload_video_to_cpu else (gpu_device or torch.device("cuda"))
        decode_device = (gpu_device or torch.device("cuda")) if torch.cuda.is_available() else torch.device("cpu")
        
        self.img_mean = torch.tensor(img_mean, dtype=torch.float16, device=self.out_device).view(3, 1, 1)
        self.img_std = torch.tensor(img_std, dtype=torch.float16, device=self.out_device).view(3, 1, 1)
        
        self.decoder = core.create_from_file(video_path, "exact")
        core.scan_all_streams_to_update_metadata(self.decoder)
        core.add_video_stream(
            self.decoder, dimension_order="NCHW", device=str(decode_device), 
            num_threads=1 if decode_device.type == "cuda" else 4
        )
        
        meta = core.get_container_metadata(self.decoder)
        stream = meta.streams[meta.best_video_stream_index]
        self.num_frames = stream.num_frames_from_content
        self.video_height = stream.height
        self.video_width = stream.width
        
        self.images = [None] * self.num_frames
        self.exception = None
        
        self.thread = threading.Thread(target=self._background_decode, daemon=True)
        self.thread.start()
        
    @torch.inference_mode()
    def _background_decode(self):
        from torchcodec import _core as core
        try:
            pbar = tqdm(desc=f"frame loading (TorchCodec) ]", total=self.num_frames)
            for i in range(self.num_frames):
                frame_data, *_ = core.get_frame_at_index(self.decoder, frame_index=i)
                frame = frame_data.float()
          
                if self.image_size:
                    frame = torch.nn.functional.interpolate(frame.unsqueeze(0), size=(self.image_size, self.image_size), mode="bicubic", align_corners=False).squeeze(0)
                    
                frame = frame.half() / 255.0
                if frame.device != self.out_device:
                    frame = frame.to(self.out_device, non_blocking=True)
                    
                frame = (frame - self.img_mean) / self.img_std
                
                self.images[i] = frame
                pbar.update(1)
            pbar.close()
        except Exception as e:
            self.exception = e

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        if idx < 0: idx += self.num_frames
        if idx < 0 or idx >= self.num_frames: raise IndexError("Frame index out of bounds")
        
        max_retries = 1200
        for _ in range(max_retries):
            if self.exception: raise RuntimeError("Background decoding failed") from self.exception
            if self.images[idx] is not None:
                return self.images[idx]
            time.sleep(0.01)
            
        raise RuntimeError(f"Timeout waiting for frame {idx} to decode.")
        
    def get_all_frames(self, start=0, max_frames=None):
        end = min(start + max_frames, self.num_frames) if max_frames else self.num_frames
        return torch.stack([self[i] for i in range(start, end)])