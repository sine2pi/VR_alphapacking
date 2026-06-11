
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from model.edt import edt_triton

def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,
    noise_bound: int = 20,
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> tuple[NDArray, NDArray]:

    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label], dtype=torch.int, device=device
    ).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        img_bounds = (
            torch.tensor([W, H, W, H], device=device) - 1
        )
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)

    box_coords = box_coords.reshape(-1, 2, 2)
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels

def mask_to_box(masks: torch.Tensor):

    B, _, h, w = masks.shape
    device = masks.device
    mask_area = masks.sum(dim=(-1, -2))
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)
    bbox_coords = torch.where(
        mask_area[..., None] > 0, bbox_coords, torch.zeros_like(bbox_coords)
    )
    return bbox_coords

def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device

    fp_masks = ~gt_masks & pred_masks
    fn_masks = gt_masks & ~pred_masks
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels

def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, H, W = gt_masks.shape

    fp_masks = (~gt_masks & pred_masks).squeeze(1)
    fn_masks = (gt_masks & ~pred_masks).squeeze(1)

    if padding:
        padded_fp_masks = torch.zeros(
            B, H + 2, W + 2, dtype=fp_masks.dtype, device=fp_masks.device
        )
        padded_fp_masks[:, 1 : H + 1, 1 : W + 1] = fp_masks
        padded_fn_masks = torch.zeros(
            B, H + 2, W + 2, dtype=fp_masks.dtype, device=fp_masks.device
        )
        padded_fn_masks[:, 1 : H + 1, 1 : W + 1] = fn_masks
    else:
        padded_fp_masks = fp_masks
        padded_fn_masks = fn_masks

    fn_mask_dt = edt_triton(padded_fn_masks)
    fp_mask_dt = edt_triton(padded_fp_masks)
    if padding:
        fn_mask_dt = fn_mask_dt[:, 1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[:, 1:-1, 1:-1]

    fn_max, fn_argmax = fn_mask_dt.reshape(B, -1).max(dim=-1)
    fp_max, fp_argmax = fp_mask_dt.reshape(B, -1).max(dim=-1)
    is_positive = fn_max > fp_max
    chosen = torch.where(is_positive, fn_argmax, fp_argmax)
    points_x = chosen % W
    points_y = chosen // W

    labels = is_positive.long()
    points = torch.stack([points_x, points_y], -1)
    return points.unsqueeze(1), labels.unsqueeze(1)

def sample_one_point_from_error_center_slow(gt_masks, pred_masks, padding=True):

    import cv2

    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device

    fp_masks = ~gt_masks & pred_masks
    fn_masks = gt_masks & ~pred_masks

    fp_masks = fp_masks.cpu().numpy()
    fn_masks = fn_masks.cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn_mask = fn_masks[b, 0]
        fp_mask = fp_masks[b, 0]
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), "constant")
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), "constant")
        fn_mask_dt = cv2.distanceTransform(fn_mask.astype(np.uint8), cv2.DIST_L2, 0)
        fp_mask_dt = cv2.distanceTransform(fp_mask.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        fn_mask_dt_flat = fn_mask_dt.reshape(-1)
        fp_mask_dt_flat = fp_mask_dt.reshape(-1)
        fn_argmax = np.argmax(fn_mask_dt_flat)
        fp_argmax = np.argmax(fp_mask_dt_flat)
        is_positive = fn_mask_dt_flat[fn_argmax] > fp_mask_dt_flat[fp_argmax]
        pt_idx = fn_argmax if is_positive else fp_argmax
        points[b, 0, 0] = pt_idx % W_im
        points[b, 0, 1] = pt_idx // W_im
        labels[b, 0] = int(is_positive)

    points = points.to(device)
    labels = labels.to(device)
    return points, labels

def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    elif method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    else:
        raise ValueError(f"unknown sampling method {method}")

def select_closest_cond_frames(
    frame_idx, cond_frame_outputs, max_cond_frame_num, keep_first_cond_frame=False
):

    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        selected_outputs = cond_frame_outputs
        unselected_outputs = {}
    else:
        assert max_cond_frame_num >= 2, "we should allow using 2+ conditioning frames"
        selected_outputs = {}
        if keep_first_cond_frame:
            idx_first = min(
                (t for t in cond_frame_outputs if t < frame_idx), default=None
            )
            if idx_first is None:
                idx_first = max(
                    (t for t in cond_frame_outputs if t > frame_idx), default=None
                )
            if idx_first is not None:
                selected_outputs[idx_first] = cond_frame_outputs[idx_first]
        idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
        if idx_before is not None:
            selected_outputs[idx_before] = cond_frame_outputs[idx_before]

        idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
        if idx_after is not None:
            selected_outputs[idx_after] = cond_frame_outputs[idx_after]

        num_remain = max_cond_frame_num - len(selected_outputs)
        inds_remain = sorted(
            (t for t in cond_frame_outputs if t not in selected_outputs),
            key=lambda x: abs(x - frame_idx),
        )[:num_remain]
        selected_outputs.update((t, cond_frame_outputs[t]) for t in inds_remain)
        unselected_outputs = {
            t: v for t, v in cond_frame_outputs.items() if t not in selected_outputs
        }

    return selected_outputs, unselected_outputs

def get_1d_sine_pe(pos_inds, dim, temperature=10000):

    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)

    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    pos_embed = torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)
    return pos_embed

def get_best_gt_match_from_multimasks(pred_multimasks, gt_masks, pred_scores=None):

    assert pred_multimasks.ndim == 4 and gt_masks.ndim == 4
    if pred_multimasks.size(1) == 1:
        return pred_multimasks

    pred_multimasks_binary = pred_multimasks > 0
    area_i = torch.sum(pred_multimasks_binary & gt_masks, dim=(2, 3)).float()
    area_u = torch.sum(pred_multimasks_binary | gt_masks, dim=(2, 3)).float()
    ious = area_i / torch.clamp(area_u, min=1.0)

    if pred_scores is not None:
        has_nonzero_ious = torch.any(ious > 0).expand_as(ious)
        scores = torch.where(has_nonzero_ious, ious, pred_scores)
    else:
        scores = ious

    best_scores_inds = torch.argmax(scores, dim=-1)
    batch_inds = torch.arange(scores.size(0), device=scores.device)
    best_pred_mask = pred_multimasks[batch_inds, best_scores_inds].unsqueeze(1)
    return best_pred_mask

def fill_holes_in_mask_scores(mask, max_area, fill_holes=True, remove_sprinkles=True):

    if max_area <= 0:
        return mask

    if fill_holes:
        mask_bg = mask <= 0
        bg_area_thresh = max_area
        _, areas_bg = _get_connected_components_with_padding(mask_bg)
        small_components_bg = mask_bg & (areas_bg <= bg_area_thresh)
        mask = torch.where(small_components_bg, 0.1, mask)

    if remove_sprinkles:
        mask_fg = mask > 0
        fg_area_thresh = torch.sum(mask_fg, dim=(2, 3), keepdim=True, dtype=torch.int32)
        fg_area_thresh.floor_divide_(2).clamp_(max=max_area)
        _, areas_fg = _get_connected_components_with_padding(mask_fg)
        small_components_fg = mask_fg & (areas_fg <= fg_area_thresh)
        mask = torch.where(small_components_fg, -0.1, mask)
    return mask

def _get_connected_components_with_padding(mask):

    from perflib.connected_components import connected_components

    mask = mask.to(torch.uint8)
    _, _, H, W = mask.shape
    pad_h = H % 2
    pad_w = W % 2
    if pad_h == 0 and pad_w == 0:
        labels, counts = connected_components(mask)
    else:
        mask_pad = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)
        labels, counts = connected_components(mask_pad)
        labels = labels[:, :, :H, :W]
        counts = counts[:, :, :H, :W]

    return labels, counts
