import cv2, torch, subprocess, numpy as np, json, logging, os, glob, gc, math
from typing import List, Optional
import torch.nn.functional as F
from tqdm import tqdm
import logging
from essentials import *
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr
from model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from model.geometry_encoders import SequenceGeometryEncoder
from model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from model.necks import Sam3DualViTDetNeck
from model.position_encoding import PositionEmbeddingSine
from model.sam1_task_predictor import SAM3InteractiveImagePredictor
from model.sam3_image import Sam3Image, Sam3ImageOnVideoMultiGPU
from model.sam3_tracking_predictor import Sam3TrackerPredictor
from model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from model.text_encoder_ve import VETextEncoder
from model.tokenizer_ve import SimpleTokenizer
from model.vitdet import ViT
from model.vl_combiner import SAM3VLBackbone
from sam.transformer import RoPEAttention
import torchvision.transforms.functional as V
try:
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights # from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    HAS_RAFT = True
except ImportError:
    HAS_RAFT = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def l2norm(t):
    return F.normalize(t, dim = -1)

def exact_div(x, y):
    assert x % y == 0
    return x // y

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
    
def _setup_tf32() -> None:

    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

_setup_tf32()

def _create_position_encoding(precompute_resolution=None):

    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )

def _create_vit_backbone(compile_mode=None):

    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )

def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):

    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )

def _create_vl_backbone(vit_neck, text_encoder):
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)

def _create_transformer_encoder() -> TransformerEncoderFusion:

    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder

def _create_transformer_decoder(presence_token=True) -> TransformerDecoder:

    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=presence_token,
    )
    return decoder

def _create_dot_product_scoring():

    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)

def _create_segmentation_head(compile_mode=None):

    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head

def _create_geometry_encoder(geo_encoder_use_img_cross_attn=True):

    geo_pos_enc = _create_position_encoding()
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,

        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder

def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
):

    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    matcher = None
    if not eval_mode:
        from train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )
    common_params["matcher"] = matcher
    model = Sam3Image(**common_params)

    return model

def _create_tracker_maskmem_backbone():
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone

def _create_tracker_transformer():

    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )

    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )

    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )

    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )

    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer

def build_tracker(
    apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None
) -> Sam3TrackerPredictor:

    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        multimask_output_in_sam=False,
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        always_start_from_first_ann_frame=False,
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )

    return model

def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )

def _create_vision_backbone(
    compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:

    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    return vit_neck

def _create_sam3_transformer(has_presence_token: bool = True) -> TransformerWrapper:

    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder(presence_token=has_presence_token)

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)

def _load_checkpoint(model, checkpoint_path):

    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
    missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
    if len(missing_keys) > 0:
        print(
            f"loaded {checkpoint_path} and found "
            f"missing and/or unexpected keys:\n{missing_keys=}"
        )

def _setup_device_and_mode(model, device, eval_mode):
    if device == "cuda":
        model = model.cuda()
    if eval_mode:
        model.eval()
    return model

def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
):

    if bpe_path is None:
        bpe_path = os.path.join(os.path.dirname(__file__), "assets/bpe_simple_vocab_16e6.txt.gz")

    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(
        compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
    )

    text_encoder = _create_text_encoder(bpe_path)
    backbone = _create_vl_backbone(vision_encoder, text_encoder)
    transformer = _create_sam3_transformer()
    dot_prod_scoring = _create_dot_product_scoring()

    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )

    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )
    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path)

    model = _setup_device_and_mode(model, device, eval_mode)

    return model

def download_ckpt_from_hf():
    SAM3_MODEL_ID = "Translsis/sam3-model"
    SAM3_CKPT_NAME = "assets/sam3.pt"
    SAM3_CFG_NAME = "config.json"
    _ = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CFG_NAME, force_download=False)
    checkpoint_path = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CKPT_NAME, force_download=False)
    return checkpoint_path

def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF: bool = False,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = True,
    apply_temporal_disambiguation: bool = False,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
) -> Sam3VideoInferenceWithInstanceInteractivity:

    if bpe_path is None:
        bpe_path = os.path.join(os.path.dirname(__file__), "assets/bpe_simple_vocab_16e6.txt.gz")

    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(__file__), "assets/sam3.pt")

    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)

    visual_neck = _create_vision_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
    segmentation_head: UniversalSegmentationHead = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder(geo_encoder_use_img_cross_attn=geo_encoder_use_img_cross_attn)

    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
    )

    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
    )

    if apply_temporal_disambiguation:
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.65,
            assoc_iou_thresh=0.2,
            det_nms_thresh=0.5,
            new_det_thresh=0.99,
            hotstart_delay=0,
            hotstart_unmatch_thresh=0,
            hotstart_dup_thresh=0,
            suppress_unmatched_only_within_hotstart=False,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=100,
            init_trk_keep_alive=10,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.5,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=32,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )
    else:
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.75,
            assoc_iou_thresh=0.15,
            det_nms_thresh=0.7,
            new_det_thresh=0.99,
            hotstart_delay=3,
            hotstart_unmatch_thresh=1,
            hotstart_dup_thresh=1,
            suppress_unmatched_only_within_hotstart=False,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=100,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=64,
            recondition_every_nth_frame=64,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )

    if load_from_HF and checkpoint_path is None:
        checkpoint_path = download_ckpt_from_hf()
    if checkpoint_path is not None:
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, weights_only=True)
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        missing_keys, unexpected_keys = model.load_state_dict(
            ckpt, strict=strict_state_dict_loading
        )
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    model.to(device=device)
    return model

def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )

#######################
#######################

def ffmpeg_pipe(out_path, width, height, fps):

    # filter_complex = f"[0:v]v360=hequirect:fisheye:w=iw:h=ih,setpts=PTS-STARTPTS[left];[2:v]v360=hequirect:fisheyew=iw:h=ih,setpts=PTS-STARTPTS[right];[left][right]hstack=inputs=2[stacked];[1:v][stacked]scale2ref[mask][stacked_ref];[stacked_ref][mask]overlay=0:0,fps={fps},setpts=N/({fps}*TB),scale=w={width}:h=ih:flags=bicubic[v]"
    # filter_complex = '[0:v]v360=hequirect:fisheye:ih_fov=180:iv_fov=180:h_fov=180:v_fov=180:in_stereo=sbs:out_stereo=sbs[v]'
    #  '-filter_complex', '[0:v]v360=hequirect:fisheye:ih_fov=180:iv_fov=180:h_fov=180:v_fov=180:in_stereo=sbs:out_stereo=sbs[v]', '-map', '[v]',

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

    def pack_frame(self, sbs_rgb, mask_l = None, mask_r = None):

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
            blended[..., 2] += mask_1ch
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
        packed_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l] = blend_red_mask(packed_frame[y1_bl_l:y2_bl_l, x1_bl_l:x2_bl_l], q_tr_mask)
        return packed_frame

def loadsam3(segmentor="video", device="cuda", precision="fp16"):

    model_path = "assets/sam3.pt"
    if model_path is None:
        raise ValueError(f"Model file not found at '{model_path}'")

    if "fp16" in model.lower():
        precision = "fp16"

    if segmentor == "image":
        from model.sam3_image_processor import Sam3Processor
        model = build_sam3_image_model(
            device=device,
            eval_mode=True,
            checkpoint_path=model_path,
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False
        )
        processor = Sam3Processor(
            model=model,
            resolution=1008,
            confidence_threshold=0.3
        )

    elif segmentor == "video":
        model = build_sam3_video_model(
            device = device,
            checkpoint_path=model_path,
            load_from_HF = False,
            bpe_path = "assets/bpe_simple_vocab_16e6.txt.gz",
            has_presence_token = False,
            geo_encoder_use_img_cross_attn = False,
            strict_state_dict_loading = False,
            apply_temporal_disambiguation = False,
            compile = False,
        )
        processor = None

    else:
        raise ValueError(f"Unknown segmentor type: {segmentor}")

    if precision != 'fp32' and device == 'cpu':
        raise ValueError("fp16 and bf16 are not supported on cpu")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    device = {"cuda": torch.device("cuda"), "cpu": torch.device("cpu"), "mps": torch.device("mps")}[device]

    sam3_model = {
        "model": model,
        "processor": processor,
        "segmentor": segmentor,
        "device": device,
        "dtype": dtype,
    }
    return sam3_model

def Sam3Video(video_frames, prompt, frame_index=0, object_id=0, score_threshold_detection=0.65, new_det_thresh=0.9, propagation_direction="forward", start_frame_index=0, max_frames_to_track=-1, close_after_propagation=True,  keep_model_loaded=False, session_id=None, extra_config=None, positive_coords=None, negative_coords=None,
                bbox=None,):

    sam3_model = loadsam3(segmentor="video", device="cuda", precision="fp16")                

    predictor = sam3_model.get("model", None)
    device = sam3_model.get("device", torch.device("cpu"))
    dtype = sam3_model.get("dtype", torch.float32)
    segmentor = sam3_model.get("segmentor", None)
    B, H, W, _ = video_frames.shape

    if predictor is None or segmentor != "video":
        raise ValueError("Invalid SAM3 model. Please load a SAM3 model in 'video' mode")

    if frame_index > B - 1:
        frame_index = B - 1

    predictor.model.score_threshold_detection = score_threshold_detection
    predictor.model.new_det_thresh = new_det_thresh

    predictor.model.assoc_iou_thresh = 0.1
    predictor.model.det_nms_thresh = 0.1
    predictor.model.hotstart_delay = 15
    predictor.model.hotstart_unmatch_thresh = 8
    predictor.model.hotstart_dup_thresh = 8
    predictor.model.suppress_unmatched_only_within_hotstart = True
    predictor.model.min_trk_keep_alive = -1
    predictor.model.max_trk_keep_alive = 30
    predictor.model.init_trk_keep_alive = 30
    predictor.model.suppress_overlapping_based_on_recent_occlusion_threshold = 0.7
    predictor.model.suppress_det_close_to_boundary = False
    predictor.model.fill_hole_area = 16
    predictor.model.recondition_every_nth_frame = 16
    predictor.model.masklet_confirmation_enable = False
    predictor.model.decrease_trk_keep_alive_for_empty_masklets = False
    predictor.model.image_size = 1008

    if extra_config is not None and isinstance(extra_config, dict):
    
        for key, value in extra_config.items():
            if hasattr(predictor.model, key):
                setattr(predictor.model, key, value)

    video_pil = BigClassOfThings.tensor2pil(video_frames)
    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_pil,
            session_id=session_id
        )
    )

    session_id = response.get("session_id", None)
    if session_id is None:
        raise ValueError("Failed to start video prediction session")

    predictor.model.to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):

        pos_points, pos_count, pos_errors = parse_points(positive_coords, video_frames.shape)
        neg_points, neg_count, neg_errors = parse_points(negative_coords, video_frames.shape)
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
            bbox_coords, bbox_count = parse_bbox(bbox, video_frames.shape)
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

        output_masks = torch.zeros((B, H, W), dtype=torch.float32)
        processed_frames = 0

        object_outputs = {
            "obj_ids":None,
            "obj_masks":[]
        }
        object_masks_dict = {}

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
            if outputs:
                if "out_binary_masks" in outputs:
                    mask = outputs["out_binary_masks"]
                    if mask.shape[0] > 0:
                        object_masks_dict[frame_idx] = mask

                        merged_mask = np.any(mask, axis=0).astype(np.float32)
                        frame_masks = torch.from_numpy(merged_mask)
                        output_masks[frame_idx] = frame_masks
                    else:
                        object_masks_dict[frame_idx] = np.zeros((1, H, W), dtype=np.float32)
                else:
                    object_masks_dict[frame_idx] = np.zeros((1, H, W), dtype=np.float32)

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

    if len(object_masks_dict) > 0:
        max_num_objects = max(mask.shape[0] for mask in object_masks_dict.values())

        ordered_obj_masks = []
        padded_masks = []
        for frame_idx in range(B):
            if frame_idx in object_masks_dict:
                mask = object_masks_dict[frame_idx]
                num_objects = mask.shape[0]
                if num_objects < max_num_objects:
                    padding = np.zeros((max_num_objects - num_objects, H, W), dtype=np.float32)
                    padded_mask = np.concatenate([mask, padding], axis=0)
                    ordered_obj_masks.append(padded_mask)
                    padded_masks.append(torch.from_numpy(padded_mask))
                else:
                    ordered_obj_masks.append(mask)
                    padded_masks.append(torch.from_numpy(mask))
            else:
                empty_mask = np.zeros((max_num_objects, H, W), dtype=np.float32)
                ordered_obj_masks.append(empty_mask)
                padded_masks.append(torch.zeros((max_num_objects, H, W)))

        object_masks = torch.stack(padded_masks, dim=0)
        object_outputs["obj_masks"] = ordered_obj_masks
    else:
        object_masks = torch.zeros((B, 1, H, W))
        object_outputs["obj_masks"] = []

    return output_masks, session_id, object_outputs, object_masks

def process_frames(predictor, raft, frames_pil=None, frames_bgr=None, prompt_text=None, prior_mask=None, warp=False):

    chunk = len(frames_pil)
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
    tensors_rgb = []

    for f_bgr in frames_bgr:
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

                # flow[0] *= (w_mask / height)
                # flow[1] *= (h_mask / height)

                warped_logits = raft.warp_frame(prev_logits, flow)
                # dummy_point_inputs = {
                #     "point_coords": torch.zeros(1, 1, 2, device=device),
                #     "point_labels": -torch.ones(1, 1, dtype=torch.int32, device=device)
                # }

            else:
                warped_logits = prev_logits
                # dummy_point_inputs = {
                #     "point_coords": torch.zeros(batch_size, 1, 2, device=device),
                #     "point_labels": -torch.ones(batch_size, 1, dtype=torch.int32, device=device)
                # }

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

def process_videos(video_path, out_path, out_mask_path, 
prompt_text, batch_size, matte_size, warp, sbs=False, debug=None):  
    bbox=None

    things =  BigClassOfThings(device="cuda")
    predictor = build_sam3_video_predictor(
        has_presence_token=False,
        geo_encoder_use_img_cross_attn=False,
        strict_state_dict_loading=False,
        async_loading_frames=True,
        video_loader_type="cv2",
        offload_video_to_cpu = True,
        apply_temporal_disambiguation = False,
        compile = False,
    )

    raft =  raft_flow(device="cuda")
    frames, keyframes, width, height, duration, fps = metadata(video_path)
    cap = cv2.VideoCapture(video_path)
    writer = ffmpeg_pipe(out_path, width, height, fps) if out_path else None
    mask_writer = ffmpeg_pipe(out_mask_path, width, height, fps) if out_mask_path else None
    half_w = width // 2
    packer = AlphaPacker(scale=matte_size)

    frame_count = 0
    frames = frames if debug is None else debug
    pbar = tqdm(total=frames, desc="Processing .. beep.boop.bop.. beep.")

    last_mask_l = None
    last_mask_r = None

    while frame_count < frames:
        frames_bgr = []
      
        for _ in range(keyframes if batch_size is None else batch_size):
            ret, frame = cap.read()
            ret = ret if debug is None else debug
            if not ret: break
            frames_bgr.append(frame)
        if not frames_bgr:
            break
        
        chunk = len(frames_bgr) if debug is None else debug
        frames_l = [f[:, :half_w] for f in frames_bgr]
        frames_r = [f[:, half_w:] for f in frames_bgr]

        if sbs:
            sam_w = int(width)
            sam_h = int(height)            
            # sam_w = min(2048, width)
            # sam_h = min(1024, height)
            l_small = [cv2.resize(f, (sam_w, sam_h), interpolation=cv2.INTER_AREA) for f in frames_bgr] 
            r_small = None
        else:
            sam_w = 1024
            sam_h = 1024
            l_small = [cv2.resize(f, (sam_w, sam_h), interpolation=cv2.INTER_AREA) for f in frames_l]
            r_small = [cv2.resize(f, (sam_w, sam_h), interpolation=cv2.INTER_AREA) for f in frames_r]

        prior_l = last_mask_l if (last_mask_l is not None and np.sum(last_mask_l) > 0) else None
        pil_l = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in l_small]

        bboxes = []
        if bbox is not None and len(bbox) > 0:
            for i in bbox:
                if matte_size is not None:
                    x = i['x'] * matte_size
                    y = i['y'] * matte_size
                    w = i['w'] * matte_size
                    h = i['h'] * matte_size
                else:
                    x = i['x']
                    y = i['y']
                    w = i['w']
                    h = i['h']
                bboxes.append([x, y, x + w, y + h])

        masks_l, soft_l = process_frames(
            predictor=predictor,
            raft=raft,
            frames_pil=pil_l,
            frames_bgr=l_small,
            prompt_text=prompt_text,
            prior_mask=prior_l,
            warp=warp
            ) if l_small is not None else (None, None)

        if r_small is not None:
            prior_r = last_mask_r if (last_mask_r is not None and np.sum(last_mask_r) > 0) else None
            pil_r = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in r_small]

            masks_r, soft_r = process_frames(
                predictor=predictor,
                raft=raft,
                frames_pil=pil_r,
                frames_bgr=r_small,
                prompt_text=prompt_text,
                prior_mask=prior_r,
                warp=warp
                )
        else:
            masks_r, soft_r = None, None

        last_mask_l = masks_l[-1] if masks_l is not None else None
        last_mask_r = masks_r[-1] if masks_r is not None else None

        # masks_l, session_id, object_outputs, object_masks = Sam3Video(r_small, prompt=prompt_text, frame_index=0, object_id=0, score_threshold_detection=0.65, new_det_thresh=0.9, propagation_direction="forward", start_frame_index=0, max_frames_to_track=-1, close_after_propagation=True,  keep_model_loaded=False, session_id=None, extra_config=None, positive_coords=None, negative_coords=None, bbox=None,)

        masks = things.process_mask(masks_l, sensitivity=1.0, mask_blur=1, mask_offset=0, smooth=2, 
                    fill_holes=False, invert_output=False)
        # masks_l = things.apply_effects(masks_l, dilation = 0, feather_radius = 0.0, smooth_edges = 2)

        if masks_r is not None:
            masks_r = things.process_mask(masks_r, sensitivity=1.0, mask_blur=2, mask_offset=0, smooth=2.0, 
                    fill_holes=False, invert_output=False)            
            masks_r = things.apply_effects(masks_r, dilation = 0, feather_radius = 0.0, smooth_edges = 2)

        for i in range(chunk):

            if masks_r is not None:
                packed_frame = packer.pack_frame(frames_bgr[i], masks_l[i], masks_r[i])
                writer.stdin.write(packed_frame.astype(np.uint8).tobytes())

                full_mask_l = cv2.resize(masks_l[i], (half_w, height), interpolation=cv2.INTER_LINEAR)
                full_mask_r = cv2.resize(masks_r[i], (half_w, height), interpolation=cv2.INTER_LINEAR)
                red_sbs = np.zeros((height, width, 3), dtype=np.uint8)
                red_sbs[:, :half_w, 2] = full_mask_l
                red_sbs[:, half_w:, 2] = full_mask_r
                mask_writer.stdin.write(red_sbs.tobytes())
            else:
                full_sbs_mask = cv2.resize(masks[i], (width, height), interpolation=cv2.INTER_LINEAR)
                sbs_out = np.stack([full_sbs_mask, full_sbs_mask, full_sbs_mask], axis=-1).astype(np.uint8)

                if mask_writer is not None:
                    mask_writer.stdin.write(sbs_out.tobytes())

        frame_count += chunk
        pbar.update(chunk)
        
    cap.release()
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
        out_mask_path = os.path.join(output_dir, f"{base_name}_redmask.mp4")
        
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

if __name__ == "__main__":
    INPUT_FOLDER = "assets/video_segments"
    OUTPUT_FOLDER = "assets/matted_segments"
    
    process_directory(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        prompt_text="One girl",
        batch_size=50,
        matte_size=0.4,
        warp=False,
        sbs=True,
        debug=50,
    )
