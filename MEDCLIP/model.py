import torch
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from torch import nn

@dataclass
class BiomedCLIPVisionCfg:
    layers: int = 12                     # BiomedCLIP ViT-B has 12 transformer blocks
    width: int = 768                     # embedding dimension
    head_width: int = 64                 # attention head width
    mlp_ratio: float = 4.0               # MLP expansion ratio
    patch_size: int = 16                 # 16x16 patches
    image_size: int = 224                # input image size
    ls_init_value: Optional[float] = None
    patch_dropout: float = 0.0           # BiomedCLIP typically disables patch dropout
    input_patchnorm: bool = False        # no dual patchnorm
    global_average_pool: bool = False    # CLS token used instead of global pooling
    attentional_pool: bool = False       # no attentional pooling
    n_queries: int = 256                 # default queries (unused here)
    attn_pooler_heads: int = 8           # default attention heads for pooler
    timm_model_name: str = 'vit_base_patch16_224'  # ViT-B/16
    timm_model_pretrained: bool = False  # pretrained weights not used
    timm_pool: str = ''                   # no extra pooling
    timm_proj: str = 'linear'             # linear projection for output
    timm_proj_bias: bool = False          # projection bias disabled
    timm_drop: float = 0.0                # head dropout
    timm_drop_path: Optional[float] = None
    output_tokens: bool = True            # output tokens for CLS embedding


    def __post_init__(self):
        print(f"BiomedCLIP Vision Config Initialized: layers={self.layers}, width={self.width}, patch_size={self.patch_size}, image_size={self.image_size}")



@dataclass
class BiomedCLIPTextCfg:
    context_length: int = 512
    vocab_size: int = 30522
    width: int = 768
    heads: int = 12
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    hf_tokenizer_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    hf_model_pretrained: bool = True
    proj: str = 'mlp'  # projection type
    pooler_type: str = 'cls_last_hidden_state_pooler'
    embed_cls: bool = False
    pad_id: int = 0
    output_tokens: bool = False
   


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype




class CustomTextCLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        if self.output_dict:
            return {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
        return image_features, text_features, self.logit_scale.exp()















def build_model_from_biomedclip_state_dict(
    state_dict: dict,
    quick_gelu=True,
    cast_dtype=torch.float16,
):
    print("Starting BiomedCLIP model build from state_dict...")

    # Detect ViT backbone from BiomedCLIP
    vit = any(k.startswith("visual.trunk") for k in state_dict.keys())
    print(f"Using ViT backbone: {vit}")

    if vit:
        vision_width = state_dict["visual.trunk.patch_embed.proj.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.trunk.blocks") and k.endswith(".attn.qkv.weight")]
        )
        vision_patch_size = state_dict["visual.trunk.patch_embed.proj.weight"].shape[-1]
        grid_size = round((state_dict["visual.trunk.pos_embed"].shape[1] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
        print(f"Vision config -> width: {vision_width}, layers: {vision_layers}, patch_size: {vision_patch_size}, image_size: {image_size}")
        print(f"visual.trunk.cls_token shape: {state_dict['visual.trunk.cls_token'].shape}")
        print(f"visual.trunk.pos_embed shape: {state_dict['visual.trunk.pos_embed'].shape}")
    else:
        raise NotImplementedError("Non-ViT BiomedCLIP not supported")

    # Text config
    text_width = state_dict["text.transformer.embeddings.word_embeddings.weight"].shape[1]
    context_length = state_dict["text.transformer.embeddings.position_ids"].shape[1]
    vocab_size = state_dict["text.transformer.embeddings.word_embeddings.weight"].shape[0]
    transformer_layers = len(
        [k for k in state_dict.keys() if k.startswith("text.transformer.encoder.layer") and k.endswith(".attention.self.query.weight")]
    )
    transformer_heads = text_width // 64
    print(f"Text config -> width: {text_width}, layers: {transformer_layers}, heads: {transformer_heads}, vocab_size: {vocab_size}, context_length: {context_length}")
    print(f"Example text embedding shape: {state_dict['text.transformer.embeddings.word_embeddings.weight'].shape}")

    # Embed dim from final projection
    embed_dim = state_dict["visual.head.proj.weight"].shape[0]
    print(f"Embed dim (from visual.head.proj.weight): {embed_dim}")

    # Create configs
    vision_cfg = BiomedCLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    print("\n BiomedCLIPVisionCfg object created.",vision_cfg,"\n")
    text_cfg =BiomedCLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=text_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    print("\n CLIPTextCfg objects created.",text_cfg,"\n")

    # Build CLIP model
    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,
        cast_dtype=cast_dtype,
    )
    print("CLIP model instance created.")

    # Remove unused keys
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            state_dict.pop(key)
            print(f"Removed unused key from state_dict: {key}")

    # Convert weights to fp16 if needed
    convert_weights_to_fp16(model)
    print("Converted model weights to fp16 (if applicable).")

    # Load weights
    model.load_state_dict(state_dict)
    print("BiomedCLIP model successfully loaded!")

    return model.eval()
