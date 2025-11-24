import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import numpy as np

#from model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict, resize_pos_embed, get_cast_dtype
#from .openai import load_openai_model

# === UPDATE: Add your custom BioMedCLIP paths ===
_MODEL_CONFIG_PATHS = [
    Path(__file__).parent / "model_configs/",
    Path("/kaggle/working/modelbuilding/BioMedClip/model_configs/")  # Your BioMedCLIP config
]

_MODEL_CKPT_PATHS = {
   # 'ViT-L-14-336': Path(__file__).parent / "ckpt/ViT-L-14-336px.pt",
    'BiomedCLIP-PubMedBERT-ViT-B-16': Path("/kaggle/working/modelbuilding/ckpt/open_clip_pytorch_model.bin")  # Your checkpoint
}

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def _rescan_model_configs():
    global _MODEL_CONFIGS
    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        p = Path(config_path)
        if p.is_file() and p.suffix in config_ext:
            config_files.append(p)
        elif p.is_dir():
            for ext in config_ext:
                config_files.extend(p.glob(f'*{ext}'))

     for cf in config_files:
        with open(cf, 'r') as f:
            raw_cfg = json.load(f)
            # Support both old format and new wrapped "model_cfg"
            if "model_cfg" in raw_cfg:
                model_cfg = raw_cfg["model_cfg"]
                # Also store preprocess config for later use
                _MODEL_CONFIGS[cf.stem] = {
                    "model_cfg": model_cfg,
                    "preprocess_cfg": raw_cfg.get("preprocess_cfg", {})
                }
            elif all(a in raw_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = {
                    "model_cfg": raw_cfg,
                    "preprocess_cfg": {}
                }
            else:
                continue

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}

_rescan_model_configs()  # Now includes BioMedCLIP

def list_models():
    return list(_MODEL_CONFIGS.keys())

def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    return None

def get_preprocess_config(model_name):
    """Extract mean/std for image normalization"""
    cfg = _MODEL_CONFIGS.get(model_name, {})
    preprocess_cfg = cfg.get("preprocess_cfg", {})
    mean = preprocess_cfg.get("mean", [0.48145466, 0.4578275, 0.40821073])
    std = preprocess_cfg.get("std", [0.26862954, 0.26130258, 0.27577711])
    return mean, std

def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

def create_model(
        model_name: str,
        img_size: int = None,  # Allow override
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        adapter=False,
):
    model_name = model_name.replace('/', '-')
    if isinstance(device, str):
        device = torch.device(device)

    config_entry = get_model_config(model_name)
    if config_entry is None:
        raise RuntimeError(f'Model config for {model_name} not found.')

    model_cfg_wrapper = config_entry
    model_cfg = model_cfg_wrapper["model_cfg"]
    preprocess_cfg = model_cfg_wrapper["preprocess_cfg"]

    # Override image size if requested
    if img_size is not None:
        model_cfg["vision_cfg"]["image_size"] = img_size
    if force_image_size is not None:
        model_cfg["vision_cfg"]["image_size"] = force_image_size

    if force_quick_gelu:
        model_cfg["quick_gelu"] = True
    if force_patch_dropout is not None:
        model_cfg["vision_cfg"]["patch_dropout"] = force_patch_dropout

    cast_dtype = get_cast_dtype(precision)

    # === Handle custom text (HuggingFace) models like PubMedBERT ===
    custom_text = (
        "hf_model_name" in model_cfg.get("text_cfg", {}) or
        model_cfg.pop("custom_text", False) or
        force_custom_text
    )

    if custom_text:
        model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
    else:
        model = CLIP(**model_cfg, cast_dtype=cast_dtype)

    # === Load pretrained weights ===
    pretrained_loaded = False
    if pretrained == "openai":
        logging.info(f"Loading OpenAI pretrained {model_name}...")
        # Your existing OpenAI logic here (unchanged)
        # ... [keep your original OpenAI block if needed]
        pass
    elif pretrained or pretrained is None:
        checkpoint_path = _MODEL_CKPT_PATHS.get(model_name)
        if checkpoint_path and checkpoint_path.exists():
            print(f"Loading pretrained {model_name} from {checkpoint_path}")
            load_checkpoint(model, checkpoint_path)
            pretrained_loaded = True
        elif require_pretrained:
            raise RuntimeError(f"Pretrained weights not found for {model_name}")

    model.to(device=device)

    if precision in ("fp16", "bf16"):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        convert_weights_to_lp(model, dtype=dtype)

    # === Set correct image normalization (critical for BioMedCLIP!) ===
    mean, std = get_preprocess_config(model_name)
    model.visual.image_mean = mean
    model.visual.image_std = std

    # Optional: expose as attributes for preprocessing
    model.preprocess_mean = torch.tensor(mean).to(device)
    model.preprocess_std = torch.tensor(std).to(device)

    if output_dict and hasattr(model, "output_dict"):
        model.output_dict = True

    if jit:
        model = torch.jit.script(model)

    return model