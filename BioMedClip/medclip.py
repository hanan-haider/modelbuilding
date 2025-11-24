import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
from .model import CLIP, CustomTextCLIP, convert_weights_to_lp, convert_to_custom_text_state_dict, resize_pos_embed, get_cast_dtype
from .openai import load_openai_model
import open_clip

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {} # directory (model_name: config) of model architecture configs
_MODEL_CKPT_PATHS = {'ViT-L-14-336': Path(__file__).parent / "ckpt/ViT-L-14-336px.pt"}

# BioMedCLIP specific configuration
_BIOMEDCLIP_CONFIG = {
    "model_cfg": {
        "embed_dim": 512,
        "vision_cfg": {
            "timm_model_name": "vit_base_patch16_224",
            "timm_model_pretrained": False,
            "timm_pool": "",
            "timm_proj": "linear",
            "image_size": 224
        },
        "text_cfg": {
            "hf_model_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            "hf_tokenizer_name": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            "hf_proj_type": "mlp",
            "hf_pooler_type": "cls_last_hidden_state_pooler",
            "context_length": 256
        }
    },
    "preprocess_cfg": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711]
    }
}

def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def _rescan_model_configs():
    global _MODEL_CONFIGS
    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIGS_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))
    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg
    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}

_rescan_model_configs() # initial populate of model config registry

def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())

def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None

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
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys

def create_model(
        model_name: str,
        img_size: int,
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
        adapter = False,
):
    model_name = model_name.replace('/', '-') # for callers using old naming with / in ViT names
    checkpoint_path = None
    model_cfg = None
    if isinstance(device, str):
        device = torch.device(device)

    # Handle BioMedCLIP specifically
    if model_name.lower().startswith('biomedclip'):
        logging.info(f'Loading BioMedCLIP model.')
        # Use the provided configuration
        biomed_config = _BIOMEDCLIP_CONFIG['model_cfg']
        cast_dtype = get_cast_dtype(precision)
        
        # Apply overrides
        if force_quick_gelu:
            biomed_config["quick_gelu"] = True
        if force_patch_dropout is not None:
            biomed_config["vision_cfg"]["patch_dropout"] = force_patch_dropout
        if force_image_size is not None:
            biomed_config["vision_cfg"]["image_size"] = force_image_size
        
        # Determine if custom text (BioMedCLIP uses HF text encoder, which is custom)
        custom_text = biomed_config.pop('custom_text', True) or force_custom_text  # Force custom for HF text
        
        if custom_text:
            model = CustomTextCLIP(**biomed_config, cast_dtype=cast_dtype)
        else:
            model = CLIP(**biomed_config, cast_dtype=cast_dtype)
        
        pretrained_loaded = False
        if pretrained:
            if pretrained.lower() == 'hf-hub' or pretrained.startswith('hf-hub:'):
                # Load from Hugging Face using open_clip
                hf_repo = pretrained.replace('hf-hub:', '') if pretrained.startswith('hf-hub:') else f'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
                logging.info(f'Loading BioMedCLIP weights from Hugging Face: {hf_repo}')
                model, _, _ = open_clip.create_model_and_transforms('hf-hub:' + hf_repo, pretrained='hf-hub:' + hf_repo)
                # Transfer state dict if needed, but since it's the same arch, we can use it directly
                # Note: open_clip model is already compatible; if your CLIP/CustomTextCLIP matches, load_state_dict
                # For simplicity, assuming your model class is compatible with open_clip's state dict
                state_dict = model.state_dict()
                resize_pos_embed(state_dict, model)  # Your model
                incompatible = model.load_state_dict(state_dict, strict=False)  # Load into your model
                logging.info(f'Incompatible keys: {incompatible}')
            elif pretrained.lower() == 'openai':
                # Fallback, but unlikely for BioMedCLIP
                raise RuntimeError(f'OpenAI pretrained not supported for BioMedCLIP.')
            else:
                # Assume local path
                checkpoint_path = Path(pretrained)
                if checkpoint_path.exists():
                    logging.info(f'Loading BioMedCLIP weights from local: {pretrained}')
                    load_checkpoint(model, checkpoint_path)
                else:
                    raise RuntimeError(f'Checkpoint not found: {pretrained}')
            pretrained_loaded = True
        
        if require_pretrained and not pretrained_loaded:
            raise RuntimeError(f'Pretrained weights required but not loaded for BioMedCLIP.')
        
        model.to(device=device)
        if precision in ("fp16", "bf16"):
            convert_weights_to_lp(model, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16)
        
        # Set preprocess metadata from config
        model.visual.image_mean = torch.tensor(_BIOMEDCLIP_CONFIG['preprocess_cfg']['mean'])
        model.visual.image_std = torch.tensor(_BIOMEDCLIP_CONFIG['preprocess_cfg']['std'])
        
        if output_dict and hasattr(model, "output_dict"):
            model.output_dict = True
        if jit:
            model = torch.jit.script(model)
        
        return model
