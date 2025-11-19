import torch


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype















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
    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=text_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    print("CLIPVisionCfg and CLIPTextCfg objects created.")

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
