


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
    
    # Check if vision trunk exists
    vit = "visual.trunk" in list(state_dict.keys())[0]  # BiomedCLIP uses visual.trunk
    print(f"Using ViT backbone: {vit}")

    # Vision config
    if vit:
        vision_width = state_dict["visual.trunk.patch_embed.proj.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.trunk.blocks") and k.endswith(".attn.qkv.weight")])
        vision_patch_size = state_dict["visual.trunk.patch_embed.proj.weight"].shape[-1]
        grid_size = round((state_dict["visual.trunk.pos_embed"].shape[0] - 1) ** 0.5)
        image_size = vision_patch_size * grid_size
        print(f"Vision config -> width: {vision_width}, layers: {vision_layers}, patch_size: {vision_patch_size}, image_size: {image_size}")
    else:
        raise NotImplementedError("Non-ViT BiomedCLIP not supported")

    # Text config
    context_length = state_dict["text.transformer.embeddings.position_ids"].shape[0] if "text.transformer.embeddings.position_ids" in state_dict else 256
    vocab_size = state_dict["text.transformer.embeddings.word_embeddings.weight"].shape[0]
    transformer_width = state_dict["text.transformer.embeddings.LayerNorm.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len([k for k in state_dict.keys() if k.startswith("text.transformer.encoder.layer") and k.endswith(".attention.self.query.weight")])
    print(f"Text config -> context_length: {context_length}, vocab_size: {vocab_size}, width: {transformer_width}, heads: {transformer_heads}, layers: {transformer_layers}")

    # Build configs
    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )

    embed_dim = state_dict["text.proj.0.weight"].shape[0]  # BiomedCLIP uses text.proj.* for final projection
    print(f"Embedding dimension: {embed_dim}")

    model = CLIP(
        embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,
        cast_dtype=cast_dtype,
    )
    print("CLIP model instance created.")

    # Remove non-model keys
    for key in ["logit_scale", "text.transformer.embeddings.position_ids"]:
        if key in state_dict:
            state_dict.pop(key)
            print(f"Removed key from state_dict: {key}")

    # Convert weights if needed
    convert_weights_to_fp16(model)  # if you want FP16
    print("Converted model weights to FP16 (if applicable).")

    # Load the state dict
    model.load_state_dict(state_dict, strict=False)
    print("Loaded state_dict into model successfully.")
    
    return model.eval()
