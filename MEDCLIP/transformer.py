from torch import nn
import torch
from typing import Optional, Tuple, Union
import collections.abc
from itertools import repeat
import math
import torch
import torch.nn.functional as F
from typing import Optional






# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)






class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,                # BiomedCLIP ViT-B/16 uses 12 heads
        qkv_bias=True,
        scaled_cosine=False,
        scale_heads=False,
        logit_scale_max=math.log(1. / 0.01),
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()

        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # QKV projection (OpenCLIP style)
        self.in_proj_weight = nn.Parameter(torch.randn(3 * dim, dim) * self.scale)
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim)) if qkv_bias else None

        # logit scale (for cosine attention)
        if scaled_cosine:
            self.logit_scale = nn.Parameter(
                torch.log(10 * torch.ones((num_heads, 1, 1)))
            )
        else:
            self.logit_scale = None

        # optional head scaling
        if scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        # x: (L, N, C)
        L, N, C = x.shape

        # project QKV
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

        # reshape to (N * heads, L, head_dim)
        q = q.reshape(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, L, self.head_dim)
        k = k.reshape(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, L, self.head_dim)
        v = v.reshape(L, N, self.num_heads, self.head_dim).permute(1, 2, 0, 3).reshape(N * self.num_heads, L, self.head_dim)

        # cosine or scaled dot-product attention
        if self.logit_scale is not None:
            # cosine attention
            q_norm = F.normalize(q, dim=-1)
            k_norm = F.normalize(k, dim=-1)
            attn = torch.bmm(q_norm, k_norm.transpose(-1, -2))

            scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * scale
            attn = attn.view(N * self.num_heads, L, L)
        else:
            attn = torch.bmm(q * self.scale, k.transpose(-1, -2))

        # optional mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                m = torch.zeros_like(attn_mask, dtype=attn.dtype)
                m.masked_fill_(attn_mask, float('-inf'))
                attn_mask = m
            attn = attn + attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attention-output
        out = torch.bmm(attn, v)

        # per-head scaling
        if self.head_scale is not None:
            out = out.view(N, self.num_heads, L, self.head_dim) * self.head_scale
            out = out.view(N * self.num_heads, L, self.head_dim)

        # reshape back to (L, N, C)
        out = out.view(N, self.num_heads, L, self.head_dim).permute(2, 0, 1, 3).reshape(L, N, C)

        out = self.out_proj(out)
        out = self.out_drop(out)
        return out



class TimmModel(nn.Module):
    """
    Timm-based Vision Transformer for BiomedCLIP.
    
    Wraps timm's Vision Transformer with ImageNet pretraining.
    BiomedCLIP uses ViT-B/16 pretrained on ImageNet-21k.
    
    Reference: Zhang et al. "BiomedCLIP" (2023), Supplementary Table 3
    Shows that ImageNet-pretrained weights provide more stable downstream performance.
    """
    
    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        embed_dim: int = 768,
        image_size: Union[Tuple[int, int], int] = 224,
        pool: str = '',
        proj: str = 'linear',
        proj_bias: bool = False,
        drop: float = 0.,
        drop_path: Optional[float] = None,
        pretrained: bool = False,
        output_tokens: bool = False,
    ):
        """
        Initialize timm-based vision model.
        
        Args:
            model_name: Timm model name (e.g., 'vit_base_patch16_224')
            embed_dim: Output embedding dimension
            image_size: Input image size
            pool: Pooling type ('' for default, 'avg' for average pooling)
            proj: Projection type ('linear' or 'mlp')
            proj_bias: Whether to use bias in projection
            drop: Dropout rate for head
            drop_path: Stochastic depth rate
            pretrained: Load ImageNet pretrained weights
            output_tokens: Return patch tokens in addition to pooled output
        """
        super().__init__()
        
        import timm
        
        self.output_tokens = output_tokens
        self.image_size = to_2tuple(image_size)
        
        # Create timm model with optional pretraining
        # BiomedCLIP uses pretrained=True for ImageNet initialization
        self.trunk = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=pool if pool else '',
        )
        
        # Get feature dimension from trunk
        if hasattr(self.trunk, 'num_features'):
            feat_size = self.trunk.num_features
        elif hasattr(self.trunk, 'embed_dim'):
            feat_size = self.trunk.embed_dim
        else:
            feat_size = self.trunk.head.in_features if hasattr(self.trunk, 'head') else 768
        
        self.feat_size = feat_size
        
        # Apply dropout if specified
        if drop > 0.:
            self.head_drop = nn.Dropout(drop)
        else:
            self.head_drop = nn.Identity()
        
        # Create projection layer
        if proj == 'linear':
            self.proj = nn.Linear(feat_size, embed_dim, bias=proj_bias)
        elif proj == 'mlp':
            self.proj = nn.Sequential(
                nn.Linear(feat_size, feat_size, bias=proj_bias),
                nn.GELU(),
                nn.Linear(feat_size, embed_dim, bias=proj_bias),
            )
        else:
            raise ValueError(f"Unknown projection type: {proj}")
        
        # Initialize projection layer
        # BiomedCLIP: Pretrained trunk is kept, only projection is newly initialized
        self._init_projection()
    
    def _init_projection(self):
        """
        Initialize projection layer for BiomedCLIP.
        
        The Vision Transformer trunk uses ImageNet-pretrained weights.
        Only the projection layer needs initialization.
        
        Follows CLIP's initialization strategy for projection.
        """
        if isinstance(self.proj, nn.Sequential):
            # MLP projection
            for module in self.proj:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=self.feat_size ** -0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif isinstance(self.proj, nn.Linear):
            # Linear projection
            nn.init.normal_(self.proj.weight, std=self.feat_size ** -0.5)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
    
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """
        Lock (freeze) layers for fine-tuning.
        
        Args:
            unlocked_groups: Number of layer groups to keep unlocked from the end
            freeze_bn_stats: Whether to freeze batch norm statistics
        """
        # Freeze all trunk parameters
        for param in self.trunk.parameters():
            param.requires_grad = False
        
        if freeze_bn_stats:
            # Freeze batch norm layers
            for module in self.trunk.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        
        # Optionally unlock later layers
        if unlocked_groups > 0:
            # This depends on timm model structure
            # For ViT, unlock last N blocks
            if hasattr(self.trunk, 'blocks'):
                for block in self.trunk.blocks[-unlocked_groups:]:
                    for param in block.parameters():
                        param.requires_grad = True
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through vision encoder.
        
        Args:
            x: Input images [batch_size, 3, height, width]
        
        Returns:
            If output_tokens=False: Pooled features [batch_size, embed_dim]
            If output_tokens=True: (pooled features, patch tokens)
        """
        # Get features from trunk
        x = self.trunk.forward_features(x)
        
        # Handle different output formats from timm
        if isinstance(x, (tuple, list)):
            x = x[0]  # Take first element if multiple outputs
        
        pooled = x
        patch_tokens = None
        
        # Extract patch tokens if needed
        if self.output_tokens:
            if len(x.shape) == 3:  # [batch, num_patches, dim]
                pooled = x[:, 0]  # CLS token
                patch_tokens = [x[:, 1:]]  # Patch tokens (excluding CLS)
            else:  # [batch, dim]
                pooled = x
                patch_tokens = [x.unsqueeze(1)]  # Fake patch tokens
        else:
            if len(x.shape) == 3:
                pooled = x[:, 0]  # CLS token
        
        # Apply dropout and projection
        pooled = self.head_drop(pooled)
        pooled = self.proj(pooled)
        
        if self.output_tokens:
            return pooled, patch_tokens
        
        return pooled
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Enable/disable gradient checkpointing."""
        try:
            self.trunk.set_grad_checkpointing(enable)
        except AttributeError:
            pass  # Not all timm models support this


class HFTextEncoder(nn.Module):
    """
    HuggingFace Text Encoder for BiomedCLIP.
    
    Wraps a pretrained HuggingFace model (BiomedBERT) for use in CLIP.
    Uses pretrained weights from BiomedBERT instead of training from scratch.
    
    Reference: Zhang et al. "BiomedCLIP: a multimodal biomedical foundation model 
    pretrained from fifteen million scientific image-text pairs" (2023)
    Paper: https://arxiv.org/abs/2303.00915
    """
    
    def __init__(
        self,
        model_name: str,
        output_dim: int,
        proj_type: str = 'mlp',
        pooler_type: str = 'cls_last_hidden_state_pooler',
        pretrained: bool = True,
        output_tokens: bool = False,
    ):
        """
        Initialize HuggingFace text encoder.
        
        Args:
            model_name: HuggingFace model identifier (e.g., 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract')
            output_dim: Dimension of output embeddings (typically 512 for CLIP)
            proj_type: Type of projection layer ('mlp' or 'linear')
            pooler_type: Type of pooling to use ('cls_last_hidden_state_pooler', 'mean_pooler', etc.)
            pretrained: Whether to load pretrained weights
            output_tokens: Whether to output token embeddings in addition to pooled output
        """
        super().__init__()
        
        from transformers import AutoModel, AutoConfig
        
        self.output_tokens = output_tokens
        self.output_dim = output_dim
        self.pooler_type = pooler_type
        
        # Load pretrained model configuration
        if pretrained:
            self.transformer = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.transformer = AutoModel.from_config(config)
        
        # Get hidden size from the model
        self.hidden_size = self.transformer.config.hidden_size
        
        # Create projection layer
        if proj_type == 'mlp':
            # MLP projection: hidden -> hidden -> output_dim
            self.proj = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, output_dim)
            )
        elif proj_type == 'linear':
            # Simple linear projection: hidden -> output_dim
            self.proj = nn.Linear(self.hidden_size, output_dim)
        else:
            raise ValueError(f"Unknown projection type: {proj_type}")
        
        # Initialize projection layer
        self._init_projection()
    
    def _init_projection(self):
        """
        Initialize the projection layer.
        
        For BiomedCLIP: The projection layer is initialized while the transformer
        uses pretrained BiomedBERT weights.
        
        Initialization follows CLIP's approach for projection layers.
        """
        if isinstance(self.proj, nn.Sequential):
            # MLP projection initialization
            for module in self.proj:
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=self.hidden_size ** -0.5)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        elif isinstance(self.proj, nn.Linear):
            # Linear projection initialization
            nn.init.normal_(self.proj.weight, std=self.hidden_size ** -0.5)
            if self.proj.bias is not None:
                nn.init.zeros_(self.proj.bias)
    
    def pool_features(self, hidden_states, attention_mask):
        """
        Pool features from transformer outputs.
        
        Args:
            hidden_states: Transformer output hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            Pooled features [batch_size, hidden_size]
        """
        if self.pooler_type == 'cls_last_hidden_state_pooler' or self.pooler_type == 'cls':
            # Use CLS token (first token) from last hidden state
            pooled = hidden_states[:, 0]
        
        elif self.pooler_type == 'mean_pooler' or self.pooler_type == 'mean':
            # Mean pooling over all tokens (excluding padding)
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        
        elif self.pooler_type == 'max_pooler' or self.pooler_type == 'max':
            # Max pooling over all tokens
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states.clone()
            hidden_states[attention_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            pooled = torch.max(hidden_states, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooler type: {self.pooler_type}")
        
        return pooled
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the text encoder.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
        
        Returns:
            If output_tokens=False: Pooled text embeddings [batch_size, output_dim]
            If output_tokens=True: (pooled embeddings, token embeddings)
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()
        
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get hidden states
        hidden_states = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Pool features
        pooled = self.pool_features(hidden_states, attention_mask)
        
        # Project to output dimension
        pooled = self.proj(pooled)
        
        if self.output_tokens:
            # Also project token embeddings if requested
            batch_size, seq_len, hidden_size = hidden_states.shape
            token_embeddings = self.proj(hidden_states.view(-1, hidden_size))
            token_embeddings = token_embeddings.view(batch_size, seq_len, self.output_dim)
            return pooled, token_embeddings
        
        return pooled
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """Enable/disable gradient checkpointing for the transformer."""
        self.transformer.gradient_checkpointing_enable() if enable else self.transformer.gradient_checkpointing_disable()


