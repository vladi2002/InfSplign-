"""
Modular Spatial Guidance for SD3

A refactored version with:
1. Injectable prompts via PromptConfig
2. Pluggable loss strategies (point-based vs shape-based)
3. Clean separation of metrics extraction and loss computation

This design allows easy experimentation with:
- Different spatial loss formulations (centroid, GMM, percentile-bbox, etc.)
- Different presence/balance losses
- Custom metric representations
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any, NamedTuple, Protocol, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum
import re

from bayesian_gmm_strategy import BayesianGMMLossStrategy

# ============================================================================
#                              TYPE DEFINITIONS
# ============================================================================

class GuidanceTarget(Enum):
    LATENTS = "latents"
    VELOCITY = "velocity"


# ============================================================================
#                           SPATIAL CONSTRAINTS
# ============================================================================

@dataclass(frozen=True)
class SpatialConstraint:
    """Defines a spatial relationship between two objects."""
    object_A: str
    object_B: str
    relation: str  # "left", "right", "above", "below"
    
    @property
    def axis(self) -> str:
        return "x" if self.relation in ("right", "left") else "y"
    
    @property
    def sign(self) -> int:
        """Returns +1 for right/below, -1 for left/above."""
        return +1 if self.relation in ("right", "below") else -1
    
    def get_target_delta(self) -> Tuple[str, int]:
        return (self.axis, self.sign)
    
    def __str__(self) -> str:
        return f"{self.object_A} {self.relation} of {self.object_B}"


# ============================================================================
#                             PROMPT CONFIG
# ============================================================================

# Mapping from VISOR relation strings to internal relation names
VISOR_RELATION_MAP = {
    "to the left of": "left",
    "to the right of": "right",
    "above": "above",
    "below": "below",
}


@dataclass
class PromptConfig:
    """
    Encapsulates all prompt-related settings for generation.
    
    This makes the prompt injectable and separates prompt concerns
    from guidance/loss configuration.
    """
    prompt: str
    constraint: SpatialConstraint
    negative_prompt: str = ""
    prompt_2: Optional[str] = None  # For SD3's second text encoder (OpenCLIP ViT-bigG)
    prompt_3: Optional[str] = None  # For SD3's third text encoder (T5-XXL)
    unique_id: Optional[int] = None  # For tracking (e.g., VISOR dataset ID)
    
    @classmethod
    def simple(cls, prompt: str, object_A: str, object_B: str, relation: str) -> "PromptConfig":
        """Convenience factory for simple prompts."""
        return cls(
            prompt=prompt,
            constraint=SpatialConstraint(object_A, object_B, relation)
        )
    
    @classmethod
    def from_visor(cls, entry: Dict[str, Any]) -> "PromptConfig":
        """
        Create PromptConfig from a VISOR dataset entry.
        
        Expected format:
        {
            "unique_id": 16220,
            "num_objects": 2,
            "obj_1_attributes": ["backpack"],
            "obj_2_attributes": ["tie"],
            "rel_type": "to the left of",
            "text": "a backpack to the left of a tie"
        }
        """
        # Extract object names (take first attribute, which is the object name)
        object_A = entry["obj_1_attributes"][0]
        object_B = entry["obj_2_attributes"][0]
        
        # Map relation type
        rel_type = entry["rel_type"]
        relation = VISOR_RELATION_MAP.get(rel_type)
        if relation is None:
            raise ValueError(f"Unknown relation type: {rel_type}")
        
        return cls(
            prompt=entry["text"],
            constraint=SpatialConstraint(object_A, object_B, relation),
            unique_id=entry.get("unique_id"),
        )
    
    @classmethod
    def load_visor_dataset(cls, json_path: str) -> List["PromptConfig"]:
        """Load all entries from a VISOR JSON file."""
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        return [cls.from_visor(entry) for entry in data]
    
    def validate(self) -> None:
        """Validate that constraint objects appear in prompt."""
        prompt_lower = self.prompt.lower()
        if self.constraint.object_A.lower() not in prompt_lower:
            raise ValueError(f"Object A '{self.constraint.object_A}' not found in prompt")
        if self.constraint.object_B.lower() not in prompt_lower:
            raise ValueError(f"Object B '{self.constraint.object_B}' not found in prompt")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert back to VISOR-like dict format."""
        inverse_map = {v: k for k, v in VISOR_RELATION_MAP.items()}
        return {
            "unique_id": self.unique_id,
            "num_objects": 2,
            "obj_1_attributes": [self.constraint.object_A],
            "obj_2_attributes": [self.constraint.object_B],
            "rel_type": inverse_map.get(self.constraint.relation, self.constraint.relation),
            "text": self.prompt,
        }


# ============================================================================
#                         METRICS REPRESENTATIONS
# ============================================================================

class ObjectMetrics(NamedTuple):
    """
    Point-based object metrics (centroid representation).
    Used by CentroidLossStrategy.
    """
    cx: Tensor          # Centroid x coordinate
    cy: Tensor          # Centroid y coordinate
    variance: Tensor    # Spatial variance around centroid
    entropy: Optional[Tensor] = None  # Attention entropy (optional)


@dataclass
class GMMMetrics:
    """
    TODO: Gaussian Mixture Model based metrics (shape representation).
    Will be used by GMMShapeLossStrategy once implemented.
    
    This will represent objects as mixtures of Gaussians, which better
    captures complex shapes like hollow objects (bicycle wheels) or
    elongated objects (giraffes).
    
    Planned fields:
    - means: Tensor           # [K, 2] - K component means
    - covariances: Tensor     # [K, 2, 2] - K component covariances  
    - weights: Tensor         # [K] - mixture weights
    - attention_map: Tensor   # [H, W] - raw attention for fallback
    """
    pass  # Placeholder - implement when GMMShapeLossStrategy is ready


@dataclass  
class BBoxMetrics:
    """
    TODO: Bounding box based metrics (percentile approach).
    An alternative shape representation using axis-aligned boxes.
    
    Planned fields:
    - x_min, x_max, y_min, y_max: Tensor - box coordinates
    - attention_map: Tensor - raw attention
    
    Properties to implement:
    - center: Tuple[Tensor, Tensor]
    - width, height: Tensor
    """
    pass  # Placeholder - implement if BBox strategy is needed


# ============================================================================
#                          LOSS STRATEGY PROTOCOL
# ============================================================================

@runtime_checkable
class LossStrategy(Protocol):
    """
    Protocol for spatial guidance loss strategies.
    
    Implementations must provide:
    1. extract_metrics(): Convert attention maps to object representations
    2. compute_loss(): Compute guidance loss from metrics
    3. name property: Human-readable name for logging
    
    Note: Uses structural subtyping - classes don't need to explicitly
    inherit from this Protocol, they just need matching method signatures.
    """
    
    @property
    def name(self) -> str:
        """Human-readable name for logging."""
        ...
    
    def extract_metrics(
        self,
        attention: Tensor,
        token_indices_A: List[int],
        token_indices_B: List[int],
        threshold_percentile: float,
        threshold_steepness: float,
    ) -> Tuple[Any, Any]:
        """Extract metrics for both objects from attention map."""
        ...
    
    def compute_loss(
        self,
        metrics_A: Any,
        metrics_B: Any,
        constraint: SpatialConstraint,
        config: "GuidanceConfig",
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute total loss and return diagnostics dict."""
        ...


# ============================================================================
#                         COORDINATE GRID CACHE
# ============================================================================

class CoordinateGridCache:
    """Caches coordinate grids for efficient reuse."""
    _cache: Dict[Tuple[int, int, str], Tuple[Tensor, Tensor]] = {}
    
    @classmethod
    def get(cls, H: int, W: int, device) -> Tuple[Tensor, Tensor]:
        key = (H, W, str(device))
        if key not in cls._cache:
            y, x = torch.meshgrid(
                torch.linspace(0, 1, H, device=device),
                torch.linspace(0, 1, W, device=device),
                indexing='ij'
            )
            cls._cache[key] = (x, y)
        return cls._cache[key]


# ============================================================================
#                      SPATIAL OPERATIONS (Shared Utilities)
# ============================================================================

class SpatialOps:
    """Shared spatial operations used by multiple loss strategies."""
    
    @staticmethod
    def apply_soft_threshold(attention: Tensor, percentile: float, steepness: float = 50.0) -> Tensor:
        """Apply differentiable soft thresholding to attention map."""
        if percentile <= 0:
            return attention
        
        squeeze = attention.dim() == 2
        if squeeze:
            attention = attention.unsqueeze(0)
        
        B = attention.shape[0]
        flat = attention.view(B, -1)
        threshold = torch.quantile(flat.float(), percentile, dim=1, keepdim=True)
        threshold = threshold.to(attention.dtype).view(B, 1, 1)
        mask = torch.sigmoid(steepness * (attention - threshold))
        result = attention * mask
        
        return result.squeeze(0) if squeeze else result
    
    @staticmethod
    def compute_entropy(attention: Tensor) -> Tensor:
        """Compute entropy of attention distribution (lower = more focused)."""
        flat = attention.view(attention.shape[0], -1)
        p = flat / (flat.sum(dim=-1, keepdim=True) + 1e-8)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1)
        return entropy
    
    @staticmethod
    def extract_object_attention(
        attention: Tensor,
        token_indices: List[int],
    ) -> Tensor:
        """Extract and average attention for specific tokens."""
        if attention.dim() == 2:
            attention = attention.unsqueeze(0)
        
        B, num_spatial, num_tokens = attention.shape
        H = W = int(num_spatial ** 0.5)
        
        attn = attention[:, :, token_indices].mean(dim=-1).view(B, H, W)
        return attn


# ============================================================================
#                      CENTROID LOSS STRATEGY (Point-Based)
# ============================================================================

class CentroidLossStrategy:
    """
    Original InfSplign-style centroid-based loss.
    
    Represents objects as single points (centroids) computed as the
    weighted average of attention coordinates.
    
    Limitations:
    - Fails for hollow objects (bicycle wheels) where centroid is outside object
    - Fails for elongated objects where single point doesn't capture extent
    - Can't distinguish "below" from "on the bottom of"
    """
    
    def __init__(self, compute_entropy: bool = False):
        self.compute_entropy = compute_entropy
    
    @property
    def name(self) -> str:
        return "centroid"
    
    def extract_metrics(
        self,
        attention: Tensor,
        token_indices_A: List[int],
        token_indices_B: List[int],
        threshold_percentile: float,
        threshold_steepness: float,
    ) -> Tuple[ObjectMetrics, ObjectMetrics]:
        """Extract centroid-based metrics for both objects."""
        
        if attention.dim() == 2:
            attention = attention.unsqueeze(0)
        
        B, num_spatial, num_tokens = attention.shape
        device = attention.device
        H = W = int(num_spatial ** 0.5)
        
        x_coords, y_coords = CoordinateGridCache.get(H, W, device)
        
        # Extract attention for both objects
        attn_A = attention[:, :, token_indices_A].mean(dim=-1).view(B, H, W)
        attn_B = attention[:, :, token_indices_B].mean(dim=-1).view(B, H, W)
        
        # Compute entropy before thresholding (on raw attention)
        entropy_A = entropy_B = None
        if self.compute_entropy:
            entropy_A = SpatialOps.compute_entropy(attn_A)
            entropy_B = SpatialOps.compute_entropy(attn_B)
        
        # Apply thresholding
        if threshold_percentile > 0:
            attn_A = SpatialOps.apply_soft_threshold(attn_A, threshold_percentile, threshold_steepness)
            attn_B = SpatialOps.apply_soft_threshold(attn_B, threshold_percentile, threshold_steepness)
        
        # Normalize
        sum_A = attn_A.sum(dim=(-2, -1), keepdim=True) + 1e-8
        sum_B = attn_B.sum(dim=(-2, -1), keepdim=True) + 1e-8
        norm_A = attn_A / sum_A
        norm_B = attn_B / sum_B
        
        # Centroids
        cx_A = (norm_A * x_coords).sum(dim=(-2, -1))
        cy_A = (norm_A * y_coords).sum(dim=(-2, -1))
        cx_B = (norm_B * x_coords).sum(dim=(-2, -1))
        cy_B = (norm_B * y_coords).sum(dim=(-2, -1))
        
        # Variances
        cx_A_exp, cy_A_exp = cx_A.view(B, 1, 1), cy_A.view(B, 1, 1)
        cx_B_exp, cy_B_exp = cx_B.view(B, 1, 1), cy_B.view(B, 1, 1)
        
        dist_sq_A = (x_coords - cx_A_exp) ** 2 + (y_coords - cy_A_exp) ** 2
        dist_sq_B = (x_coords - cx_B_exp) ** 2 + (y_coords - cy_B_exp) ** 2
        
        var_A = (norm_A * dist_sq_A).sum(dim=(-2, -1))
        var_B = (norm_B * dist_sq_B).sum(dim=(-2, -1))
        
        return (
            ObjectMetrics(cx_A, cy_A, var_A, entropy_A),
            ObjectMetrics(cx_B, cy_B, var_B, entropy_B)
        )
    
    def compute_loss(
        self,
        metrics_A: ObjectMetrics,
        metrics_B: ObjectMetrics,
        constraint: SpatialConstraint,
        config: "GuidanceConfig",
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute compound loss from centroid metrics."""
        
        # Spatial loss: encourage correct relative positioning
        if constraint.axis == "x":
            delta = constraint.sign * (metrics_A.cx - metrics_B.cx)
        else:
            delta = constraint.sign * (metrics_A.cy - metrics_B.cy)
        L_spatial = F.gelu(config.alpha * (config.margin - delta)).mean()
        
        # Presence loss: low variance = good localization
        if config.use_entropy_presence and metrics_A.entropy is not None:
            L_presence = (metrics_A.entropy + metrics_B.entropy).mean()
        else:
            L_presence = (metrics_A.variance + metrics_B.variance).mean()
        
        # Balance loss: similar variance between objects
        L_balance = torch.abs(metrics_A.variance - metrics_B.variance).mean()
        
        # Compound loss
        L_total = (
            config.lambda_spatial * L_spatial +
            config.lambda_presence * L_presence +
            config.lambda_balance * L_balance
        )
        
        # Compute raw delta for diagnostics
        raw_delta = (metrics_A.cx - metrics_B.cx).mean().item() if constraint.axis == "x" \
                    else (metrics_A.cy - metrics_B.cy).mean().item()
        
        diagnostics = {
            'L_spatial': L_spatial.item(),
            'L_presence': L_presence.item(),
            'L_balance': L_balance.item(),
            'L_total': L_total.item(),
            'cx_A': metrics_A.cx.mean().item(),
            'cy_A': metrics_A.cy.mean().item(),
            'cx_B': metrics_B.cx.mean().item(),
            'cy_B': metrics_B.cy.mean().item(),
            'var_A': metrics_A.variance.mean().item(),
            'var_B': metrics_B.variance.mean().item(),
            'delta': raw_delta,
            'strategy': self.name,
        }
        
        return L_total, diagnostics


# ============================================================================
#                     GMM SHAPE LOSS STRATEGY (Shape-Based)
# ============================================================================

# TODO: Implement GMMShapeLossStrategy
#
# This will be a shape-aware loss strategy using Gaussian Mixture Models.
# 
# Motivation:
# - Centroid-based loss fails for hollow objects (bicycle wheels) where 
#   the centroid falls outside the actual object
# - Centroid-based loss fails for elongated objects (giraffes) where a 
#   single point doesn't capture the object's extent
# - Can't distinguish semantic relationships like "below" vs "on the bottom of"
#
# Approach:
# - Represent objects as mixtures of Gaussians fitted to attention maps
# - Use differentiable soft-EM to maintain gradient flow
# - Compare distributions rather than single points
#
# Required components:
# 1. GMMMetrics dataclass (defined above) with means, covariances, weights
# 2. _fit_gmm() method using differentiable soft-EM
# 3. extract_metrics() returning GMMMetrics for both objects
# 4. compute_loss() comparing GMM distributions
#
# Example skeleton:
#
# class GMMShapeLossStrategy:
#     def __init__(self, num_components: int = 3, em_iterations: int = 10):
#         self.num_components = num_components
#         self.em_iterations = em_iterations
#     
#     @property
#     def name(self) -> str:
#         return f"gmm-{self.num_components}"
#     
#     def extract_metrics(self, attention, token_indices_A, token_indices_B, 
#                         threshold_percentile, threshold_steepness) -> Tuple[GMMMetrics, GMMMetrics]:
#         # TODO: Fit GMM to attention maps
#         raise NotImplementedError
#     
#     def compute_loss(self, metrics_A, metrics_B, constraint, config) -> Tuple[Tensor, Dict]:
#         # TODO: Compare GMM distributions for spatial loss
#         # Options to explore:
#         # - Weighted centroid comparison (simple)
#         # - All component-pair comparisons (thorough)
#         # - Wasserstein distance between distributions
#         # - KL divergence based approaches
#         raise NotImplementedError


# ============================================================================
#                          GUIDANCE CONFIGURATION
# ============================================================================

@dataclass
class GuidanceConfig:
    """
    Configuration for spatial guidance.
    
    Combines loss hyperparameters with guidance settings.
    The loss_strategy field allows plugging in different loss implementations.
    """
    
    # === Loss Strategy (injectable) ===
    loss_strategy: LossStrategy = field(default_factory=CentroidLossStrategy)
    
    # === Loss Hyperparameters ===
    lambda_spatial: float = 0.5
    lambda_presence: float = 1.0
    lambda_balance: float = 1.0
    alpha: float = 1.5
    margin: float = 0.25
    
    # === Threshold Settings ===
    threshold_percentile: float = 0.5
    threshold_steepness: float = 50.0
    
    # === Guidance Settings ===
    guidance_strength: float = 30.0
    guidance_timestep_fraction: float = 0.3
    guidance_target: GuidanceTarget = GuidanceTarget.LATENTS
    velocity_guidance_scale: float = 1.0
    
    # === Early Exit ===
    enable_early_exit: bool = True
    early_exit_loss_threshold: float = 0.08
    early_exit_margin_buffer: float = 1.3
    min_guidance_steps: int = 3
    
    # === Adaptive Blocks ===
    enable_adaptive_blocks: bool = True
    measure_blocks: Tuple[int, ...] = (4, 5, 6, 7)
    
    # === Entropy Option (for CentroidLossStrategy) ===
    use_entropy_presence: bool = False
    
    # === Performance ===
    use_flash_attention: bool = True
    reuse_velocity: bool = True


# ============================================================================
#                         ADAPTIVE BLOCK SELECTION
# ============================================================================

def get_adaptive_blocks(timestep_fraction: float) -> Tuple[int, ...]:
    """
    Select measurement blocks based on where we are in denoising.
    
    Based on SD3 attention evolution:
    - Early (t≈1): Blocks 4-7, layout still forming
    - Mid (t≈0.5): Blocks 7-10, clear object localization  
    - Late (t≈0): Blocks 10-13, fine details (less useful for layout)
    """
    if timestep_fraction > 0.7:
        return (4, 5, 6, 7)
    elif timestep_fraction > 0.35:
        return (4, 5, 6, 7)
    else:
        return (8, 9)


# ============================================================================
#                            EARLY EXIT LOGIC
# ============================================================================

def should_continue_guidance(
    diagnostics: Dict[str, float],
    constraint: SpatialConstraint,
    config: GuidanceConfig,
    step: int
) -> Tuple[bool, str]:
    """Determine if guidance should continue."""
    if step < config.min_guidance_steps:
        return True, "min_steps"
    
    if not config.enable_early_exit:
        return True, "early_exit_disabled"
    
    loss_ok = diagnostics['L_total'] < config.early_exit_loss_threshold
    required_delta = config.margin * config.early_exit_margin_buffer
    delta_ok = (constraint.sign * diagnostics['delta']) > required_delta
    
    if loss_ok and delta_ok:
        return False, f"converged (L={diagnostics['L_total']:.3f}, Δ={diagnostics['delta']:.3f})"
    
    return True, "not_converged"


# ============================================================================
#                         ATTENTION PROCESSOR
# ============================================================================

class SpatialGuidanceAttnProcessor:
    """
    Attention processor that captures I2T attention for spatial guidance.
    Follows diffusers AttnProcessor pattern for clean integration.
    """
    
    def __init__(self, block_idx: int, capture: bool = True, use_flash: bool = True):
        self.block_idx = block_idx
        self.capture = capture
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        self.stored_i2t = None
    
    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        attention_mask: Tensor = None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Compute joint attention over image and text tokens, capturing I2T attention.
        
        This replaces the default attention forward pass in SD3's transformer blocks.
        SD3 uses joint attention where image and text tokens attend to each other
        in a single attention operation. We intercept this to extract the 
        Image-to-Text (I2T) attention weights, which tell us where in the image
        each text token is "looking".
        
        Args:
            attn: The attention module from the transformer block. Contains:
                - to_q, to_k, to_v: Linear projections for image tokens
                - add_q_proj, add_k_proj, add_v_proj: Linear projections for text tokens (if separate)
                - norm_q, norm_k: QK normalization for image tokens
                - norm_added_q, norm_added_k: QK normalization for text tokens
                - to_out: Output projection layers
                - to_add_out: Output projection for text (if separate)
                - heads: Number of attention heads
                
            hidden_states: Image token embeddings [B, num_image_tokens, dim]
                For 512x512 with 8x downscaling: num_image_tokens = 64*64 = 4096
                For 1024x1024: num_image_tokens = 128*128 = 16384
                
            encoder_hidden_states: Text token embeddings [B, num_text_tokens, dim]
                Typically 77 or 154 tokens depending on prompt length.
                If None, performs self-attention on image tokens only.
                
            attention_mask: Optional attention mask [B, 1, seq_len, seq_len]
                Applied additively to attention logits before softmax.
                
        Returns:
            If encoder_hidden_states is provided (joint attention):
                Tuple of (image_output, text_output) both [B, num_tokens, dim]
            If encoder_hidden_states is None (self-attention):
                Single tensor of image_output [B, num_image_tokens, dim]
                
        Side Effects:
            If self.capture is True and encoder_hidden_states is provided,
            stores I2T attention in self.stored_i2t with shape [B, num_image, num_text].
            This is the attention from image tokens (queries) to text tokens (keys),
            averaged across all attention heads.
        """
        
        batch_size = hidden_states.shape[0]
        
        if encoder_hidden_states is not None:
            num_text = encoder_hidden_states.shape[1]
            num_image = hidden_states.shape[1]
            
            # Project Q/K/V
            query_img = attn.to_q(hidden_states)
            key_img = attn.to_k(hidden_states)
            value_img = attn.to_v(hidden_states)
            
            if hasattr(attn, 'add_q_proj') and attn.add_q_proj is not None:
                query_txt = attn.add_q_proj(encoder_hidden_states)
                key_txt = attn.add_k_proj(encoder_hidden_states)
                value_txt = attn.add_v_proj(encoder_hidden_states)
            else:
                query_txt = attn.to_q(encoder_hidden_states)
                key_txt = attn.to_k(encoder_hidden_states)
                value_txt = attn.to_v(encoder_hidden_states)
            
            query = torch.cat([query_txt, query_img], dim=1)
            key = torch.cat([key_txt, key_img], dim=1)
            value = torch.cat([value_txt, value_img], dim=1)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            num_text = 0
            num_image = hidden_states.shape[1]
        
        # Reshape for multi-head
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Apply QK normalization
        if encoder_hidden_states is not None:
            q_txt, q_img = query[:, :, :num_text], query[:, :, num_text:]
            k_txt, k_img = key[:, :, :num_text], key[:, :, num_text:]
            
            if hasattr(attn, 'norm_q') and attn.norm_q is not None:
                q_img = attn.norm_q(q_img)
            if hasattr(attn, 'norm_k') and attn.norm_k is not None:
                k_img = attn.norm_k(k_img)
            if hasattr(attn, 'norm_added_q') and attn.norm_added_q is not None:
                q_txt = attn.norm_added_q(q_txt)
            if hasattr(attn, 'norm_added_k') and attn.norm_added_k is not None:
                k_txt = attn.norm_added_k(k_txt)
            
            query = torch.cat([q_txt, q_img], dim=2)
            key = torch.cat([k_txt, k_img], dim=2)
        
        # Compute attention
        if self.use_flash and not self.capture:
            hidden_out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
            self.stored_i2t = None
        else:
            scale = head_dim ** -0.5
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            # Capture I2T attention
            if self.capture and num_text > 0:
                i2t = attn_weights[:, :, num_text:, :num_text]
                self.stored_i2t = i2t.mean(dim=1)  # [B, img, txt]
            else:
                self.stored_i2t = None
            
            hidden_out = torch.matmul(attn_weights, value)
        
        # Reshape from multi-head format back to [B, seq_len, dim]
        hidden_out = hidden_out.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_out = hidden_out.to(query.dtype)
        
        # === Output Projection ===
        # At this point hidden_out contains concatenated [text_tokens, image_tokens]
        # We need to split them and project through output layers separately
        
        if encoder_hidden_states is not None:
            # Split back into text and image portions
            enc_out = hidden_out[:, :num_text]   # Text features [B, num_text, dim]
            hid_out = hidden_out[:, num_text:]   # Image features [B, num_image, dim]
            
            # Project image features through output layers
            # attn.to_out is typically [Linear, Dropout]
            hid_out = attn.to_out[0](hid_out)    # Linear projection
            if len(attn.to_out) > 1:
                hid_out = attn.to_out[1](hid_out)  # Dropout (if present)
            
            # Project text features - SD3 has separate projection for text
            if hasattr(attn, 'to_add_out') and attn.to_add_out is not None:
                enc_out = attn.to_add_out(enc_out)  # Text-specific output projection
            else:
                # Fallback: use same projection as image (for compatibility)
                enc_out = attn.to_out[0](enc_out)
                if len(attn.to_out) > 1:
                    enc_out = attn.to_out[1](enc_out)
            
            # Return separately - transformer block handles them on different residual streams
            return hid_out, enc_out
        else:
            # Self-attention case: just project image features
            hidden_out = attn.to_out[0](hidden_out)
            if len(attn.to_out) > 1:
                hidden_out = attn.to_out[1](hidden_out)
            return hidden_out


# ============================================================================
#                              TOKEN FINDER
# ============================================================================

class TokenFinder:
    """Finds token indices for phrases in prompts."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._cache = {}
    
    def find(self, prompt: str, phrase: str) -> List[int]:
        key = (prompt.lower(), phrase.lower())
        if key in self._cache:
            return self._cache[key]
        
        prompt_tokens = self.tokenizer.tokenize(prompt.lower())
        phrase_tokens = self.tokenizer.tokenize(phrase.lower())
        
        indices = []
        for i in range(len(prompt_tokens) - len(phrase_tokens) + 1):
            match = all(
                prompt_tokens[i + j].replace('</w>', '').replace('Ġ', '') ==
                phrase_tokens[j].replace('</w>', '').replace('Ġ', '')
                for j in range(len(phrase_tokens))
            )
            if match:
                indices = list(range(i + 1, i + 1 + len(phrase_tokens)))
                break
        
        if not indices:
            for i, pt in enumerate(prompt_tokens):
                if phrase.lower() in pt.lower().replace('</w>', '').replace('Ġ', ''):
                    indices.append(i + 1)
        
        self._cache[key] = indices
        return indices


# ============================================================================
#                            MAIN GENERATOR
# ============================================================================

class SpatialGuidedSD3:
    """
    Modular spatial-guided generation for SD3.
    
    Key features:
    - Injectable prompts via PromptConfig
    - Pluggable loss strategies via GuidanceConfig.loss_strategy
    - Clean separation of concerns
    """
    
    def __init__(self, pipe, config: Optional[GuidanceConfig] = None):
        self.pipe = pipe
        self.config = config or GuidanceConfig()
        self.device = getattr(pipe, '_execution_device', None) or "cuda"
        self.token_finder = TokenFinder(pipe.tokenizer)
        self.processors: Dict[int, SpatialGuidanceAttnProcessor] = {}
    
    def _get_captured_attention(self) -> Dict[int, Tensor]:
        """Collect captured attention from processors."""
        return {idx: proc.stored_i2t for idx, proc in self.processors.items() 
                if proc.stored_i2t is not None}
    
    def generate(
        self,
        prompt_config: PromptConfig,
        num_inference_steps: int = 28,
        guidance_scale: float = 7.0,
        height: int = 1024,
        width: int = 1024,
        seed: int = 42,
        return_diagnostics: bool = False,
    ):
        """
        Generate an image with spatial guidance.
        
        Args:
            prompt_config: Encapsulates prompt and spatial constraint
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            height, width: Output image dimensions
            seed: Random seed for reproducibility
            return_diagnostics: If True, also return diagnostic info
        """
        # Validate prompt config
        prompt_config.validate()
        
        prompt = prompt_config.prompt
        constraint = prompt_config.constraint
        loss_strategy = self.config.loss_strategy
        
        # Find token indices
        tokens_A = self.token_finder.find(prompt, constraint.object_A)
        tokens_B = self.token_finder.find(prompt, constraint.object_B)
        
        print(f"Strategy: {loss_strategy.name}")
        print(f"Token indices: {constraint.object_A}={tokens_A}, {constraint.object_B}={tokens_B}")
        
        if not tokens_A or not tokens_B:
            raise ValueError("Could not find tokens for objects")
        
        transformer = self.pipe.transformer
        scheduler = self.pipe.scheduler
        vae = self.pipe.vae
        
        # Encode prompt
        (prompt_embeds, neg_prompt_embeds,
         pooled_embeds, neg_pooled_embeds) = self.pipe.encode_prompt(
            prompt=prompt, 
            prompt_2=prompt_config.prompt_2, 
            prompt_3=prompt_config.prompt_3,
            negative_prompt=prompt_config.negative_prompt or None,
            do_classifier_free_guidance=guidance_scale > 1,
        )
        
        if guidance_scale > 1:
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            pooled_embeds = torch.cat([neg_pooled_embeds, pooled_embeds], dim=0)
        
        # Initialize latents
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = self.pipe.prepare_latents(
            1, transformer.config.in_channels, height, width,
            prompt_embeds.dtype, self.device, generator
        )
        
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        max_guidance_steps = int(len(timesteps) * self.config.guidance_timestep_fraction)
        
        print(f"Max guidance steps: {max_guidance_steps}/{len(timesteps)}")
        print(f"Guidance target: {self.config.guidance_target.value}")
        print(f"Early exit: {'enabled' if self.config.enable_early_exit else 'disabled'}")
        
        all_diagnostics = []
        original_forwards = {}
        guidance_active = True
        steps_with_guidance = 0
        
        def scale_input(latent, t):
            if hasattr(scheduler, 'scale_model_input'):
                return scheduler.scale_model_input(latent, t)
            return latent
        
        def make_patched_forward(block_idx: int):
            proc = SpatialGuidanceAttnProcessor(
                block_idx, capture=True, 
                use_flash=self.config.use_flash_attention
            )
            self.processors[block_idx] = proc
            
            def patched_forward(hidden_states, encoder_hidden_states=None, 
                               attention_mask=None, **kwargs):
                attn = transformer.transformer_blocks[block_idx].attn
                return proc(attn, hidden_states, encoder_hidden_states, attention_mask)
            
            return patched_forward
        
        try:
            for i, t in enumerate(timesteps):
                t_frac = (len(timesteps) - i) / len(timesteps)
                apply_guidance = guidance_active and i < max_guidance_steps
                
                # Get adaptive blocks if enabled
                if self.config.enable_adaptive_blocks:
                    current_blocks = get_adaptive_blocks(t_frac)
                else:
                    current_blocks = self.config.measure_blocks
                
                # Install/update processors for current blocks
                if apply_guidance:
                    self.processors.clear()
                    for block_idx in current_blocks:
                        if block_idx < len(transformer.transformer_blocks):
                            if block_idx not in original_forwards:
                                block = transformer.transformer_blocks[block_idx]
                                original_forwards[block_idx] = block.attn.forward
                            transformer.transformer_blocks[block_idx].attn.forward = \
                                make_patched_forward(block_idx)
                
                # Prepare inputs
                if guidance_scale > 1:
                    latent_input = torch.cat([latents] * 2)
                else:
                    latent_input = latents
                latent_input = scale_input(latent_input, t)
                timestep_batch = t.expand(latent_input.shape[0])
                
                if apply_guidance:
                    latents = latents.detach().requires_grad_(True)
                    latent_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
                    latent_input = scale_input(latent_input, t)
                    
                    # Forward pass
                    velocity = transformer(
                        hidden_states=latent_input,
                        timestep=timestep_batch,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )[0]
                    
                    # Collect attention
                    attention_store = self._get_captured_attention()
                    
                    if attention_store:
                        attn_list = [attention_store[idx] for idx in current_blocks 
                                     if idx in attention_store]
                        
                        if attn_list:
                            aggregated = torch.stack(attn_list).mean(dim=0)
                            batch_idx = min(1, aggregated.shape[0] - 1)
                            attn = aggregated[batch_idx]
                            
                            # Extract metrics using the configured strategy
                            metrics_A, metrics_B = loss_strategy.extract_metrics(
                                attn, tokens_A, tokens_B,
                                self.config.threshold_percentile,
                                self.config.threshold_steepness,
                            )
                            
                            # Compute loss using the configured strategy
                            L_total, diag = loss_strategy.compute_loss(
                                metrics_A, metrics_B, constraint, self.config
                            )
                            
                            # Compute gradient
                            grad = torch.autograd.grad(L_total, latents, retain_graph=False)[0]
                            
                            # Apply guidance based on target
                            if self.config.guidance_target == GuidanceTarget.LATENTS:
                                latents = latents - self.config.guidance_strength * grad
                            else:  # VELOCITY
                                effective_strength = (self.config.guidance_strength * 
                                                     self.config.velocity_guidance_scale)
                                velocity = velocity + effective_strength * grad
                            
                            # Diagnostics
                            diag['step'] = i
                            diag['grad_norm'] = grad.norm().item()
                            diag['blocks'] = current_blocks
                            all_diagnostics.append(diag)
                            
                            # correct = (constraint.sign * diag['delta']) > self.config.margin
                            # print(f"Step {i}: L={diag['L_total']:.4f}, Δ={diag['delta']:.3f} "
                            #       f"{'✓' if correct else '✗'}, |∇|={diag['grad_norm']:.6f}, "
                            #       f"blocks={current_blocks}")
                            k_info = f", K_A={diag.get('K_A', '?')}, K_B={diag.get('K_B', '?')}" if 'K_A' in diag else ""
                            gap_info = f", gap={diag.get('gap', 0):.3f}" if 'gap' in diag else ""
                            print(f"Step {i}: L={diag['L_total']:.4f}, Δ={diag['delta']:.3f} "
                                f"{'✓' if diag['delta'] > self.config.margin else '✗'}, |∇|={diag['grad_norm']:.6f}{gap_info}{k_info}")
                            
                            # Check early exit
                            should_continue, reason = should_continue_guidance(
                                diag, constraint, self.config, steps_with_guidance
                            )
                            if not should_continue:
                                print(f"Early exit at step {i}: {reason}")
                                guidance_active = False
                            
                            # Cleanup
                            del grad, L_total, aggregated, attn
                    
                    latents = latents.detach()
                    
                    # Recompute velocity only if targeting latents and not reusing
                    if (self.config.guidance_target == GuidanceTarget.LATENTS and 
                        not self.config.reuse_velocity):
                        with torch.no_grad():
                            latent_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
                            latent_input = scale_input(latent_input, t)
                            velocity = transformer(
                                hidden_states=latent_input,
                                timestep=timestep_batch,
                                encoder_hidden_states=prompt_embeds,
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0]
                else:
                    # Standard step
                    with torch.no_grad():
                        velocity = transformer(
                            hidden_states=latent_input,
                            timestep=timestep_batch,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0]
                
                # Apply CFG
                if guidance_scale > 1:
                    v_uncond, v_cond = velocity.chunk(2)
                    velocity = v_uncond + guidance_scale * (v_cond - v_uncond)
                
                # Denoise step
                latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
        
        finally:
            # Restore original forwards
            for block_idx, original in original_forwards.items():
                if block_idx < len(transformer.transformer_blocks):
                    transformer.transformer_blocks[block_idx].attn.forward = original
        
        # Decode
        with torch.no_grad():
            latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latents, return_dict=False)[0]
            image = self.pipe.image_processor.postprocess(image, output_type="pil")[0]
        
        print(f"\nTotal guidance steps applied: {steps_with_guidance}")
        
        if return_diagnostics:
            return image, all_diagnostics
        return image


# ============================================================================
#                              EXAMPLES
# ============================================================================

def example_centroid():
    """Example using centroid-based loss (original InfSplign approach)."""
    from diffusers import StableDiffusion3Pipeline
    
    print("=" * 60)
    print("Centroid-Based Spatial Guidance")
    print("=" * 60)
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # # Configure with centroid loss
    # config = GuidanceConfig(
    #     loss_strategy=CentroidLossStrategy(compute_entropy=False),
    #     guidance_strength=30.0,
    #     enable_early_exit=True,
    # )
    # Create config with Bayesian GMM
    config = GuidanceConfig(
        loss_strategy=BayesianGMMLossStrategy(),
        guidance_strength=30.0,
        guidance_timestep_fraction=0.5
    )
    
    # Create injectable prompt config
    # prompt_config = PromptConfig.simple(
    #     prompt="a potted plant below a chair",
    #     object_A="potted plant",
    #     object_B="chair",
    #     relation="below"
    # )
    # prompt_config = PromptConfig.simple(
    #     prompt="a wine glass above a bus",
    #     object_A="wine glass", 
    #     object_B="bus",
    #     relation="above"
    # )
    prompt_config = PromptConfig.simple("a giraffe below a car", "giraffe", "car", "below")
    guided = SpatialGuidedSD3(pipe, config)
    image, diagnostics = guided.generate(
        prompt_config=prompt_config,
        num_inference_steps=28,
        height=512,
        width=512,
        seed=42,
        return_diagnostics=True
    )
    
    image.save("generated_centroid.png")
    return image, diagnostics


# TODO: Add example_gmm() once GMMShapeLossStrategy is implemented
#
# def example_gmm():
#     """Example using GMM-based shape loss."""
#     config = GuidanceConfig(
#         loss_strategy=GMMShapeLossStrategy(num_components=3),
#         guidance_strength=30.0,
#     )
#     # Test with challenging cases like bicycles (hollow wheels)
#     ...


# TODO: Add example_comparison() to compare strategies side-by-side
#
# def example_comparison():
#     """Run both strategies on the same prompt for comparison."""
#     for name, strategy in [
#         ("centroid", CentroidLossStrategy()),
#         ("gmm-3", GMMShapeLossStrategy(num_components=3)),
#     ]:
#         ...


if __name__ == "__main__":
    # Run centroid example by default
    example_centroid()