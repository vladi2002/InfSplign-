"""
Differentiable GMM Loss Strategy for SD3 Spatial Guidance - V5

Key improvements in this version:
1. DIRECTIONAL SEPARATION: Penalizes object A's attention in the "wrong" region
   relative to B, rather than symmetric overlap
2. PHASE-BASED LAMBDAS: Different loss weights for early/mid/late generation
3. BOUNDARY-ONLY SPATIAL: Removed L_centroid, rely solely on boundary constraints
4. AXIS-AWARE REPULSION: Repulsion only along the constraint axis

Author: Vladi (TU Delft MSc Thesis)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, NamedTuple, Any
from dataclasses import dataclass
import math


# ============================================================================
#                            DATA STRUCTURES
# ============================================================================

class GMMSpatialMetrics(NamedTuple):
    """Boundary-based metrics from GMM fitting."""
    y_top: Tensor       # Top boundary (smallest y) - soft-min
    y_bottom: Tensor    # Bottom boundary (largest y) - soft-max
    x_left: Tensor      # Left boundary (smallest x) - soft-min
    x_right: Tensor     # Right boundary (largest x) - soft-max
    centroid_x: Tensor  # Weighted centroid x (for diagnostics)
    centroid_y: Tensor  # Weighted centroid y (for diagnostics)
    variance: Tensor    # GMM ellipse area (for presence loss)


@dataclass
class GMMState:
    """GMM parameters - can be warm-started across timesteps."""
    means: Tensor      # [K, 2]
    covs: Tensor       # [K, 2, 2]
    weights: Tensor    # [K]
    
    def detach(self) -> 'GMMState':
        return GMMState(
            means=self.means.detach().clone(),
            covs=self.covs.detach().clone(),
            weights=self.weights.detach().clone()
        )


@dataclass
class GMMConfig:
    """Configuration for GMM fitting."""
    K: int = 2
    em_iterations: int = 3
    cov_regularization: float = 1e-4
    chi2_confidence: float = 2.28
    min_weight: float = 1e-3
    soft_extremal_temperature: float = 0.05


# ============================================================================
#                         SOFT-SQUARED HINGE LOSS
# ============================================================================

def soft_squared_hinge(x: Tensor, beta: float = 1.0) -> Tensor:
    """
    Smooth approximation of relu(x)².
    
    Formula: x * softplus(x) = x * log(1 + exp(x))
    
    Properties:
    - When x < 0 (satisfied): approaches 0 smoothly
    - When x > 0 (violated): approaches x² 
    - Smooth everywhere, no kinks
    - Gradient increases with violation magnitude
    """
    return x * F.softplus(beta * x) / beta


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
                torch.linspace(0, 1, H, device=device, dtype=torch.float32),
                torch.linspace(0, 1, W, device=device, dtype=torch.float32),
                indexing='ij'
            )
            cls._cache[key] = (x, y)
        return cls._cache[key]


# ============================================================================
#                         DIFFERENTIABLE EM
# ============================================================================

class DifferentiableEM:
    """Differentiable Expectation-Maximization for GMM fitting."""
    
    @staticmethod
    def gaussian_log_prob(coords: Tensor, mean: Tensor, cov: Tensor) -> Tensor:
        """Compute log N(coords | mean, cov) for each point."""
        coords_f = coords.float()
        mean_f = mean.float()
        cov_f = cov.float()
        
        diff = coords_f - mean_f
        L = torch.linalg.cholesky(cov_f)
        solved = torch.linalg.solve_triangular(L, diff.T, upper=False)
        mahalanobis = (solved ** 2).sum(dim=0)
        log_det = 2 * torch.log(torch.diag(L)).sum()
        log_prob = -0.5 * (mahalanobis + log_det + 2 * math.log(2 * math.pi))
        return log_prob
    
    @staticmethod
    def e_step(coords: Tensor, point_weights: Tensor,
               means: Tensor, covs: Tensor, mix_weights: Tensor) -> Tensor:
        """E-step: Compute responsibilities."""
        K = means.shape[0]
        
        log_probs = []
        for k in range(K):
            log_p = DifferentiableEM.gaussian_log_prob(coords, means[k], covs[k])
            log_prob_weighted = log_p + torch.log(mix_weights[k] + 1e-10)
            log_probs.append(log_prob_weighted)
        
        log_probs = torch.stack(log_probs, dim=1)
        log_sum = torch.logsumexp(log_probs, dim=1, keepdim=True)
        log_responsibilities = log_probs - log_sum
        responsibilities = torch.exp(log_responsibilities)
        
        return responsibilities
    
    @staticmethod
    def m_step(coords: Tensor, point_weights: Tensor,
               responsibilities: Tensor, config: GMMConfig) -> Tuple[Tensor, Tensor, Tensor]:
        """M-step: Update GMM parameters."""
        K = responsibilities.shape[1]
        device = coords.device
        
        weighted_resp = responsibilities * point_weights.unsqueeze(1)
        Nk = weighted_resp.sum(dim=0) + 1e-10
        
        new_weights = Nk / Nk.sum()
        new_weights = torch.clamp(new_weights, min=config.min_weight)
        new_weights = new_weights / new_weights.sum()
        
        new_means = torch.zeros(K, 2, device=device, dtype=torch.float32)
        new_covs = torch.zeros(K, 2, 2, device=device, dtype=torch.float32)
        
        coords_f = coords.float()
        
        for k in range(K):
            w = weighted_resp[:, k]
            new_means[k] = (w.unsqueeze(1) * coords_f).sum(dim=0) / Nk[k]
            
            diff = coords_f - new_means[k]
            outer = torch.einsum('ni,nj->nij', diff, diff)
            new_covs[k] = (w.unsqueeze(1).unsqueeze(2) * outer).sum(dim=0) / Nk[k]
            new_covs[k] = new_covs[k] + config.cov_regularization * torch.eye(2, device=device)
        
        return new_means, new_covs, new_weights


# ============================================================================
#                         WARM-START GMM FITTING
# ============================================================================

class WarmStartGMM:
    """GMM fitting with warm-start support."""
    
    @staticmethod
    def fit(attention: Tensor, x_coords: Tensor, y_coords: Tensor,
            config: GMMConfig, warm_start: Optional[GMMState] = None) -> GMMState:
        """Fit GMM to attention map."""
        H, W = attention.shape
        device = attention.device
        
        coords = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=1).float()
        point_weights = attention.flatten().float()
        point_weights = point_weights / (point_weights.sum() + 1e-10)
        
        K = config.K
        
        if warm_start is not None:
            means = warm_start.means.clone()
            covs = warm_start.covs.clone()
            mix_weights = warm_start.weights.clone()
        else:
            # Differentiable initialization
            weighted_mean = (point_weights.unsqueeze(1) * coords).sum(dim=0)
            
            diff = coords - weighted_mean
            weighted_cov = (point_weights.unsqueeze(1).unsqueeze(2) * 
                           torch.einsum('ni,nj->nij', diff, diff)).sum(dim=0)
            
            eigvals = torch.linalg.eigvalsh(weighted_cov.float() + 1e-6 * torch.eye(2, device=device))
            spread = torch.sqrt(eigvals.max()) * 0.5
            
            means = torch.zeros(K, 2, device=device, dtype=torch.float32)
            angles = torch.linspace(0, 2 * math.pi * (K-1)/K, K, device=device)
            for k in range(K):
                offset = spread * torch.stack([torch.cos(angles[k]), torch.sin(angles[k])])
                means[k] = weighted_mean + offset
            
            covs = torch.eye(2, device=device, dtype=torch.float32).unsqueeze(0).expand(K, -1, -1) * 0.01
            covs = covs.clone()
            mix_weights = torch.ones(K, device=device, dtype=torch.float32) / K
        
        # Run EM
        for _ in range(config.em_iterations):
            responsibilities = DifferentiableEM.e_step(
                coords, point_weights, means, covs, mix_weights)
            means, covs, mix_weights = DifferentiableEM.m_step(
                coords, point_weights, responsibilities, config)
        
        return GMMState(means=means, covs=covs, weights=mix_weights)
    
    @staticmethod
    def compute_metrics(state: GMMState, config: GMMConfig) -> GMMSpatialMetrics:
        """Compute soft-extremal boundary metrics."""
        K = state.means.shape[0]
        chi2_val = config.chi2_confidence
        device = state.means.device
        T = config.soft_extremal_temperature
        
        y_tops_list, y_bottoms_list = [], []
        x_lefts_list, x_rights_list = [], []
        component_weights_list = []
        ellipse_areas_list = []
        
        for k in range(K):
            mean = state.means[k].float()
            cov = state.covs[k].float()
            weight = state.weights[k]
            
            if weight < config.min_weight:
                continue
            
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = torch.clamp(eigenvalues, min=1e-6)
            
            theta = torch.atan2(eigenvectors[1, 1], eigenvectors[0, 1])
            a = torch.sqrt(chi2_val * eigenvalues[1])  # semi-major axis
            b = torch.sqrt(chi2_val * eigenvalues[0])  # semi-minor axis
            
            ellipse_area = math.pi * a * b
            ellipse_areas_list.append(ellipse_area)
            
            y_extent = torch.sqrt((a * torch.sin(theta))**2 + (b * torch.cos(theta))**2)
            x_extent = torch.sqrt((a * torch.cos(theta))**2 + (b * torch.sin(theta))**2)
            
            y_tops_list.append(mean[1] - y_extent)
            y_bottoms_list.append(mean[1] + y_extent)
            x_lefts_list.append(mean[0] - x_extent)
            x_rights_list.append(mean[0] + x_extent)
            component_weights_list.append(weight)
        
        if len(y_tops_list) == 0:
            return GMMSpatialMetrics(
                y_top=torch.tensor(0.5, device=device),
                y_bottom=torch.tensor(0.5, device=device),
                x_left=torch.tensor(0.5, device=device),
                x_right=torch.tensor(0.5, device=device),
                centroid_x=torch.tensor(0.5, device=device),
                centroid_y=torch.tensor(0.5, device=device),
                variance=torch.tensor(0.1, device=device)
            )
        
        y_tops = torch.stack(y_tops_list)
        y_bottoms = torch.stack(y_bottoms_list)
        x_lefts = torch.stack(x_lefts_list)
        x_rights = torch.stack(x_rights_list)
        comp_weights = torch.stack(component_weights_list)
        comp_weights = comp_weights / (comp_weights.sum() + 1e-8)
        
        # Soft-extremal: weighted soft-min/max
        y_top_weights = F.softmax(-y_tops / T, dim=0) * comp_weights
        y_top_weights = y_top_weights / (y_top_weights.sum() + 1e-8)
        y_top_soft = (y_top_weights * y_tops).sum()
        
        y_bottom_weights = F.softmax(y_bottoms / T, dim=0) * comp_weights
        y_bottom_weights = y_bottom_weights / (y_bottom_weights.sum() + 1e-8)
        y_bottom_soft = (y_bottom_weights * y_bottoms).sum()
        
        x_left_weights = F.softmax(-x_lefts / T, dim=0) * comp_weights
        x_left_weights = x_left_weights / (x_left_weights.sum() + 1e-8)
        x_left_soft = (x_left_weights * x_lefts).sum()
        
        x_right_weights = F.softmax(x_rights / T, dim=0) * comp_weights
        x_right_weights = x_right_weights / (x_right_weights.sum() + 1e-8)
        x_right_soft = (x_right_weights * x_rights).sum()
        
        # Clamp
        y_top_soft = torch.clamp(y_top_soft, 0.0, 1.0)
        y_bottom_soft = torch.clamp(y_bottom_soft, 0.0, 1.0)
        x_left_soft = torch.clamp(x_left_soft, 0.0, 1.0)
        x_right_soft = torch.clamp(x_right_soft, 0.0, 1.0)
        
        # Weighted centroid (for diagnostics)
        weights_f = state.weights.float()
        means_f = state.means.float()
        cx = (weights_f * means_f[:, 0]).sum()
        cy = (weights_f * means_f[:, 1]).sum()
        
        # GMM ellipse area for presence
        if len(ellipse_areas_list) > 0:
            ellipse_areas = torch.stack(ellipse_areas_list)
            gmm_area = (comp_weights * ellipse_areas).sum()
        else:
            gmm_area = torch.tensor(0.1, device=device)
        
        return GMMSpatialMetrics(
            y_top=y_top_soft,
            y_bottom=y_bottom_soft,
            x_left=x_left_soft,
            x_right=x_right_soft,
            centroid_x=torch.clamp(cx, 0.0, 1.0),
            centroid_y=torch.clamp(cy, 0.0, 1.0),
            variance=gmm_area
        )


# ============================================================================
#                    DIFFERENTIABLE GMM LOSS STRATEGY V5
# ============================================================================

class DifferentiableGMMLossStrategy:
    """
    GMM-based spatial guidance V5 with:
    - BOUNDARY-ONLY spatial constraints (no centroid loss)
    - DIRECTIONAL separation loss
    - PHASE-BASED lambda scheduling
    - AXIS-AWARE repulsion
    
    Uses soft-squared hinge loss: x * softplus(x)
    """
    
    def __init__(
        self,
        K: int = 3,
        em_iterations: int = 3,
        chi2_confidence: float = 2.28,
        use_warm_start: bool = True,
        soft_extremal_temperature: float = 0.05,
        # Base loss weights (modified by phase)
        lambda_boundary: float = 1.0,
        lambda_directional: float = 1.0,
        lambda_presence: float = 0.5,
        lambda_size: float = 1.0,
        # Margins
        margin_boundary: float = 0.2,
        # Size constraint
        target_size: float = 0.15,
        # Softplus sharpness
        softplus_beta: float = 5.0,
        # Directional separation sigmoid steepness
        directional_steepness: float = 10.0,
        # Enable phase-based lambdas
        use_phase_lambdas: bool = True,
    ):
        self.gmm_config = GMMConfig(
            K=K,
            em_iterations=em_iterations,
            chi2_confidence=chi2_confidence,
            soft_extremal_temperature=soft_extremal_temperature,
        )
        self.use_warm_start = use_warm_start
        self.warm_states: Dict[str, GMMState] = {}
        
        # Base loss parameters
        self.lambda_boundary = lambda_boundary
        self.lambda_directional = lambda_directional
        self.lambda_presence = lambda_presence
        self.lambda_size = lambda_size
        self.margin_boundary = margin_boundary
        self.target_size = target_size
        self.softplus_beta = softplus_beta
        self.directional_steepness = directional_steepness
        self.use_phase_lambdas = use_phase_lambdas
        
        # Store normalized attention maps for directional loss
        self.current_attn_A_norm: Optional[Tensor] = None
        self.current_attn_B_norm: Optional[Tensor] = None
        self.current_y_coords: Optional[Tensor] = None
        self.current_x_coords: Optional[Tensor] = None
    
    @property
    def name(self) -> str:
        return f"gmm-K{self.gmm_config.K}-v5"
    
    def reset(self):
        """Reset warm-start states."""
        self.warm_states.clear()
        self.current_attn_A_norm = None
        self.current_attn_B_norm = None
    
    def get_phase_lambdas(self, step_fraction: float) -> Dict[str, float]:
        """
        Get phase-appropriate lambda values.
        
        step_fraction: fraction of generation remaining (1.0 = start, 0.0 = end)
        
        Phase 1 (early, step_fraction > 0.7): Establish positions aggressively
        Phase 2 (mid, 0.4 < step_fraction <= 0.7): Balance position and separation  
        Phase 3 (late, step_fraction <= 0.4): Lock in, strong separation
        """
        if not self.use_phase_lambdas:
            return {
                'lambda_boundary': self.lambda_boundary,
                'lambda_directional': self.lambda_directional,
                'lambda_size': self.lambda_size,
            }
        
        if step_fraction > 0.7:
            # Early: strong boundary push, weak directional (allow crossing)
            return {
                'lambda_boundary': self.lambda_boundary * 2.0,
                'lambda_directional': self.lambda_directional * 0.3,
                'lambda_size': self.lambda_size * 1.0,
            }
        elif step_fraction > 0.4:
            # Mid: balance all terms
            return {
                'lambda_boundary': self.lambda_boundary * 1.0,
                'lambda_directional': self.lambda_directional * 1.0,
                'lambda_size': self.lambda_size * 1.0,
            }
        else:
            # Late: lock positions, strong directional to fix blending
            return {
                'lambda_boundary': self.lambda_boundary * 1.5,
                'lambda_directional': self.lambda_directional * 2.0,
                'lambda_size': self.lambda_size * 0.5,
            }
    
    def compute_directional_separation(
        self,
        attn_A_norm: Tensor,
        attn_B_norm: Tensor,
        y_coords: Tensor,
        x_coords: Tensor,
        constraint: Any,
    ) -> Tensor:
        """
        Compute directional separation loss.
        
        Penalizes object A's attention in the "wrong" region relative to B.
        For "A below B": penalize A's attention where y < B's centroid
        For "A above B": penalize A's attention where y > B's centroid
        etc.
        
        This replaces the symmetric overlap penalty.
        """
        # Get B's centroid as reference point
        B_cy = (attn_B_norm * y_coords).sum()
        B_cx = (attn_B_norm * x_coords).sum()
        
        steepness = self.directional_steepness
        
        if constraint.relation == "below":
            # A should be below B → penalize A's attention where y < B_cy (above B)
            wrong_region_weight = torch.sigmoid(steepness * (B_cy - y_coords))
            L_directional = (attn_A_norm * wrong_region_weight).sum()
            
        elif constraint.relation == "above":
            # A should be above B → penalize A's attention where y > B_cy (below B)
            wrong_region_weight = torch.sigmoid(steepness * (y_coords - B_cy))
            L_directional = (attn_A_norm * wrong_region_weight).sum()
            
        elif constraint.relation == "left":
            # A should be left of B → penalize A's attention where x > B_cx (right of B)
            wrong_region_weight = torch.sigmoid(steepness * (x_coords - B_cx))
            L_directional = (attn_A_norm * wrong_region_weight).sum()
            
        elif constraint.relation == "right":
            # A should be right of B → penalize A's attention where x < B_cx (left of B)
            wrong_region_weight = torch.sigmoid(steepness * (B_cx - x_coords))
            L_directional = (attn_A_norm * wrong_region_weight).sum()
        
        else:
            L_directional = torch.tensor(0.0, device=attn_A_norm.device)
        
        return L_directional
    
    def extract_metrics(
        self,
        attention: Tensor,
        token_indices_A: List[int],
        token_indices_B: List[int],
        threshold_percentile: float,
        threshold_steepness: float,
    ) -> Tuple[GMMSpatialMetrics, GMMSpatialMetrics]:
        """Extract GMM-based metrics for both objects."""
        
        if attention.dim() == 2:
            attention = attention.unsqueeze(0)
        
        B, num_spatial, num_tokens = attention.shape
        device = attention.device
        H = W = int(num_spatial ** 0.5)
        
        x_coords, y_coords = CoordinateGridCache.get(H, W, device)
        
        attn_A = attention[:, :, token_indices_A].mean(dim=-1).view(B, H, W)
        attn_B = attention[:, :, token_indices_B].mean(dim=-1).view(B, H, W)
        
        if threshold_percentile > 0:
            attn_A = self._apply_soft_threshold(attn_A, threshold_percentile, threshold_steepness)
            attn_B = self._apply_soft_threshold(attn_B, threshold_percentile, threshold_steepness)
        
        # Normalize attention maps
        attn_A_norm = attn_A / (attn_A.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        attn_B_norm = attn_B / (attn_B.sum(dim=(-2, -1), keepdim=True) + 1e-8)
        
        # Store for directional loss computation
        self.current_attn_A_norm = attn_A_norm[0]  # [H, W]
        self.current_attn_B_norm = attn_B_norm[0]
        self.current_y_coords = y_coords
        self.current_x_coords = x_coords
        
        attn_A_2d = attn_A_norm[0]
        attn_B_2d = attn_B_norm[0]
        
        warm_A = self.warm_states.get('object_A') if self.use_warm_start else None
        warm_B = self.warm_states.get('object_B') if self.use_warm_start else None
        
        state_A = WarmStartGMM.fit(attn_A_2d, x_coords, y_coords, self.gmm_config, warm_A)
        state_B = WarmStartGMM.fit(attn_B_2d, x_coords, y_coords, self.gmm_config, warm_B)
        
        if self.use_warm_start:
            self.warm_states['object_A'] = state_A.detach()
            self.warm_states['object_B'] = state_B.detach()
        
        metrics_A = WarmStartGMM.compute_metrics(state_A, self.gmm_config)
        metrics_B = WarmStartGMM.compute_metrics(state_B, self.gmm_config)
        
        return metrics_A, metrics_B
    
    def _apply_soft_threshold(self, attention: Tensor, percentile: float, steepness: float) -> Tensor:
        """Apply differentiable soft thresholding."""
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
    
    def compute_loss(
        self,
        metrics_A: GMMSpatialMetrics,
        metrics_B: GMMSpatialMetrics,
        constraint: Any,
        config: Any,
        step_fraction: float = 0.5,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Compute V5 loss with directional separation and phase-based lambdas.
        
        Loss = λ_b * SSH(margin - gap_boundary) 
             + λ_d * L_directional
             + λ_p * L_presence
             + λ_s * L_size
        
        Args:
            metrics_A, metrics_B: GMM spatial metrics
            constraint: SpatialConstraint with .relation attribute
            config: GuidanceConfig (unused in V5, kept for API compatibility)
            step_fraction: fraction of generation remaining (1.0 = start, 0.0 = end)
        """
        device = metrics_A.y_top.device
        
        # Get phase-appropriate lambdas
        lambdas = self.get_phase_lambdas(step_fraction)
        
        # =====================================================
        # 1. BOUNDARY GAP LOSS (primary spatial constraint)
        # =====================================================
        if constraint.relation == "below":
            # A below B: A's top should be below B's bottom
            gap_boundary = metrics_A.y_top - metrics_B.y_bottom
        elif constraint.relation == "above":
            # A above B: B's top should be below A's bottom
            gap_boundary = metrics_B.y_top - metrics_A.y_bottom
        elif constraint.relation == "right":
            # A right of B: A's left should be right of B's right
            gap_boundary = metrics_A.x_left - metrics_B.x_right
        elif constraint.relation == "left":
            # A left of B: B's left should be right of A's right
            gap_boundary = metrics_B.x_left - metrics_A.x_right
        else:
            gap_boundary = torch.tensor(0.0, device=device)
        
        L_boundary = soft_squared_hinge(
            self.margin_boundary - gap_boundary, 
            beta=self.softplus_beta
        )
        
        # =====================================================
        # 2. DIRECTIONAL SEPARATION LOSS
        # =====================================================
        if self.current_attn_A_norm is not None:
            L_directional = self.compute_directional_separation(
                self.current_attn_A_norm,
                self.current_attn_B_norm,
                self.current_y_coords,
                self.current_x_coords,
                constraint,
            )
        else:
            L_directional = torch.tensor(0.0, device=device)
        
        # =====================================================
        # 3. PRESENCE LOSS (keep objects visible)
        # =====================================================
        L_presence = metrics_A.variance + metrics_B.variance
        
        # =====================================================
        # 4. SIZE CONSTRAINT (cap maximum size)
        # =====================================================
        L_size_A = soft_squared_hinge(
            metrics_A.variance - self.target_size,
            beta=self.softplus_beta
        )
        L_size_B = soft_squared_hinge(
            metrics_B.variance - self.target_size,
            beta=self.softplus_beta
        )
        L_size = L_size_A + L_size_B
        
        # =====================================================
        # TOTAL LOSS
        # =====================================================
        L_total = (
            lambdas['lambda_boundary'] * L_boundary +
            lambdas['lambda_directional'] * L_directional +
            self.lambda_presence * L_presence +
            lambdas['lambda_size'] * L_size
        )
        
        # Compute delta for diagnostics (centroid-based, for reference)
        if constraint.axis == "x":
            delta = constraint.sign * (metrics_A.centroid_x - metrics_B.centroid_x)
        else:
            delta = constraint.sign * (metrics_A.centroid_y - metrics_B.centroid_y)
        
        diagnostics = {
            'L_boundary': L_boundary.item(),
            'L_directional': L_directional.item() if torch.is_tensor(L_directional) else L_directional,
            'L_presence': L_presence.item(),
            'L_size': L_size.item(),
            'L_total': L_total.item(),
            'delta': delta.item(),
            'gap_boundary': gap_boundary.item(),
            'step_fraction': step_fraction,
            # Phase info
            'phase_lambda_boundary': lambdas['lambda_boundary'],
            'phase_lambda_directional': lambdas['lambda_directional'],
            # Size info
            'size_A': metrics_A.variance.item(),
            'size_B': metrics_B.variance.item(),
            # Boundary info
            'A_y_top': metrics_A.y_top.item(),
            'A_y_bottom': metrics_A.y_bottom.item(),
            'A_x_left': metrics_A.x_left.item(),
            'A_x_right': metrics_A.x_right.item(),
            'B_y_top': metrics_B.y_top.item(),
            'B_y_bottom': metrics_B.y_bottom.item(),
            'B_x_left': metrics_B.x_left.item(),
            'B_x_right': metrics_B.x_right.item(),
            # Centroids (for reference/visualization)
            'cx_A': metrics_A.centroid_x.item(),
            'cy_A': metrics_A.centroid_y.item(),
            'cx_B': metrics_B.centroid_x.item(),
            'cy_B': metrics_B.centroid_y.item(),
            # GMM info
            'K': self.gmm_config.K,
        }
        
        return L_total, diagnostics


# ============================================================================
#                              TEST
# ============================================================================

if __name__ == "__main__":
    print("Testing GMM Strategy V5 (Directional + Phase-Based)...")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    strategy = DifferentiableGMMLossStrategy(
        K=3, 
        em_iterations=3,
        lambda_boundary=1.0,
        lambda_directional=1.0,
        lambda_presence=0.5,
        lambda_size=1.0,
        margin_boundary=0.2,
        use_phase_lambdas=True,
    )
    
    H, W = 64, 64
    num_tokens = 10
    
    # Create synthetic attention
    attention = torch.rand(1, H*W, num_tokens, device=device, requires_grad=True)
    
    tokens_A = [2]
    tokens_B = [5]
    
    metrics_A, metrics_B = strategy.extract_metrics(
        attention, tokens_A, tokens_B,
        threshold_percentile=0.5,
        threshold_steepness=50.0
    )
    
    print("=== Metrics ===")
    print(f"A: y=[{metrics_A.y_top.item():.3f}, {metrics_A.y_bottom.item():.3f}], cy={metrics_A.centroid_y.item():.3f}")
    print(f"B: y=[{metrics_B.y_top.item():.3f}, {metrics_B.y_bottom.item():.3f}], cy={metrics_B.centroid_y.item():.3f}")
    
    class MockConstraint:
        relation = "below"
        axis = "y"
        sign = 1
    
    class MockConfig:
        margin = 0.05
    
    # Test different phases
    for step_frac in [0.9, 0.5, 0.2]:
        loss, diag = strategy.compute_loss(
            metrics_A, metrics_B, MockConstraint(), MockConfig(),
            step_fraction=step_frac
        )
        
        print(f"\n=== Step Fraction: {step_frac} ===")
        print(f"L_boundary:    {diag['L_boundary']:.4f} (λ={diag['phase_lambda_boundary']:.1f})")
        print(f"L_directional: {diag['L_directional']:.4f} (λ={diag['phase_lambda_directional']:.1f})")
        print(f"L_presence:    {diag['L_presence']:.4f}")
        print(f"L_size:        {diag['L_size']:.4f}")
        print(f"L_total:       {diag['L_total']:.4f}")
        print(f"gap_boundary:  {diag['gap_boundary']:.3f}")
    
    # Test gradient flow
    loss, _ = strategy.compute_loss(
        metrics_A, metrics_B, MockConstraint(), MockConfig(), step_fraction=0.5
    )
    loss.backward()
    has_grad = attention.grad is not None and attention.grad.abs().sum() > 0
    print()
    print(f"Gradient flow: {'OK' if has_grad else 'BROKEN'}")
    
    print()
    print("Test complete!")