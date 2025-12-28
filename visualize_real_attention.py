#!/usr/bin/env python3
"""
Generate images with GMM guidance and save attention maps for visualization.

This script:
1. Runs generation with GMM guidance
2. Saves attention maps during generation  
3. Creates centroid vs GMM visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
import math
from typing import Tuple, Dict, List, Any, Optional
from dataclasses import dataclass
from diffusers import StableDiffusion3Pipeline

# Will be imported from local files
import sys
sys.path.insert(0, '/tudelft.net/staff-umbrella/StudentsCVlab/vpetkov/InfSplign')

from sd3_guidance_modular import (
    GuidanceConfig, SpatialGuidedSD3, PromptConfig, CentroidLossStrategy
)
from gmm_loss_strategy import DifferentiableGMMLossStrategy


# ============================================================================
#                    GMM CONFIG (matching gmm_loss_strategy.py)
# ============================================================================

@dataclass
class GMMConfig:
    """Configuration for GMM fitting - EXACT match to working code."""
    K: int = 3
    em_iterations: int = 3
    cov_regularization: float = 1e-4
    chi2_confidence: float = 2.28
    min_weight: float = 1e-3
    soft_extremal_temperature: float = 0.05


# ============================================================================
#                    DIFFERENTIABLE EM (from gmm_loss_strategy.py)
# ============================================================================

class DifferentiableEM:
    """Differentiable Expectation-Maximization for GMM fitting."""

    @staticmethod
    def gaussian_log_prob(coords: torch.Tensor, mean: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
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
    def e_step(coords: torch.Tensor, point_weights: torch.Tensor,
               means: torch.Tensor, covs: torch.Tensor, mix_weights: torch.Tensor) -> torch.Tensor:
        K = means.shape[0]
        log_probs = []
        for k in range(K):
            log_p = DifferentiableEM.gaussian_log_prob(coords, means[k], covs[k])
            log_prob_weighted = log_p + torch.log(mix_weights[k] + 1e-10)
            log_probs.append(log_prob_weighted)

        log_probs = torch.stack(log_probs, dim=1)
        log_sum = torch.logsumexp(log_probs, dim=1, keepdim=True)
        log_responsibilities = log_probs - log_sum
        return torch.exp(log_responsibilities)

    @staticmethod
    def m_step(coords: torch.Tensor, point_weights: torch.Tensor,
               responsibilities: torch.Tensor, config: GMMConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def fit_gmm(attention: torch.Tensor, config: GMMConfig) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit GMM to attention map using differentiable EM."""
    H, W = attention.shape
    device = attention.device
    
    y, x = torch.meshgrid(
        torch.linspace(0, 1, H, device=device, dtype=torch.float32),
        torch.linspace(0, 1, W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    coords = torch.stack([x.flatten(), y.flatten()], dim=1).float()
    point_weights = attention.flatten().float()
    point_weights = point_weights / (point_weights.sum() + 1e-10)
    
    K = config.K
    
    # Initialization
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
        responsibilities = DifferentiableEM.e_step(coords, point_weights, means, covs, mix_weights)
        means, covs, mix_weights = DifferentiableEM.m_step(coords, point_weights, responsibilities, config)
    
    return means, covs, mix_weights


def compute_centroid(attention: torch.Tensor) -> Tuple[float, float]:
    """Compute simple weighted centroid."""
    H, W = attention.shape
    device = attention.device
    
    y, x = torch.meshgrid(
        torch.linspace(0, 1, H, device=device, dtype=torch.float32),
        torch.linspace(0, 1, W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    attention_f = attention.float()
    total = attention_f.sum()
    
    if total < 1e-8:
        return 0.5, 0.5
    
    cx = (attention_f * x).sum() / total
    cy = (attention_f * y).sum() / total
    
    return cx.item(), cy.item()


def plot_gmm_ellipse(ax, mean, cov, chi2_val, H, W, **kwargs):
    """Plot confidence ellipse for a Gaussian component."""
    mean_np = mean.detach().cpu().numpy()
    cov_np = cov.detach().cpu().numpy()
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_np)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    width_norm = 2 * np.sqrt(chi2_val * eigenvalues[0])
    height_norm = 2 * np.sqrt(chi2_val * eigenvalues[1])
    
    width_px = width_norm * W
    height_px = height_norm * H
    
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    mean_px = (mean_np[0] * W, mean_np[1] * H)
    
    ellipse = Ellipse(xy=mean_px, width=width_px, height=height_px, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    return ellipse


def visualize_attention(attention: torch.Tensor, object_name: str, output_path: str):
    """Create centroid vs GMM visualization."""
    config = GMMConfig()
    
    H, W = attention.shape
    attention_np = attention.detach().float().cpu().numpy()
    
    cx, cy = compute_centroid(attention)
    means, covs, weights = fit_gmm(attention, config)
    
    means_np = means.detach().cpu().numpy()
    weights_np = weights.detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # LEFT: Centroid Only
    ax1 = axes[0]
    im1 = ax1.imshow(attention_np, cmap='viridis', origin='upper')
    cx_px, cy_px = cx * W, cy * H
    ax1.scatter([cx_px], [cy_px], c='red', s=150, marker='x', linewidths=3, 
                label=f'Centroid ({cx_px:.1f}, {cy_px:.1f})')
    ax1.set_title(f'{object_name}: Centroid Approach\n(No spatial extent modeling)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(0, W)
    ax1.set_ylim(H, 0)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # RIGHT: GMM with Ellipses
    ax2 = axes[1]
    im2 = ax2.imshow(attention_np, cmap='viridis', origin='upper')
    
    colors = plt.cm.Set1(np.linspace(0, 1, max(config.K, 3)))
    
    active_components = 0
    for k in range(config.K):
        weight = weights_np[k]
        if weight < config.min_weight:
            continue
        active_components += 1
        
        plot_gmm_ellipse(ax2, means[k], covs[k], config.chi2_confidence, H, W,
                        fill=False, edgecolor=colors[k], linewidth=2.5)
        
        mean_px = (means_np[k, 0] * W, means_np[k, 1] * H)
        ax2.scatter([mean_px[0]], [mean_px[1]], c=[colors[k]], s=100, marker='o',
                    edgecolors='white', linewidths=1.5,
                    label=f'K{k+1}: w={weight:.2f}')
    
    ax2.set_title(f'{object_name}: GMM Approach (K={config.K}, active={active_components})\n'
                  f'(χ²={config.chi2_confidence} confidence ellipses)', 
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, W)
    ax2.set_ylim(H, 0)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {output_path}")


# ============================================================================
#                    MODIFIED GENERATION WITH ATTENTION SAVING
# ============================================================================

class AttentionCaptureMixin:
    """Mixin to capture attention maps during generation."""
    
    captured_attention_A: Optional[torch.Tensor] = None
    captured_attention_B: Optional[torch.Tensor] = None


def generate_with_attention_capture(
    pipe,
    prompt_config: PromptConfig,
    seed: int = 42,
    output_dir: str = "/tudelft.net/staff-umbrella/StudentsCVlab/vpetkov/InfSplign",
):
    """
    Generate image with GMM guidance and capture attention maps for visualization.
    """
    output_dir = Path(output_dir)
    
    # Setup GMM strategy
    gmm_strategy = DifferentiableGMMLossStrategy(
        K=3, em_iterations=3,
        lambda_boundary=1.0, lambda_directional=1.0,
        lambda_presence=0.5, lambda_size=0.3,
        margin_boundary=0.1, use_phase_lambdas=False,
    )
    
    gmm_config = GuidanceConfig(
        loss_strategy=gmm_strategy,
        guidance_strength=30.0,
        guidance_timestep_fraction=0.5,
        threshold_percentile=0.7
    )
    
    gmm_guided = SpatialGuidedSD3(pipe, gmm_config)
    
    # Storage for attention maps
    attention_maps = {'A': [], 'B': []}
    
    # Monkey-patch extract_metrics to capture attention
    original_extract = gmm_strategy.extract_metrics
    
    def capturing_extract(attention, token_indices_A, token_indices_B, 
                         threshold_percentile, threshold_steepness):
        # Call original
        metrics_A, metrics_B = original_extract(
            attention, token_indices_A, token_indices_B,
            threshold_percentile, threshold_steepness
        )
        
        # Capture raw attention for objects
        if attention.dim() == 2:
            attention = attention.unsqueeze(0)
        
        B, num_spatial, num_tokens = attention.shape
        H = W = int(num_spatial ** 0.5)
        
        attn_A = attention[:, :, token_indices_A].mean(dim=-1).view(B, H, W)[0]
        attn_B = attention[:, :, token_indices_B].mean(dim=-1).view(B, H, W)[0]
        
        attention_maps['A'].append(attn_A.detach().cpu())
        attention_maps['B'].append(attn_B.detach().cpu())
        
        return metrics_A, metrics_B
    
    gmm_strategy.extract_metrics = capturing_extract
    
    # Generate
    print(f"\n{'='*50}")
    print(f"Generating with GMM guidance (seed={seed})")
    print(f"Prompt: {prompt_config.prompt}")
    print(f"Constraint: {prompt_config.constraint}")
    print(f"{'='*50}\n")
    
    gmm_strategy.reset()
    image, diagnostics = gmm_guided.generate(
        prompt_config=prompt_config,
        num_inference_steps=28,
        height=512,
        width=512,
        seed=seed,
        return_diagnostics=True
    )
    
    # Save generated image
    image_path = output_dir / f"generated_{prompt_config.constraint.object_A}_seed{seed}.png"
    image.save(str(image_path))
    print(f"Saved image: {image_path}")
    
    # Average attention maps across timesteps (use later timesteps for cleaner attention)
    if attention_maps['A']:
        # Use last few timesteps where attention is most refined
        n_use = min(5, len(attention_maps['A']))
        attn_A_avg = torch.stack(attention_maps['A'][-n_use:]).mean(dim=0)
        attn_B_avg = torch.stack(attention_maps['B'][-n_use:]).mean(dim=0)
        
        # Create visualizations
        obj_A = prompt_config.constraint.object_A
        obj_B = prompt_config.constraint.object_B
        
        viz_path_A = output_dir / f"centroid_vs_gmm_{obj_A}.png"
        viz_path_B = output_dir / f"centroid_vs_gmm_{obj_B}.png"
        
        visualize_attention(attn_A_avg, obj_A, str(viz_path_A))
        visualize_attention(attn_B_avg, obj_B, str(viz_path_B))
    
    return image, diagnostics


def main():
    print("Loading SD3 pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Test case: giraffe below car
    prompt_config = PromptConfig.simple(
        prompt="a giraffe below a car",
        object_A="giraffe",
        object_B="car", 
        relation="below"
    )
    
    generate_with_attention_capture(
        pipe=pipe,
        prompt_config=prompt_config,
        seed=42,
    )
    
    print("\n" + "="*50)
    print("Done! Check output files:")
    print("  - generated_giraffe_seed42.png")
    print("  - centroid_vs_gmm_giraffe.png")
    print("  - centroid_vs_gmm_car.png")
    print("="*50)


if __name__ == "__main__":
    main()