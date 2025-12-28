"""
MCMC-Enhanced Spatial Guidance for SD3

This integrates MCMC sampling (ULA, MALA, UHMC) with the existing
SD3 spatial guidance framework to help overcome semantic prior resistance.

Based on "Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models" (ICML 2023)
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import argparse
import os
import json
from PIL import Image

# ============================================================================
#                              MCMC SAMPLERS
# ============================================================================

class MCMCSampler:
    """Base class for MCMC samplers."""
    
    def __init__(self, num_steps: int = 5, step_size: float = 0.01, noise_scale: float = 0.3):
        self.num_steps = num_steps
        self.step_size = step_size
        self.noise_scale = noise_scale
    
    def sample(self, x: Tensor, energy_grad_fn, timestep_scale: float = 1.0) -> Tuple[Tensor, Dict]:
        raise NotImplementedError


class ULASampler(MCMCSampler):
    """Unadjusted Langevin Algorithm - simplest MCMC sampler."""
    
    def sample(self, x: Tensor, energy_grad_fn, timestep_scale: float = 1.0) -> Tuple[Tensor, Dict]:
        diagnostics = {'sampler': 'ula', 'steps': [], 'accepted': self.num_steps}
        
        effective_step = self.step_size * timestep_scale
        
        for k in range(self.num_steps):
            energy, grad, diag = energy_grad_fn(x)
            
            # Normalize gradient to unit norm for stability
            grad_norm = grad.norm()
            if grad_norm > 1e-8:
                grad = grad / grad_norm
            
            # Langevin update: x = x - step_size * grad + sqrt(2 * step_size) * noise
            noise = torch.randn_like(x) * self.noise_scale
            x = x - effective_step * grad + (2 * effective_step) ** 0.5 * noise
            
            diagnostics['steps'].append({
                'k': k,
                'energy': energy.item(),
                'grad_norm': grad_norm.item(),
                'delta': diag.get('delta', 0),
            })
        
        return x, diagnostics


class MALASampler(MCMCSampler):
    """Metropolis-Adjusted Langevin Algorithm - adds accept/reject step."""
    
    def sample(self, x: Tensor, energy_grad_fn, timestep_scale: float = 1.0) -> Tuple[Tensor, Dict]:
        diagnostics = {'sampler': 'mala', 'steps': [], 'accepted': 0}
        
        effective_step = self.step_size * timestep_scale
        energy_curr, grad_curr, diag_curr = energy_grad_fn(x)
        
        # Normalize gradient
        grad_norm = grad_curr.norm()
        if grad_norm > 1e-8:
            grad_curr = grad_curr / grad_norm
        
        for k in range(self.num_steps):
            # Propose new state
            noise = torch.randn_like(x) * self.noise_scale
            x_prop = x - effective_step * grad_curr + (2 * effective_step) ** 0.5 * noise
            
            # Compute energy and gradient at proposed state
            energy_prop, grad_prop, diag_prop = energy_grad_fn(x_prop)
            grad_prop_norm = grad_prop.norm()
            if grad_prop_norm > 1e-8:
                grad_prop = grad_prop / grad_prop_norm
            
            # Compute acceptance probability (simplified MH ratio)
            log_alpha = -(energy_prop - energy_curr)
            alpha = torch.exp(torch.clamp(log_alpha, max=0.0))
            
            # Accept/reject
            if torch.rand(1, device=x.device) < alpha:
                x = x_prop
                energy_curr = energy_prop
                grad_curr = grad_prop
                diag_curr = diag_prop
                diagnostics['accepted'] += 1
                accepted = True
            else:
                accepted = False
            
            diagnostics['steps'].append({
                'k': k,
                'energy': energy_curr.item(),
                'alpha': alpha.item(),
                'accepted': accepted,
                'delta': diag_curr.get('delta', 0),
            })
        
        return x, diagnostics


class UHMCSampler(MCMCSampler):
    """Unadjusted Hamiltonian Monte Carlo - uses momentum for better exploration."""
    
    def __init__(self, num_steps: int = 5, step_size: float = 0.01, 
                 noise_scale: float = 0.3, leapfrog_steps: int = 3):
        super().__init__(num_steps, step_size, noise_scale)
        self.leapfrog_steps = leapfrog_steps
    
    def sample(self, x: Tensor, energy_grad_fn, timestep_scale: float = 1.0) -> Tuple[Tensor, Dict]:
        diagnostics = {'sampler': 'uhmc', 'steps': [], 'accepted': self.num_steps}
        
        effective_step = self.step_size * timestep_scale
        
        for k in range(self.num_steps):
            # Sample momentum
            p = torch.randn_like(x) * self.noise_scale
            
            # Leapfrog integration
            energy, grad, diag = energy_grad_fn(x)
            grad_norm = grad.norm()
            if grad_norm > 1e-8:
                grad = grad / grad_norm
            
            # Half step for momentum
            p = p - 0.5 * effective_step * grad
            
            # Full steps
            for _ in range(self.leapfrog_steps - 1):
                x = x + effective_step * p
                energy, grad, diag = energy_grad_fn(x)
                grad_norm = grad.norm()
                if grad_norm > 1e-8:
                    grad = grad / grad_norm
                p = p - effective_step * grad
            
            # Final position step
            x = x + effective_step * p
            
            # Half step for momentum
            energy, grad, diag = energy_grad_fn(x)
            grad_norm = grad.norm()
            if grad_norm > 1e-8:
                grad = grad / grad_norm
            p = p - 0.5 * effective_step * grad
            
            diagnostics['steps'].append({
                'k': k,
                'energy': energy.item(),
                'grad_norm': grad_norm.item(),
                'delta': diag.get('delta', 0),
            })
        
        return x, diagnostics


def create_sampler(sampler_type: str, **kwargs) -> MCMCSampler:
    """Factory function for MCMC samplers."""
    samplers = {
        'ula': ULASampler,
        'mala': MALASampler,
        'uhmc': UHMCSampler,
    }
    if sampler_type not in samplers:
        raise ValueError(f"Unknown sampler: {sampler_type}. Choose from {list(samplers.keys())}")
    return samplers[sampler_type](**kwargs)


# ============================================================================
#                           SPATIAL CONSTRAINT
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
        return +1 if self.relation in ("right", "below") else -1


# ============================================================================
#                        ATTENTION PROCESSOR
# ============================================================================

class SpatialGuidanceAttnProcessor:
    """
    Custom attention processor for SD3 that captures I2T attention.
    Handles the joint self-attention in MM-DiT properly.
    """
    
    def __init__(self, block_idx: int, capture: bool = True):
        self.block_idx = block_idx
        self.capture = capture
        self.stored_i2t = None
    
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size = hidden_states.shape[0]
        
        # SD3 MM-DiT: encoder_hidden_states = text, hidden_states = image
        if encoder_hidden_states is not None:
            num_text = encoder_hidden_states.shape[1]
            num_image = hidden_states.shape[1]
            
            # Compute Q, K, V for both text and image
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
        
        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        
        # Apply QK normalization (SD3 specific)
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
        
        # Compute attention weights
        scale = head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Capture I2T attention (image queries attending to text keys)
        if self.capture and num_text > 0:
            i2t = attn_weights[:, :, num_text:, :num_text]  # [B, heads, img, txt]
            self.stored_i2t = i2t.mean(dim=1)  # [B, img, txt] - average over heads
        else:
            self.stored_i2t = None
        
        # Compute attention output
        hidden_out = torch.matmul(attn_weights, value)
        hidden_out = hidden_out.transpose(1, 2).reshape(batch_size, -1, inner_dim)
        hidden_out = hidden_out.to(query.dtype)
        
        # Output projection
        if encoder_hidden_states is not None:
            enc_out = hidden_out[:, :num_text]
            hid_out = hidden_out[:, num_text:]
            
            hid_out = attn.to_out[0](hid_out)
            if len(attn.to_out) > 1:
                hid_out = attn.to_out[1](hid_out)
            
            if hasattr(attn, 'to_add_out') and attn.to_add_out is not None:
                enc_out = attn.to_add_out(enc_out)
            else:
                enc_out = attn.to_out[0](enc_out)
                if len(attn.to_out) > 1:
                    enc_out = attn.to_out[1](enc_out)
            
            return hid_out, enc_out
        else:
            hidden_out = attn.to_out[0](hidden_out)
            if len(attn.to_out) > 1:
                hidden_out = attn.to_out[1](hidden_out)
            return hidden_out


# ============================================================================
#                            TOKEN FINDER
# ============================================================================

class TokenFinder:
    """Finds token indices for phrases in prompts."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def find(self, prompt: str, phrase: str) -> List[int]:
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
        
        return indices if indices else [1]


# ============================================================================
#                          LOSS COMPUTATION
# ============================================================================

def compute_spatial_loss(
    attention: Tensor,
    tokens_A: List[int],
    tokens_B: List[int],
    constraint: SpatialConstraint,
    threshold_percentile: float = 0.7,
    margin: float = 0.1,
) -> Tuple[Tensor, Dict]:
    """
    Compute spatial guidance loss from attention map.
    Uses centroid-based approach for simplicity.
    """
    if attention.dim() == 2:
        attention = attention.unsqueeze(0)
    
    B, num_spatial, num_tokens = attention.shape
    H = W = int(num_spatial ** 0.5)
    device = attention.device
    
    # Create coordinate grids
    y_coords = torch.linspace(0, 1, H, device=device).view(H, 1).expand(H, W)
    x_coords = torch.linspace(0, 1, W, device=device).view(1, W).expand(H, W)
    
    # Extract attention for each object
    attn_A = attention[:, :, tokens_A].mean(dim=-1).view(B, H, W)
    attn_B = attention[:, :, tokens_B].mean(dim=-1).view(B, H, W)
    
    # Apply soft thresholding
    if threshold_percentile > 0:
        for attn in [attn_A, attn_B]:
            flat = attn.view(B, -1)
            thresh = torch.quantile(flat.float(), threshold_percentile, dim=1, keepdim=True)
            thresh = thresh.to(attn.dtype).view(B, 1, 1)
            mask = torch.sigmoid(50.0 * (attn - thresh))
            attn.mul_(mask)
    
    # Normalize
    attn_A = attn_A / (attn_A.sum(dim=(-2, -1), keepdim=True) + 1e-8)
    attn_B = attn_B / (attn_B.sum(dim=(-2, -1), keepdim=True) + 1e-8)
    
    # Compute centroids
    cx_A = (attn_A * x_coords).sum(dim=(-2, -1))
    cy_A = (attn_A * y_coords).sum(dim=(-2, -1))
    cx_B = (attn_B * x_coords).sum(dim=(-2, -1))
    cy_B = (attn_B * y_coords).sum(dim=(-2, -1))
    
    # Spatial loss
    if constraint.axis == "x":
        delta = constraint.sign * (cx_A - cx_B)
    else:
        delta = constraint.sign * (cy_A - cy_B)
    
    L_spatial = F.relu(margin - delta).mean()
    
    # Presence loss (variance)
    var_A = (attn_A * ((x_coords - cx_A.view(-1, 1, 1))**2 + 
                       (y_coords - cy_A.view(-1, 1, 1))**2)).sum(dim=(-2, -1))
    var_B = (attn_B * ((x_coords - cx_B.view(-1, 1, 1))**2 + 
                       (y_coords - cy_B.view(-1, 1, 1))**2)).sum(dim=(-2, -1))
    L_presence = (var_A + var_B).mean()
    
    # Balance loss
    L_balance = torch.abs(var_A - var_B).mean()
    
    # Total loss
    L_total = L_spatial + 0.5 * L_presence + 0.5 * L_balance
    
    diagnostics = {
        'L_total': L_total.item(),
        'L_spatial': L_spatial.item(),
        'delta': delta.mean().item(),
        'cx_A': cx_A.mean().item(),
        'cy_A': cy_A.mean().item(),
        'cx_B': cx_B.mean().item(),
        'cy_B': cy_B.mean().item(),
    }
    
    return L_total, diagnostics


# ============================================================================
#                        MAIN GENERATION FUNCTION
# ============================================================================

def generate_with_mcmc_guidance(
    pipe,
    prompt: str,
    constraint: SpatialConstraint,
    # MCMC settings
    sampler_type: str = "ula",
    mcmc_steps: int = 5,
    mcmc_step_size: float = 0.01,
    mcmc_noise_scale: float = 0.3,
    # Guidance settings
    guidance_strength: float = 30.0,
    guidance_timestep_fraction: float = 0.4,
    threshold_percentile: float = 0.7,
    margin: float = 0.1,
    # Generation settings
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
) -> Tuple[Image.Image, Dict]:
    """Generate image with MCMC-enhanced spatial guidance."""
    
    device = pipe.device
    dtype = pipe.transformer.dtype
    
    print(f"\n{'='*60}")
    print(f"MCMC-Enhanced SD3 Spatial Guidance")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Constraint: {constraint.object_A} {constraint.relation} {constraint.object_B}")
    print(f"Sampler: {sampler_type}, steps: {mcmc_steps}")
    print(f"{'='*60}\n")
    
    # Initialize MCMC sampler
    sampler = create_sampler(
        sampler_type,
        num_steps=mcmc_steps,
        step_size=mcmc_step_size,
        noise_scale=mcmc_noise_scale,
    )
    
    # Find token indices
    token_finder = TokenFinder(pipe.tokenizer)
    tokens_A = token_finder.find(prompt, constraint.object_A)
    tokens_B = token_finder.find(prompt, constraint.object_B)
    
    print(f"Token indices: {constraint.object_A}={tokens_A}, {constraint.object_B}={tokens_B}")
    
    # Pipeline components
    transformer = pipe.transformer
    scheduler = pipe.scheduler
    vae = pipe.vae
    
    # Encode prompt
    (prompt_embeds, neg_prompt_embeds,
     pooled_embeds, neg_pooled_embeds) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt="",
        do_classifier_free_guidance=guidance_scale > 1,
    )
    
    if guidance_scale > 1:
        prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
        pooled_embeds = torch.cat([neg_pooled_embeds, pooled_embeds], dim=0)
    
    # Initialize latents
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = pipe.prepare_latents(
        1, transformer.config.in_channels, height, width,
        prompt_embeds.dtype, device, generator
    )
    
    # Setup scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    max_guidance_steps = int(len(timesteps) * guidance_timestep_fraction)
    
    print(f"Guidance steps: {max_guidance_steps}/{len(timesteps)}")
    
    # Attention capture setup
    measurement_blocks = [4, 5, 6, 7]
    processors: Dict[int, SpatialGuidanceAttnProcessor] = {}
    original_forwards = {}
    
    def make_patched_forward(block_idx: int):
        proc = SpatialGuidanceAttnProcessor(block_idx, capture=True)
        processors[block_idx] = proc
        
        def patched_forward(hidden_states, encoder_hidden_states=None, 
                           attention_mask=None, **kwargs):
            attn = transformer.transformer_blocks[block_idx].attn
            return proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        
        return patched_forward
    
    all_diagnostics = []
    
    try:
        for i, t in enumerate(timesteps):
            apply_guidance = i < max_guidance_steps
            
            # Install attention capture for guidance steps
            if apply_guidance:
                processors.clear()
                for block_idx in measurement_blocks:
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
            timestep_batch = t.expand(latent_input.shape[0])
            
            if apply_guidance:
                latents = latents.detach().requires_grad_(True)
                latent_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
                
                # Forward pass to get attention
                velocity = transformer(
                    hidden_states=latent_input,
                    timestep=timestep_batch,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]
                
                # Collect attention
                attention_store = {idx: proc.stored_i2t for idx, proc in processors.items() 
                                  if proc.stored_i2t is not None}
                
                if attention_store:
                    attn_list = [attention_store[idx] for idx in measurement_blocks 
                                if idx in attention_store]
                    
                    if attn_list:
                        aggregated = torch.stack(attn_list).mean(dim=0)
                        batch_idx = min(1, aggregated.shape[0] - 1)
                        attn = aggregated[batch_idx]
                        
                        # Create energy function for MCMC
                        def energy_grad_fn(lat):
                            lat = lat.clone().requires_grad_(True)
                            loss, diag = compute_spatial_loss(
                                attn, tokens_A, tokens_B, constraint,
                                threshold_percentile, margin,
                            )
                            grad = torch.autograd.grad(loss, lat, retain_graph=False)[0]
                            return loss.detach(), grad.detach(), diag
                        
                        # MCMC sampling
                        timestep_scale = 1.0 + (i / max_guidance_steps) * 0.5
                        new_latents, mcmc_diag = sampler.sample(
                            latents.detach(),
                            energy_grad_fn,
                            timestep_scale=timestep_scale,
                        )
                        
                        # Apply guidance with strength scaling
                        latents = latents.detach() + guidance_strength * (new_latents - latents.detach())
                        
                        # Store diagnostics
                        step_diag = {
                            'step': i,
                            'mcmc_sampler': mcmc_diag['sampler'],
                            'mcmc_accepted': mcmc_diag.get('accepted', mcmc_steps),
                        }
                        if mcmc_diag['steps']:
                            step_diag['final_energy'] = mcmc_diag['steps'][-1]['energy']
                            step_diag['final_delta'] = mcmc_diag['steps'][-1]['delta']
                        all_diagnostics.append(step_diag)
                        
                        correct = (constraint.sign * step_diag.get('final_delta', 0)) > margin
                        print(f"Step {i}: E={step_diag.get('final_energy', 0):.4f}, "
                              f"Δ={step_diag.get('final_delta', 0):.3f} "
                              f"{'✓' if correct else '✗'}, "
                              f"acc={step_diag['mcmc_accepted']}/{mcmc_steps}")
                
                latents = latents.detach()
                
                # Recompute velocity after guidance
                with torch.no_grad():
                    latent_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents
                    velocity = transformer(
                        hidden_states=latent_input,
                        timestep=timestep_batch,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_embeds,
                        return_dict=False,
                    )[0]
            else:
                # Standard inference step
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
    
    # Decode latents
    with torch.no_grad():
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    
    return image, {'steps': all_diagnostics}


# ============================================================================
#                               MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="MCMC-Enhanced SD3 Spatial Guidance")
    parser.add_argument("--prompt", type=str, default="a bird below a fish")
    parser.add_argument("--object_a", type=str, default="bird")
    parser.add_argument("--object_b", type=str, default="fish")
    parser.add_argument("--relation", type=str, default="below", 
                        choices=["left", "right", "above", "below"])
    parser.add_argument("--sampler", type=str, default="ula",
                        choices=["ula", "mala", "uhmc"])
    parser.add_argument("--mcmc_steps", type=int, default=5)
    parser.add_argument("--mcmc_step_size", type=float, default=0.01)
    parser.add_argument("--mcmc_noise_scale", type=float, default=0.3)
    parser.add_argument("--guidance_strength", type=float, default=30.0)
    parser.add_argument("--guidance_fraction", type=float, default=0.4)
    parser.add_argument("--num_steps", type=int, default=28)
    parser.add_argument("--cfg_scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./outputs_mcmc")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pipeline
    print("Loading SD3 pipeline...")
    from diffusers import StableDiffusion3Pipeline
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.bfloat16,
    )
    pipe = pipe.to("cuda")
    
    # Create constraint
    constraint = SpatialConstraint(
        object_A=args.object_a,
        object_B=args.object_b,
        relation=args.relation,
    )
    
    # Generate
    image, diagnostics = generate_with_mcmc_guidance(
        pipe=pipe,
        prompt=args.prompt,
        constraint=constraint,
        sampler_type=args.sampler,
        mcmc_steps=args.mcmc_steps,
        mcmc_step_size=args.mcmc_step_size,
        mcmc_noise_scale=args.mcmc_noise_scale,
        guidance_strength=args.guidance_strength,
        guidance_timestep_fraction=args.guidance_fraction,
        num_inference_steps=args.num_steps,
        guidance_scale=args.cfg_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    
    # Save outputs
    safe_prompt = args.prompt.replace(" ", "_")[:30]
    filename = f"mcmc_{args.sampler}_{safe_prompt}_seed{args.seed}"
    
    image_path = os.path.join(args.output_dir, f"{filename}.png")
    image.save(image_path)
    print(f"\nSaved image: {image_path}")
    
    diag_path = os.path.join(args.output_dir, f"{filename}_diagnostics.json")
    with open(diag_path, 'w') as f:
        json.dump({
            'prompt': args.prompt,
            'constraint': f"{args.object_a} {args.relation} {args.object_b}",
            'sampler': args.sampler,
            'mcmc_steps': args.mcmc_steps,
            'seed': args.seed,
            'diagnostics': diagnostics,
        }, f, indent=2)
    print(f"Saved diagnostics: {diag_path}")


if __name__ == "__main__":
    main()