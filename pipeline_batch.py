import argparse
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import diffusers
import numpy as np
from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from torchvision.ops import batched_nms
from transformers.models.esm.openfold_utils.tensor_utils import batched_gather

from self_guide_combined import SelfGuidanceEdits
from functools import partial
from attn_processor import SelfGuidanceAttnProcessor2_0, SelfGuidanceAttnProcessor
from diffusers.models.attention_processor import Attention
import multiprocessing as mp
import torch
import psutil
import subprocess
from split_data_multiprocessing import *


def log_memory_usage():
    # CPU RAM usage
    process = psutil.Process(os.getpid())
    ram_usage_gb = process.memory_info().rss / (1024 ** 3)  # in GB

    # GPU VRAM usage
    if torch.cuda.is_available():
        vram_usage_gb = torch.cuda.memory_allocated() / (1024 ** 3)  # current usage
        max_vram_usage_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)  # peak usage
    else:
        vram_usage_gb = 0
        max_vram_usage_gb = 0

    print(
        f"[MEMORY USAGE] CPU RAM: {ram_usage_gb:.2f} GB | GPU VRAM: {vram_usage_gb:.2f} GB (peak {max_vram_usage_gb:.2f} GB)")

    # Optionally also call nvidia-smi and log it
    try:
        print(subprocess.check_output(["nvidia-smi"], encoding="utf-8"))
    except Exception as e:
        print(f"Could not run nvidia-smi: {e}")


def search_sequence_numpy(arr, seq):
    Na, Nseq = arr.size, seq.size

    r_seq = np.arange(Nseq)

    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    if M.any() > 0:
        return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
    else:
        return []  # No match found


def resave_aux_key(module, *args, old_key="attn", new_key="last_attn"):
    module._aux[new_key] = module._aux[old_key]


def stash_to_aux(module, args, kwargs, output, mode, key="last_feats", args_idx=None, kwargs_key=None,
                 fn_to_run=None, save_aux=True):
    to_save = None
    if mode == "args":
        to_save = input
        if args_idx is not None: to_save = args[args_idx]
    elif mode == "kwargs":
        assert kwargs_key is not None
        to_save = kwargs[kwargs_key]
    elif mode == "output":
        to_save = output
    if fn_to_run is not None: to_save = fn_to_run(to_save)
    try:
        # print("Stash_to_aux call", save_aux)

        if not save_aux:
            len_ = len(module._aux[key])
            del module._aux[key]
            module._aux[key] = [None] * len_ + [to_save]
        else:
            module._aux[key][-1] = module._aux[key][-1].cpu()
            module._aux[key].append(to_save)
    except:
        try:
            del module._aux[key]
        except:
            pass
        module._aux = {key: [to_save]}


def set_attention_processors(pipe, attn_greenlist, save_aux=False):
    print("Setting attention processors save_aux =", save_aux)

    # attention processors on layers:
    # up_blocks.0.attentions.1.transformer_blocks.1.attn2
    # up_blocks.0.attentions.1.transformer_blocks.2.attn2
    # up_blocks.0.attentions.1.transformer_blocks.3.attn2

    attn_layers = []
    for name, block in pipe.unet.named_modules():
        if isinstance(block, (
                diffusers.models.unets.unet_2d_blocks.CrossAttnDownBlock2D,
                diffusers.models.unets.unet_2d_blocks.CrossAttnUpBlock2D,
                diffusers.models.unets.unet_2d_blocks.UNetMidBlock2DCrossAttn)):
            for attn_name, attn in block.named_modules():
                full_name = name + '.' + attn_name

                # if "attn2" in attn_name and attn_name[-5:] == "attn2":
                #     print(f"Found attention processor: {full_name}")
                #     attn_layers.append(full_name)

                if 'attn2' not in attn_name or (attn_greenlist and full_name not in attn_greenlist):
                    # print(f"Skipping {full_name}")
                    continue
                # else:
                #     print(f"Processing {full_name}")
                if isinstance(attn, diffusers.models.attention_processor.Attention):
                    if isinstance(attn.processor, diffusers.models.attention_processor.AttnProcessor2_0):
                        attn.processor = SelfGuidanceAttnProcessor2_0(save_aux=save_aux)
                    elif isinstance(attn.processor, diffusers.models.attention_processor.AttnProcessor):
                        attn.processor = SelfGuidanceAttnProcessor()
                    else:
                        raise NotImplementedError(
                            f"Self-guidance is not implemented for this attention processor: {attn.processor}")

    # # save the values in a file each layer on a new license
    # with open("attn_layers.txt", "w") as f:
    #     for layer in attn_layers:
    #         f.write(layer + "\n")

    ### !!!!!!!!!!!!!!!!!!!
    # TODO: this is important to preserve the appearance but we are not doing it right now because we are not calling the functions that require it
    # pipe.unet.up_blocks[2].register_forward_hook(partial(stash_to_aux, mode="kwargs", save_aux=save_aux, kwargs_key="hidden_states"), with_kwargs=True)
    pipe.unet.up_blocks[2].register_forward_hook(partial(stash_to_aux, mode="output", save_aux=save_aux),
                                                 with_kwargs=True)

    # base.unet.up_blocks[1].resnets[1].conv2.register_forward_hook(partial(stash_to_aux,mode="args", args_idx=0), with_kwargs=True)
    pipe.unet.up_blocks[0].attentions[1].transformer_blocks[3].attn2.register_forward_hook(resave_aux_key)


def init_pipeline(device):
    base = SelfGuidanceSDXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True, torch_dtype=torch.float16,
        use_onnx=False  # variant="fp16",
    ).to(device)
    print('Using DDPM as scheduler.')
    base.scheduler = diffusers.DDPMScheduler.from_config(base.scheduler.config)
    return base


class SelfGuidanceSDXLPipeline(StableDiffusionXLPipeline):
    # TODO: here they create the dictionary where they store the attention maps
    # TODO: with .chunk(2)[1] they are getting the conditional term of the attention maps ([0] is always the unconditional from classifier-free guidance)
    def get_sg_aux(self, cfg=True, transpose=True):
        aux = defaultdict(dict)
        for name, aux_module in self.unet.named_modules():
            try:
                module_aux = aux_module._aux  # for each layer, retrieve the aux info
                if transpose:
                    for k, v in module_aux.items():
                        if cfg:
                            v = torch.utils._pytree.tree_map(lambda vv: vv.chunk(2)[1] if vv is not None else None, v)
                        aux[k][name] = v  # getting the conditional term with chunk
                else:
                    aux[name] = module_aux
                    if cfg:
                        aux[name] = {
                            k: torch.utils._pytree.tree_map(lambda vv: vv.chunk(2)[1] if vv is not None else None, v)
                            for k, v in aux[name].items()}
            except AttributeError:
                pass
        return aux

    def wipe_sg_aux(self):
        # WIPING OUT ONLY ATTENTION CONTENT (not the last feats)
        for name, aux_module in self.unet.named_modules():
            try:
                del aux_module._aux
            except AttributeError:
                pass

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            sg_grad_wt=1.0,
            sg_edits=None,
            sg_loss_rescale=1000.0,  # prevent fp16 underflow
            debug=False,
            sg_t_start=0,
            sg_t_end=-1,
            save_aux=False,
            L2_norm=False,
            logger=None,
            visualize_attn_maps=False,
            save_attn_maps=False,
            cluster_objects=False,
            self_guidance_mode=False,
            loss_type=None,
            loss_num=1,
            margin=0.5,
            plot_centroid=False,
            two_objects=False,
            relationship=None,
            update_latents=False,
    ):
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        do_self_guidance = sg_grad_wt > 0 and sg_edits is not None

        if do_self_guidance:
            for (prompt_text, edits_dict) in zip(prompt, sg_edits):
                prompt_text_ids = self.tokenizer(prompt_text, return_tensors='np')['input_ids'][0]
                for edit_key, edits in sg_edits.items():
                    for edit in edits:
                        if 'words' not in edit:
                            edit['idxs'] = np.arange(len(prompt_text_ids))
                        else:
                            words = edit['words']
                            if not isinstance(words, list): words = [words]
                            idxs = []
                            for word in words:
                                word_ids = self.tokenizer(word, return_tensors='np')['input_ids']
                                word_ids = word_ids[word_ids < 49406]
                                idxs.append(search_sequence_numpy(prompt_text_ids, word_ids))
                            edit['idxs'] = np.concatenate(idxs)

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)

        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size, # * num_images_per_prompt
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds

        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        # THIS DELETES EVERYTHING FROM THE ATTN MAPS
        self.wipe_sg_aux()
        torch.cuda.empty_cache()

        if sg_t_end < 0:
            sg_t_end = len(timesteps)

        if self_guidance_mode:
            first_steps = 3 * num_inference_steps // 16
            remaining = 25 * num_inference_steps // 32
            self_guidance_alternate_steps = list(range(first_steps, first_steps + remaining + 1, 2))

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                if not (sg_grad_wt > 0 and sg_edits is not None):
                    do_self_guidance = False  # base sdxl
                elif self_guidance_mode and i > sg_t_end and i + 1 not in self_guidance_alternate_steps:
                    do_self_guidance = False  # self-guidance when we don't apply it
                elif sg_t_start <= i < sg_t_end or (
                        self_guidance_mode and i > sg_t_end and i + 1 in self_guidance_alternate_steps):
                    do_self_guidance = True
                else:
                    do_self_guidance = False

                # expand the latents if we are doing classifier free guidance
                with torch.set_grad_enabled(do_self_guidance):
                    latents.requires_grad_(do_self_guidance)
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]  # here it's running the attention processor and saving the attn maps
                    # print(noise_pred.shape) # torch.Size([2, 4, 128, 128])

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    ### SELF GUIDANCE
                    if logger is not None:
                        logger.info(f"Timestep {i}")

                    if do_self_guidance and (sg_t_start <= i < sg_t_end or i + 1 in self_guidance_alternate_steps):
                        sg_aux = self.get_sg_aux(do_classifier_free_guidance)  # here it's extracting the cond term
                        sg_loss = 0

                        batch_size = latents.shape[0]

                        for b in range(batch_size):
                            prompt_b = prompt[b]
                            edits_b = sg_edits[b]
                            sg_loss_b = 0

                            for edit_key, edits in edits_b.items():  # keys: attn, last_attn, last_feats
                                if isinstance(edit_key, str):
                                    key_aux = sg_aux[edit_key]
                                else:
                                    key_aux = {'': {k: sg_aux[k] for k in edit_key}}

                                for edit in edits:  # dict inside 'attn' & ('last_attn', 'last_feats')
                                    wt = edit.get('weight', 1.)
                                    alpha = edit.get('alpha', 1.)
                                    centorid_type = edit.get('centorid_type', None)
                                    function = edit.get('function', None)
                                    words = edit['words']
                                    if wt:
                                        tgt = edit.get('tgt')
                                        if tgt is not None:
                                            if isinstance(edit_key, str):
                                                tgt = tgt[edit_key]
                                            else:
                                                tgt = {'': {k: tgt[k] for k in edit_key}}
                                        apply_edit = edit['fn']
                                        lst1 = []

                                        for module_name, v in key_aux.items():
                                            v_b = v[b:b+1]
                                            result = apply_edit(v_b, i=i, idxs=edit['idxs'], **edit.get('kwargs', {}),
                                                                tgt=tgt[module_name] if tgt is not None else None,
                                                                L2=L2_norm, two_objects=two_objects,
                                                                plot_centroid=plot_centroid,
                                                                loss_type=loss_type, loss_num=loss_num, alpha=alpha,
                                                                margin=margin,
                                                                self_guidance_mode=self_guidance_mode, objects=words,
                                                                prompt=prompt_b,
                                                                module_name=module_name, relationship=relationship,
                                                                centroid_type=centorid_type)
                                            lst1.extend(result)

                                        edit_loss1 = torch.stack(lst1).mean()
                                        if logger is not None:
                                            logger.info(f"{function}: {edit_loss1.item()}")
                                        sg_loss_b += wt * edit_loss1

                                sg_loss += sg_loss_b

                        sg_grad = torch.autograd.grad(sg_loss_rescale * sg_loss, latents)[0] / sg_loss_rescale

                        # non_zero_values = sg_grad[sg_grad != 0]
                        # print("Num non-zero values:", len(non_zero_values), "/", sg_grad.numel(), "mean", sg_grad.mean().item())

                        if logger is not None:
                            logger.info("Gradient mean: %s", sg_grad.mean().item())

                        if update_latents:
                            latents = latents - sg_grad_wt * sg_grad
                        else:
                            noise_pred = noise_pred + sg_grad_wt * sg_grad

                        assert not noise_pred.isnan().any()
                    latents.detach()
                    ### END SELF GUIDANCE

                # if do_classifier_free_guidance and guidance_rescale > 0.0:
                #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                #     noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        torch.cuda.empty_cache()

        if not save_aux:
            self.wipe_sg_aux()

        latents = latents.detach()
        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)


def self_guidance(pipe, device, attn_greenlist, prompts, all_words, seeds, num_inference_steps, sg_t_start, sg_t_end,
                  sg_grad_wt, sg_loss_rescale, L2_norm, visualize_attn_maps=False, shifts=None, margin=0.5,
                  save_attn_maps=False, num_images_per_prompt=1, relationship="to the left of",
                  save_dir_name="sdxl-self-guidance-1", vocab_spatial=[], cluster_objects=False,
                  loss_num="", alpha=1., self_guidance_mode=False, loss_type="sigmoid",
                  plot_centroid=False, save_aux=False, two_objects=False, weight_combinations=None,
                  do_multiprocessing=False, img_id="", update_latents=False, benchmark=None, centroid_type="sg",
                  batch_size=1):
    print("num_images_per_prompt", num_images_per_prompt)

    if benchmark is not None or do_multiprocessing:
        save_path = os.path.join("images", save_dir_name)
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = ""

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]

        if benchmark == "visor":
            seed = seeds[0]
            generators = [torch.Generator(device=device).manual_seed(seed) for _ in range(batch_size)]

        for i in range(num_images_per_prompt):
            print("batch_prompts", batch_prompts)

            batched_relationships = []
            batched_shifts = []
            for prompt in batch_prompts:
                prompt_relationship = None
                for word in vocab_spatial:
                    if word in prompt:
                        prompt_relationship = word
                        break
                batched_relationships.append(prompt_relationship)

                prompt_shift = []
                if shifts and prompt_relationship is not None:
                    prompt_shift = shifts[prompt_relationship]
                batched_shifts.append(prompt_shift)
            print("relationships: ", batched_relationships)
            print("shifts: ", batched_shifts)

            if do_multiprocessing or benchmark is not None:
                batched_words = [all_words[p] for p in batch_prompts]
            else:
                batched_words = []
                for word_list in all_words:
                    for prompt in batch_prompts:
                        if word_list[0] in prompt:
                            batched_words.append(word_list)
            print("words: ", batched_words)

            if benchmark is not None or do_multiprocessing:
                filenames = [f"{prompt}_{i}.png" for prompt in batch_prompts]
            else:
                filename = f"{batch_prompts[0]}_{img_id}_seed_{seeds[0]}_num_steps_{num_inference_steps}_test.png"

            out_filenames = [os.path.join(save_path, filename) for filename in filenames]

            _, centroid_weight, _, _ = weight_combinations[0]

            sg_edits = []
            for j in range(len(batch_prompts)):
                sg_edits.append({
                    'attn': [
                        {
                            'words': batched_words[j],
                            'fn': SelfGuidanceEdits.centroid,
                            'function': "centroid",
                            'spatial': batched_relationships[j],
                            'alpha': alpha,
                            'centorid_type': centroid_type,
                            'kwargs': {
                                'shifts': batched_shifts[j],
                                'relative': False
                            },
                            'weight': centroid_weight,
                        }
                    ]
                })

            print("SELF-GUIDANCE")
            out = pipe(prompt=batch_prompts, generator=generators, sg_grad_wt=sg_grad_wt, sg_edits=sg_edits,
                       num_inference_steps=num_inference_steps, L2_norm=L2_norm, margin=margin,
                       sg_loss_rescale=sg_loss_rescale, debug=False, sg_t_start=sg_t_start, sg_t_end=sg_t_end,
                       visualize_attn_maps=visualize_attn_maps, save_attn_maps=save_attn_maps,
                       cluster_objects=cluster_objects,
                       self_guidance_mode=self_guidance_mode, loss_type=loss_type, loss_num=int(loss_num),
                       plot_centroid=plot_centroid, save_aux=save_aux, two_objects=two_objects,
                       relationship=relationship, update_latents=update_latents).images
            for img, path in zip(out, out_filenames):
                img.save(path)


def setup_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler which logs messages to 'output_self_guidance.log'
    file_handler = logging.FileHandler(filename, mode='w')  # 'w' to overwrite, 'a' to append
    file_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    return logger


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sdxl")
    parser.add_argument("--benchmark", default=None)

    parser.add_argument("--loss_num", default="1")
    parser.add_argument("--relationship", default="left")
    parser.add_argument("--alpha", default=1)

    parser.add_argument("--loss_type", default=None)
    parser.add_argument("--margin", default=0.5)
    parser.add_argument("--plot_centroid", default=False)
    parser.add_argument("--L2_norm", default=False)

    parser.add_argument("--self_guidance_mode", default=False)
    parser.add_argument("--two_objects", default=False)
    parser.add_argument("--do_multiprocessing", default=False)
    parser.add_argument("--update_latents", default=False)
    parser.add_argument("--img_id", default="")
    parser.add_argument("--json_filename", default=None)
    parser.add_argument("--centroid_type", default="sg")
    parser.add_argument("--batch_size", default=1)

    # t2i-comp-bench
    parser.add_argument("--port", default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--outpath", type=str, default="images",
                        help="Path to output score")  # experiments/t2i-comp-bench-spatial/
    parser.add_argument("--complex", type=bool, default=False, help="Prompt is simple structure or in complex category")
    parser.add_argument("--mode", default="client")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    return args


def run_on_gpu(gpu_id, all_prompts, all_words, attn_greenlist, seeds, num_inference_steps,
               sg_t_start, sg_t_end, sg_grad_wt, sg_loss_rescale,
               L2_norm=False, shifts=[], num_images_per_prompt=1,
               vocab_spatial=[], loss_num=1, alpha=1, loss_type="relu", margin=0.1,
               self_guidance_mode=False, two_objects=False, plot_centroid=False, weight_combinations=None,
               do_multiprocessing=False, img_id="", update_latents=False, save_dir_name="", centroid_type="sg",
               benchmark=None, batch_size=1):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    pipe = init_pipeline(device)

    save_aux = False  # True
    set_attention_processors(pipe, attn_greenlist, save_aux=save_aux)

    self_guidance(pipe, device, attn_greenlist, all_prompts, all_words, seeds, num_inference_steps,
                  sg_t_start, sg_t_end, sg_grad_wt, sg_loss_rescale,
                  L2_norm=L2_norm, shifts=shifts,
                  num_images_per_prompt=num_images_per_prompt,
                  vocab_spatial=vocab_spatial,
                  loss_num=loss_num, alpha=alpha,
                  loss_type=loss_type, margin=margin,
                  self_guidance_mode=self_guidance_mode,
                  plot_centroid=plot_centroid, two_objects=two_objects,
                  weight_combinations=weight_combinations,
                  do_multiprocessing=do_multiprocessing, img_id=img_id,
                  update_latents=update_latents, save_dir_name=save_dir_name,
                  centroid_type=centroid_type, benchmark=benchmark, batch_size=batch_size)


def start_multiprocessing(attn_greenlist, json_filename, seeds,
                          num_inference_steps, sg_t_start, sg_t_end, sg_grad_wt, sg_loss_rescale,
                          L2_norm, shifts, num_images_per_prompt, vocab_spatial,
                          loss_num, alpha, loss_type, margin, self_guidance_mode,
                          two_objects, plot_centroid, weight_combinations,
                          do_multiprocessing, img_id, update_latents, benchmark,
                          save_dir_name, centroid_type, batch_size):
    # MULTIPROCESSING
    # SHELL
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # TODO: UNCOMMENT THIS FOR SIEGER
    # Prepare data for multiprocessing
    split_prompts(num_gpus, benchmark, json_filename)
    prompts_folder = os.path.join('data_splits', f'{benchmark}', f"multiprocessing_{num_gpus}")

    mp.set_start_method('spawn')
    processes = []
    for gpu_id in range(num_gpus):
        print("THIS IS GPU", gpu_id)

        with open(os.path.join(prompts_folder, f'prompts_part_{gpu_id}.json'), 'r') as f:
            all_data = json.load(f)

        if benchmark == "visor":
            prompts = [data['text'] for data in all_data]
            all_words = {data['text']: [data['obj_1_attributes'][0], data["obj_2_attributes"][0]] for data in all_data}
        if benchmark == "t2i":
            prompts = [data['prompt'] for data in all_data]
            all_words = {data['prompt']: [data['objects'][0], data["objects"][1]] for data in all_data}
        if benchmark == "geneval":
            pass

        p = mp.Process(target=run_on_gpu, args=(gpu_id, prompts, all_words, attn_greenlist, seeds,
                                                num_inference_steps, sg_t_start, sg_t_end, sg_grad_wt, sg_loss_rescale,
                                                L2_norm, shifts, num_images_per_prompt, vocab_spatial,
                                                loss_num, alpha, loss_type, margin, self_guidance_mode,
                                                two_objects, plot_centroid, weight_combinations,
                                                do_multiprocessing, img_id, update_latents, save_dir_name,
                                                centroid_type,
                                                benchmark, batch_size))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def generate_images(config):
    # log_memory_usage()

    model = config.model
    device = config.device
    L2_norm = config.L2_norm
    loss_num = config.loss_num
    alpha = float(config.alpha)
    loss_type = config.loss_type
    margin = float(config.margin)
    plot_centroid = bool(config.plot_centroid)
    self_guidance_mode = bool(config.self_guidance_mode)
    two_objects = bool(config.two_objects)
    do_multiprocessing = bool(config.do_multiprocessing)
    update_latents = bool(config.update_latents)
    img_id = config.img_id
    benchmark = config.benchmark
    json_filename = config.json_filename
    centroid_type = config.centroid_type
    batch_size = int(config.batch_size)

    print("L2_norm: ", L2_norm)
    print("self_guidance_mode: ", self_guidance_mode)
    print("two_objects: ", two_objects)
    print("loss_type: ", loss_type)
    print("loss_num: ", loss_num)
    print("margin: ", margin)
    print("alpha: ", alpha)
    print("plot_centroid: ", plot_centroid)
    print("do_multiprocessing: ", do_multiprocessing)
    print("update_latents: ", update_latents)
    print("img_id: ", img_id)
    print("benchmark: ", benchmark)
    print("json_filename: ", json_filename)
    print("centroid_type: ", centroid_type)
    print("batch_size: ", batch_size)

    # MY INTERACTIVE TESTS
    if benchmark is None:
        all_prompts = [
            "a cake to the left of a tv",
            "a cake to the right of a tv",
            "a cake above a tv",
            "a cake below a tv",
            "a person to the left of a bicycle",
            "a potted plant to the left of a clock",
            "a kite to the left of a tv",
            "a skateboard to the right of a sheep",
            "a bottle to the right of a handbag",
            "a giraffe to the right of a book",
            # "a sink to the right of an umbrella",
            # "a snowboard above a car",
            # "an apple above a dog",
            # "a tv above a car",
            # "a bird above a motorcycle",
            # "a toilet below a keyboard",
            # "a suitcase below a donut",
            # "a bowl below a sandwich",
            # "a knife below a spoon",

            # "distant shot of the tokyo tower with a massive sun in the sky",
            # "a small dog sitting in a park",
        ]

        all_words = [
            ["cake", "tv"],
            ["person", "bicycle"],
            ["plant", "clock"],
            ["kite", "tv"],
            ["skateboard", "sheep"],
            ["bottle", "handbag"],
            ["giraffe", "book"],
            ["sink", "umbrella"],
            ["snowboard", "car"],
            ["apple", "dog"],
            ["tv", "car"],
            ["bird", "motorcycle"],
            ["toilet", "keyboard"],
            ["suitcase", "donut"],
            ["bowl", "sandwich"],
            ["knife", "spoon"]
            # ["sun"]
            # ["dog"],
        ]

        seeds = [42]
        num_inference_steps = 64  # 256
        num_images_per_prompt = 1
        save_dir_name = model
        vocab_spatial = ["to the left of", "to the right of", "above", "below"]
        shifts = {
            "to the left of": [(0., 0.5), (1., 0.5)],
            "to the right of": [(1., 0.5), (0., 0.5)],
            "above": [(0.5, 0), (0.5, 1)],
            "below": [(0.5, 1), (0.5, 0)]
        }

    elif benchmark == "t2i":
        with open(os.path.join('json_files', f'{json_filename}.json'), 'r') as f:
            t2i_data = json.load(f)

        all_prompts, all_words = [], {}
        for data in t2i_data:
            prompt = data['prompt']
            all_prompts.append(prompt)
            all_words[prompt] = [data['objects'][0], data["objects"][1]]

        num_images_per_prompt = 10
        seeds = list(range(42, 42 + num_images_per_prompt))
        vocab_spatial = ['on side of', 'next to', 'near', 'on the left of', 'on the right of', 'on the bottom of',
                         'on the top of']
        shifts = {
            "on the left of": [(0., 0.5), (1., 0.5)],
            "on the right of": [(1., 0.5), (0., 0.5)],
            "on the top of": [(0.5, 0), (0.5, 1)],
            "on the bottom of": [(0.5, 1), (0.5, 0)],
            "on side of": [(0.2, 0.5), (0.8, 0.5)],  # left
            "next to": [(0.8, 0.5), (0.2, 0.5)],  # right
            "near": [(0.25, 0.5), (0.75, 0.5)]  # left
        }

    elif benchmark == "visor":
        with open(os.path.join('json_files', f'{json_filename}.json'), 'r') as f:
            visor_data = json.load(f)

        all_prompts = []
        all_words = {}
        for data in visor_data:
            prompt = data['text']
            all_prompts.append(prompt)
            all_words[prompt] = [data['obj_1_attributes'][0], data["obj_2_attributes"][0]]

        seeds = [42]
        vocab_spatial = ["to the left of", "to the right of", "above", "below"]
        num_images_per_prompt = 4
        shifts = {
            "to the left of": [(0., 0.5), (1., 0.5)],
            "to the right of": [(1., 0.5), (0., 0.5)],
            "above": [(0.5, 0), (0.5, 1)],
            "below": [(0.5, 1), (0.5, 0)]
        }

    elif benchmark == "geneval":
        with open(os.path.join('json_files', f'{json_filename}.json'), 'r') as f:
            all_words = json.load(f)
        all_prompts = all_words.keys()
        seeds = [42]
        num_images_per_prompt = 4
        vocab_spatial = ['above', 'below', 'left of', 'right of']
        shifts = {
            "left of": [(0., 0.5), (1., 0.5)],
            "right of": [(1., 0.5), (0., 0.5)],
            "above": [(0.5, 0), (0.5, 1)],
            "below": [(0.5, 1), (0.5, 0)]
        }

    if benchmark is not None:
        save_dir_name = os.path.join(benchmark, f"{model}_{img_id}")
    print("save_dir_name", save_dir_name)

    attn_greenlist = [
        "up_blocks.0.attentions.1.transformer_blocks.1.attn2",
        "up_blocks.0.attentions.1.transformer_blocks.2.attn2",
        "up_blocks.0.attentions.1.transformer_blocks.3.attn2",
    ]

    if update_latents:
        sg_grad_wt = 7.5
        weight_combinations = [(0, 100.0, 0, 0)]
    else:
        sg_grad_wt = 1000.  # weight on self guidance term in sampling
        weight_combinations = [(0, 5.0, 0, 0)]

    sg_loss_rescale = 1000.  # to avoid numerical underflow, scale loss by this amount and then divide gradients after backprop
    sg_t_start = 0
    if loss_type is not None:
        num_inference_steps = 100
        sg_t_end = 25
    else:
        num_inference_steps = 64
        sg_t_end = 3 * num_inference_steps // 16
    print("num_inference_steps", num_inference_steps)

    visualize_attn_maps = False
    save_attn_maps = False
    cluster_objects = False
    relationship = None

    if do_multiprocessing:
        start_multiprocessing(
            attn_greenlist, json_filename, seeds, num_inference_steps,
            sg_t_start, sg_t_end, sg_grad_wt, sg_loss_rescale,
            L2_norm, shifts, num_images_per_prompt, vocab_spatial,
            loss_num, alpha, loss_type, margin, self_guidance_mode,
            two_objects, plot_centroid, weight_combinations,
            do_multiprocessing, img_id, update_latents, benchmark,
            save_dir_name, centroid_type, batch_size)

    else:
        pipe = init_pipeline(device)

        save_aux = False  # True
        set_attention_processors(pipe, attn_greenlist, save_aux=save_aux)

        self_guidance(pipe, device, attn_greenlist, all_prompts, all_words, seeds, num_inference_steps, sg_t_start,
                      sg_t_end, sg_grad_wt,
                      sg_loss_rescale, L2_norm=L2_norm, visualize_attn_maps=visualize_attn_maps, shifts=shifts,
                      save_attn_maps=save_attn_maps, num_images_per_prompt=num_images_per_prompt,
                      relationship=relationship,
                      save_dir_name=save_dir_name, vocab_spatial=vocab_spatial, cluster_objects=cluster_objects,
                      loss_num=loss_num, alpha=alpha, self_guidance_mode=self_guidance_mode,
                      loss_type=loss_type, margin=margin, plot_centroid=plot_centroid, two_objects=two_objects,
                      weight_combinations=weight_combinations, do_multiprocessing=do_multiprocessing, img_id=img_id,
                      update_latents=update_latents, benchmark=benchmark, centroid_type=centroid_type,
                      batch_size=batch_size)


if __name__ == "__main__":
    config = get_config()
    generate_images(config)
