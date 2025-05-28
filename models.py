import torch
import numpy as np
import os

from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, DDPMScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput

from clip_model import ClipTextScorer
from utils.model_utils import search_sequence_numpy, setup_logger

import warnings
warnings.filterwarnings("ignore")


def set_scale(grad, correction=None, target_guidance=None, guidance_scale=None):
    grad_norm = (grad * grad).mean().sqrt().item()
    # print("grad_norm", grad_norm)
    numerator = (correction * correction).mean().sqrt().item()
    target_guidance = numerator * guidance_scale / (grad_norm + 1e-1) * target_guidance
    # print("target_guidance", target_guidance)
    return target_guidance


def predict_x0_from_xt(
        scheduler: DDPMScheduler,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
) -> Union[DDPMSchedulerOutput, Tuple]:
    assert isinstance(scheduler, DDPMScheduler)
    if scheduler.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    t = timestep

    prev_t = scheduler.previous_timestep(t)

    if model_output.shape[1] == sample.shape[1] * 2 and scheduler.variance_type in ["learned", "learned_range"]:
        model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
    else:
        predicted_variance = None

    # 1. compute alphas, betas
    alpha_prod_t = scheduler.alphas_cumprod[t]
    alpha_prod_t_prev = scheduler.alphas_cumprod[prev_t] if prev_t >= 0 else scheduler.one
    beta_prod_t = 1 - alpha_prod_t
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev

    # 2. compute predicted original sample from predicted noise also called
    # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
    elif scheduler.config.prediction_type == "sample":
        pred_original_sample = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t ** 0.5) * sample - (beta_prod_t ** 0.5) * model_output
    else:
        raise ValueError(
            f"prediction_type given as {scheduler.config.prediction_type} must be one of `epsilon`, `sample` or"
            " `v_prediction`  for the DDPMScheduler."
        )

    # 3. Clip or threshold "predicted x_0"
    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif scheduler.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range
        )

    return pred_original_sample.to(dtype=sample.dtype)


class SpatialLossSDXLPipeline(StableDiffusionXLPipeline):
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

    def compute_gradient(self, scorer, prompt, pred_original_sample):
        im_pix_un = self.vae.decode(pred_original_sample.to(self.vae.dtype) / self.vae.config.scaling_factor).sample
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1).to(torch.float).cpu()

        if isinstance(scorer, ClipTextScorer):
            prompts = [prompt] * len(im_pix)
            loss = scorer.loss_fn(im_pix, prompts)
        return loss

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
            original_size: Optional[Tuple[int, int]] = None,
            crops_coords_top_left: Tuple[int, int] = (0, 0),
            target_size: Optional[Tuple[int, int]] = None,
            sg_grad_wt=1.0,
            sg_edits=None,
            sg_loss_rescale=1000.0,  # prevent fp16 underflow
            sg_t_start=0,
            sg_t_end=-1,
            save_aux=False,
            L2_norm=False,
            logger=None,
            self_guidance_mode=False,
            loss_type=None,
            loss_num=1,
            margin=0.5,
            plot_centroid=False,
            two_objects=False,
            update_latents=False,
            img_id=None,
            smoothing=False,
            masked_mean=False,
            grad_norm_scale=False,
            target_guidance=3000,
            clip_weight=1.0,
            use_clip_loss=False,
            object_presence=False,
            masked_mean_thresh=None,
            masked_mean_weight=None,
            write_to_file=False,
            save_dir_name="save_dir",
            use_energy=False
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
                for edit_key, edits in edits_dict.items():
                    for edit in edits:
                        if 'words' not in edit:
                            edit['idxs'] = np.arange(len(prompt_text_ids))
                        else:
                            words = edit['words']
                            if not isinstance(words, list):
                                words = [words]
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
        # print("num_images_per_prompt", num_images_per_prompt)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # print("latents", latents.shape)

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

        scorer = ClipTextScorer()

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
                    # print(noise_pred.shape) # torch.Size([2, 4, 128, 128]) -> 2 for cond+uncond terms

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    ### SELF GUIDANCE
                    if logger is not None:
                        logger.info(f"Timestep {i}")

                    if do_self_guidance and (sg_t_start <= i < sg_t_end or i + 1 in self_guidance_alternate_steps):
                        sg_aux = self.get_sg_aux(do_classifier_free_guidance)  # here it's extracting the cond term
                        spatial_losses = []

                        batch_size = latents.shape[0]

                        for b in range(batch_size):
                            prompt_b = prompt[b]
                            edits_b = sg_edits[b]
                            spatial_loss_b = 0

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
                                    relationship = edit.get('spatial', None)
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
                                            result = apply_edit(v, b, i=i, idxs=edit['idxs'], **edit.get('kwargs', {}),
                                                                tgt=tgt[module_name] if tgt is not None else None,
                                                                L2=L2_norm, two_objects=two_objects,
                                                                plot_centroid=plot_centroid,
                                                                loss_type=loss_type, loss_num=loss_num, alpha=alpha,
                                                                margin=margin,
                                                                self_guidance_mode=self_guidance_mode, objects=words,
                                                                prompt=prompt_b,
                                                                module_name=module_name, relationship=relationship,
                                                                centroid_type=centorid_type,
                                                                img_id=img_id, smoothing=smoothing,
                                                                masked_mean=masked_mean, object_presence=object_presence,
                                                                masked_mean_thresh=masked_mean_thresh, masked_mean_weight=masked_mean_weight)
                                            lst1.extend(result)

                                        edit_loss1 = torch.stack(lst1).mean()
                                        if logger is not None:
                                            logger.info(f"{function}: {edit_loss1.item()}")
                                        spatial_loss_b += wt * edit_loss1

                                # right now there is just one edit dictionary!!!
                            spatial_losses.append(spatial_loss_b)

                        if use_clip_loss:
                            latent_in = latents.detach().requires_grad_(True)
                            pred_original_sample = predict_x0_from_xt(self.scheduler, noise_pred, t, latent_in)

                            obj1, obj2 = clip_objects[0], clip_objects[1]

                            clip_loss_obj1 = self.compute_gradient(scorer, obj1, pred_original_sample)
                            clip_loss_obj2 = self.compute_gradient(scorer, obj2, pred_original_sample)
                            clip_loss = clip_loss_obj1 + clip_loss_obj2

                            clip_grad = torch.autograd.grad(clip_loss, latent_in, retain_graph=True)[0]
                            # print("clip_loss", clip_loss.item())

                        spatial_losses_batch = torch.stack(spatial_losses)
                        spatial_grad = torch.autograd.grad(spatial_losses_batch, latents,
                                                           grad_outputs=torch.ones_like(spatial_losses_batch),
                                                           retain_graph=True)[0]
                        if use_clip_loss:
                            noise_pred = noise_pred + sg_grad_wt * spatial_grad + clip_weight * clip_grad
                        else:
                            noise_pred = noise_pred + sg_grad_wt * spatial_grad

                        assert not noise_pred.isnan().any()
                    latents.detach()
                    ### END SELF GUIDANCE
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


class SpatialLossSDPipeline(StableDiffusionPipeline):

    def get_sg_aux(self, cfg=True, transpose=True):
        aux = defaultdict(dict)
        for name, aux_module in self.unet.named_modules():
            try:
                module_aux = aux_module._aux
                if transpose:
                    for k, v in module_aux.items():
                        if cfg:
                            v = torch.utils._pytree.tree_map(lambda vv: vv.chunk(2)[1] if vv is not None else None, v)
                        aux[k][name] = v
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

    # @torch.enable_grad()
    def compute_gradient(self, scorer, prompt, pred_original_sample):
        im_pix_un = self.vae.decode(pred_original_sample.to(self.vae.dtype) / self.vae.config.scaling_factor).sample
        im_pix = ((im_pix_un / 2) + 0.5).clamp(0, 1).to(torch.float).cpu()

        if isinstance(scorer, ClipTextScorer):
            prompts = [prompt] * len(im_pix)
            loss = scorer.loss_fn(im_pix, prompts)
        return loss

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
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
            sg_t_start=0,
            sg_t_end=-1,
            save_aux=False,
            L2_norm=False,
            logger=None,
            self_guidance_mode=False,
            loss_type=None,
            loss_num=1,
            margin=0.5,
            plot_centroid=False,
            two_objects=False,
            update_latents=False,
            img_id=None,
            smoothing=False,
            masked_mean=False,
            grad_norm_scale=False,
            target_guidance=3000,
            clip_weight=1.0,
            use_clip_loss=False,
            object_presence=False,
            masked_mean_thresh=None,
            masked_mean_weight=None,
            write_to_file=False,
            save_dir_name="save_dir",
            use_energy=False,
            no_wt=False,
            leaky_relu_slope=0.05
    ):
        # 0. Default height and width to unet
        global clip_objects
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        # sd1.5 512 512
        # sd2.1 768 768

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = True  # guidance_scale > 1.0
        do_self_guidance = sg_grad_wt > 0 and sg_edits is not None

        if do_self_guidance:
            for (prompt_text, edits_dict) in zip(prompt, sg_edits):
                prompt_text_ids = self.tokenizer(prompt_text, return_tensors='np')['input_ids'][0]
                for edit_key, edits in edits_dict.items():
                    for edit in edits:
                        if 'words' not in edit:
                            edit['idxs'] = np.arange(len(prompt_text_ids))
                        else:
                            words = edit['words']
                            clip_objects = words
                            if not isinstance(words, list):
                                words = [words]
                            idxs = []
                            for word in words:
                                word_ids = self.tokenizer(word, return_tensors='np')['input_ids']
                                # print(word, len(word_ids))
                                word_ids = word_ids[word_ids < 49406]
                                ids = search_sequence_numpy(prompt_text_ids, word_ids)
                                # print(word, ids)
                                idxs.append(ids)
                            edit['idxs'] = np.concatenate(idxs)
                            # print(words, edit['idxs'])

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
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

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Setup logger
        if write_to_file:
            filename = os.path.basename(save_dir_name)

            logs_folder = "logs_attn"
            os.makedirs(logs_folder, exist_ok=True)

            filename = os.path.join(logs_folder, f"{filename}_{prompt[0]}.log")
            logger = setup_logger(filename=filename)

        self.wipe_sg_aux()
        torch.cuda.empty_cache()
        if sg_t_end < 0:
            sg_t_end = len(timesteps)

        if self_guidance_mode:
            first_steps = 3 * num_inference_steps // 16
            remaining = 25 * num_inference_steps // 32
            self_guidance_alternate_steps = list(range(first_steps, first_steps + remaining + 1, 2))

        scorer = ClipTextScorer()

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

                with torch.set_grad_enabled(do_self_guidance):
                    latents.requires_grad_(do_self_guidance)
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample
                    # print("noise_pred", noise_pred.shape) # sd1.5 - [2, 4, 64, 64] | sd2.1 - [2, 4, 96, 96]

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # TODO: CLIP
                    # pred_original_temp = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).pred_original_sample
                    # rewards = self.compute_scores(pred_original_temp, prompt)

                    ### SELF GUIDANCE
                    if do_self_guidance and (sg_t_start <= i < sg_t_end or i + 1 in self_guidance_alternate_steps):
                        sg_aux = self.get_sg_aux(do_classifier_free_guidance)  # here it's extracting the cond term
                        spatial_losses = []

                        batch_size = latents.shape[0]

                        for b in range(batch_size):
                            prompt_b = prompt[b]
                            edits_b = sg_edits[b]
                            spatial_loss_b = 0

                            for edit_key, edits in edits_b.items():  # keys: attn, last_attn, last_feats
                                if isinstance(edit_key, str):
                                    key_aux = sg_aux[edit_key]
                                else:
                                    key_aux = {'': {k: sg_aux[k] for k in edit_key}}

                                for edit in edits:  # dict inside 'attn' & ('last_attn', 'last_feats')
                                    wt = edit.get('weight', 1.)
                                    # print("wt: ", wt)
                                    alpha = edit.get('alpha', 1.)
                                    # print("alpha: ", alpha)
                                    centorid_type = edit.get('centorid_type', None)
                                    # print("centorid_type: ", centorid_type)
                                    function = edit.get('function', None)
                                    # print("function: ", function)
                                    words = edit['words']
                                    # print("words: ", words)
                                    relationship = edit.get('spatial', None)
                                    # print("relationship: ", relationship)
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
                                            result = apply_edit(v, b, i=i, idxs=edit['idxs'], **edit.get('kwargs', {}),
                                                                tgt=tgt[module_name] if tgt is not None else None,
                                                                L2=L2_norm, two_objects=two_objects,
                                                                plot_centroid=plot_centroid,
                                                                loss_type=loss_type, loss_num=loss_num, alpha=alpha,
                                                                margin=margin,
                                                                self_guidance_mode=self_guidance_mode, objects=words,
                                                                prompt=prompt_b,
                                                                module_name=module_name, relationship=relationship,
                                                                centroid_type=centorid_type,
                                                                img_id=img_id, smoothing=smoothing,
                                                                masked_mean=masked_mean, object_presence=object_presence,
                                                                masked_mean_thresh=masked_mean_thresh, masked_mean_weight=masked_mean_weight,
                                                                use_energy=use_energy, leaky_relu_slope=leaky_relu_slope)
                                            lst1.extend(result)
                                            if logger is not None:
                                                logger.info(f"Timestep {i}, spatial loss: {result[0].item()}, block: {module_name}")

                                        edit_loss1 = torch.stack(lst1).mean()
                                        if no_wt:
                                            spatial_loss_b += edit_loss1
                                        else:
                                            spatial_loss_b += wt *  edit_loss1

                                # right now there is just one edit dictionary!!!
                            spatial_losses.append(spatial_loss_b)

                        if use_clip_loss:
                            latent_in = latents.detach().requires_grad_(True)
                            pred_original_sample = predict_x0_from_xt(self.scheduler, noise_pred, t, latent_in)

                            obj1, obj2 = clip_objects[0], clip_objects[1]
                            # breakpoint()

                            clip_loss_obj1 = self.compute_gradient(scorer, obj1, pred_original_sample)
                            clip_loss_obj2 = self.compute_gradient(scorer, obj2, pred_original_sample)
                            clip_loss = clip_loss_obj1 + clip_loss_obj2

                            clip_grad = torch.autograd.grad(clip_loss, latent_in, retain_graph=True)[0]
                            # print("sg_loss", sg_loss.item())
                            # print("clip_loss", clip_loss.item())

                        # breakpoint()
                        spatial_losses_batch = torch.stack(spatial_losses)
                        spatial_grad = torch.autograd.grad(spatial_losses_batch, latents,
                                                           grad_outputs=torch.ones_like(spatial_losses_batch),
                                                           retain_graph=True)[0]  # no underflow / sg_loss_rescale
                        # # checking the first sample -> [0]
                        # idx = torch.argmin(noise_pred[0])
                        # print("grad at idx", spatial_grad[0].flatten()[idx].item())
                        # print("before", noise_pred[0].flatten()[idx].item())
                        # noise_pred_before = noise_pred.clone()

                        if logger is not None: # noise_pred min: {noise_pred.min().item()}, noise_pred mean: {noise_pred.mean().item()}, noise_pred max: {noise_pred.max().item()}
                            logger.info(
                                f"Timestep {i}, grad min: {sg_grad_wt * spatial_grad.min().item()}, grad mean: {sg_grad_wt * spatial_grad.mean().item()}, grad max: {sg_grad_wt * spatial_grad.max().item()}")

                        if use_clip_loss:
                            noise_pred = noise_pred + sg_grad_wt * spatial_grad + clip_weight * clip_grad
                        else:
                            noise_pred = noise_pred + sg_grad_wt * spatial_grad
                        # noise_pred_after = noise_pred
                        # expected = noise_pred_before + sg_grad_wt * spatial_grad
                        # print("max error", (expected - noise_pred_after).abs().max().item())

                        # for b in range(batch_size):
                        #     expected_b = noise_pred_before[b] + sg_grad_wt * spatial_grad[b]
                        #     actual_b = noise_pred_after[b]
                        #     error = (expected_b - actual_b).abs().max().item()
                        #     print(f"[Sample {b}] max error: {error}")

                        # print("after", noise_pred[0].flatten()[idx].item())

                        # if grad_norm_scale:
                        #     correction = noise_pred_text - noise_pred_uncond
                        #     target_guidance = set_scale(sg_grad, correction, target_guidance, guidance_scale)
                        #     weighted_spatial_grad = target_guidance * sg_grad

                        assert not noise_pred.isnan().any()
                    latents.detach()
                    ### END SELF GUIDANCE

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample # , generator=generator

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        torch.cuda.empty_cache()

        if not save_aux:
            self.wipe_sg_aux()

        latents = latents.detach()

        if output_type == "latent":
            image = latents
            has_nsfw_concept = None
        elif output_type == "pil": # goes in here
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
        else:
            # 8. Post-processing
            image = self.decode_latents(latents)

        ##############
        has_nsfw_concept = False
        ##############

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
