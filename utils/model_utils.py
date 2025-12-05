import torch
import psutil
import subprocess
import diffusers
import numpy as np
import os
import logging

from attn_processor_batch import SelfGuidanceAttnProcessor2_0


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


def get_model_id(model):
    if model == "sd1.4":
        model_id = "CompVis/stable-diffusion-v1-4"
    if model == "sd1.5":
        model_id = "runwayml/stable-diffusion-v1-5"
    if model == "sd2.1":
        model_id = "stabilityai/stable-diffusion-2-1-base"
    if model == "sdxl":
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    if model == "spright":
        model_id = "SPRIGHT-T2I/spright-t2i-sd2"
    if model =="flux":
        model_id="black-forest-labs/FLUX.1-schnell"
    if model=="controlnet":
        model_id="stabilityai/stable-diffusion-xl-base-1.0"
    return model, model_id


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
        if args_idx is not None: 
            to_save = args[args_idx]
    elif mode == "kwargs":
        assert kwargs_key is not None
        to_save = kwargs[kwargs_key]
    elif mode == "output":
        to_save = output
    if fn_to_run is not None: 
        to_save = fn_to_run(to_save)
    try:
        # print("Stash_to_aux call", save_aux)

        if not save_aux:
            len_ = len(module._aux[key])
            del module._aux[key]
            module._aux[key] = [None] * len_ + [to_save]
        else:
            module._aux[key][-1] = module._aux[key][-1].cpu()
            module._aux[key].append(to_save)
    except:  # noqa: E722
        try:
            del module._aux[key]
        except:  # noqa: E722
            pass
        module._aux = {key: [to_save]}


def set_attention_processors(pipe, attn_greenlist, save_aux=False):
    #print("Setting attention processors save_aux =", save_aux)

    # attention processors on layers:
    # up_blocks.0.attentions.1.transformer_blocks.1.attn2
    # up_blocks.0.attentions.1.transformer_blocks.2.attn2
    # up_blocks.0.attentions.1.transformer_blocks.3.attn2

    # attn_layers = []# works for both UNet-based and FLUX-based pipelines
    backbone = (
        getattr(pipe, "transformer", None)
        or pipe.components.get("transformer", None)
        or getattr(pipe, "unet", None)
    )
    if backbone is None:
        raise RuntimeError(f"No transformer/unet found. Components: {list(pipe.components.keys())}")


    
    cnt=0
    for name, block in backbone.named_modules():#pipe.unet.
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
    # pipe.unet.up_blocks[2].register_forward_hook(partial(stash_to_aux, mode="output", save_aux=save_aux),
    #                                              with_kwargs=True)

    # base.unet.up_blocks[1].resnets[1].conv2.register_forward_hook(partial(stash_to_aux,mode="args", args_idx=0), with_kwargs=True)
    # pipe.unet.up_blocks[0].attentions[1].transformer_blocks[3].attn2.register_forward_hook(resave_aux_key)


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