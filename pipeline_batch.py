import torch
import diffusers
import argparse
import os
import json
import multiprocessing as mp

from self_guide_batch import Splign
from split_data_multiprocessing import split_prompts
from models import SpatialLossSDPipeline, SpatialLossSDXLPipeline
from utils.model_utils import set_attention_processors
from utils.model_utils import get_model_id


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


def init_pipeline(device, model_information):
    model, model_id = model_information
    if model == "sdxl":
        pipe = SpatialLossSDXLPipeline.from_pretrained(model_id, 
                                                    use_safetensors=True, torch_dtype=torch.float16, 
                                                    use_onnx=False).to(device)
    if model == "sd1.4" or model == "sd1.5" or model == "sd2.1":
        pipe = SpatialLossSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    if model == "spright":
        pipe = SpatialLossSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True,)
        
    pipe = pipe.to(device)
    pipe.scheduler = diffusers.DDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    return pipe


def self_guidance(pipe, device, attn_greenlist, prompts, all_words, seeds, num_inference_steps, sg_t_start, sg_t_end,
                  sg_grad_wt, sg_loss_rescale, L2_norm, shifts=None, margin=0.5,
                  num_images_per_prompt=1, relationship="to the left of",
                  save_dir_name="sdxl-self-guidance-1", vocab_spatial=[],
                  loss_num="", alpha=1., self_guidance_mode=False, loss_type="sigmoid",
                  plot_centroid=False, save_aux=False, two_objects=False, weight_combinations=None,
                  do_multiprocessing=False, img_id="", update_latents=False, benchmark=None, centroid_type="sg",
                  batch_size=1, model="model_name", run_base=False):
    # print("num_images_per_prompt", num_images_per_prompt)

    if benchmark is not None or do_multiprocessing:
        save_path = os.path.join("images", save_dir_name)
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = ""

    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        # print("batch_prompts", batch_prompts)

        if benchmark == "visor":
            seed = seeds[0]
            generators = [torch.Generator(device=device).manual_seed(seed) for _ in range(len(batch_prompts))]

        for i in range(num_images_per_prompt):
            if benchmark == "t2i" or benchmark == "geneval":
                seed = seeds[i]
                # print("seed", seed)
                generators = [torch.Generator(device=device).manual_seed(seed) for _ in range(len(batch_prompts))]

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
            # print("relationships: ", batched_relationships)
            # print("shifts: ", batched_shifts)

            if do_multiprocessing or benchmark is not None:
                batched_words = [all_words[p] for p in batch_prompts]
            else:
                batched_words = []
                for word_list in all_words:
                    for prompt in batch_prompts:
                        if word_list[0] in prompt:
                            batched_words.append(word_list)
            # print("words: ", batched_words)

            if benchmark is not None or do_multiprocessing:
                filenames = [f"{prompt}_{i}.png" for prompt in batch_prompts]
                # print("filenames", filenames)
            else:
                filenames = [f"{prompt}_{img_id}.png" for prompt in batch_prompts]

            out_filenames = [os.path.join(save_path, filename) for filename in filenames]

            _, centroid_weight, _, _ = weight_combinations[0]

            sg_edits = []
            for j in range(len(batch_prompts)):
                sg_edits.append({
                    'attn': [
                        {
                            'words': batched_words[j],
                            'fn': Splign.centroid,
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

            if run_base:
                print("MODEL", model)
                out = pipe(prompt=batch_prompts, generator=generators, num_inference_steps=num_inference_steps, save_aux=save_aux).images
                base_filenames = [f"{prompt}_{model}_{i}.png" for prompt in batch_prompts]
                base_out_filenames = [os.path.join(save_path, filename) for filename in base_filenames]
                for img, path in zip(out, base_out_filenames):
                    img.save(path)
            # print("SELF-GUIDANCE")
            out = pipe(prompt=batch_prompts, generator=generators, sg_grad_wt=sg_grad_wt, sg_edits=sg_edits,
                       num_inference_steps=num_inference_steps, L2_norm=L2_norm, margin=margin,
                       sg_loss_rescale=sg_loss_rescale, sg_t_start=sg_t_start, sg_t_end=sg_t_end,
                       self_guidance_mode=self_guidance_mode, loss_type=loss_type, loss_num=int(loss_num),
                       plot_centroid=plot_centroid, save_aux=save_aux, two_objects=two_objects,
                       update_latents=update_latents).images
            for img, path in zip(out, out_filenames):
                img.save(path)


def run_on_gpu(gpu_id, all_prompts, all_words, attn_greenlist, seeds, num_inference_steps,
               sg_t_start, sg_t_end, sg_grad_wt, sg_loss_rescale,
               L2_norm=False, shifts=[], num_images_per_prompt=1,
               vocab_spatial=[], loss_num=1, alpha=1, loss_type="relu", margin=0.1,
               self_guidance_mode=False, two_objects=False, plot_centroid=False, weight_combinations=None,
               do_multiprocessing=False, img_id="", update_latents=False, save_dir_name="", centroid_type="sg",
               benchmark=None, batch_size=1, model="model_name"):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    model_information = get_model_id(model)
    pipe = init_pipeline(device, model_information)

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
                  centroid_type=centroid_type, benchmark=benchmark, batch_size=batch_size, model=model)


def start_multiprocessing(attn_greenlist, json_filename, seeds,
                          num_inference_steps, sg_t_start, sg_t_end, sg_grad_wt, sg_loss_rescale,
                          L2_norm, shifts, num_images_per_prompt, vocab_spatial,
                          loss_num, alpha, loss_type, margin, self_guidance_mode,
                          two_objects, plot_centroid, weight_combinations,
                          do_multiprocessing, img_id, update_latents, benchmark,
                          save_dir_name, centroid_type, batch_size, model):
    # MULTIPROCESSING
    # SHELL
    num_gpus = torch.cuda.device_count()
    # print(f"Number of GPUs available: {num_gpus}")

    # TODO: UNCOMMENT THIS FOR SIEGER
    # Prepare data for multiprocessing
    split_prompts(num_gpus, benchmark, json_filename)
    prompts_folder = os.path.join('data_splits', f'{benchmark}', f"multiprocessing_{num_gpus}")

    mp.set_start_method('spawn')
    processes = []
    for gpu_id in range(num_gpus):
        # print("THIS IS GPU", gpu_id)

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
                                                benchmark, batch_size, model))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def generate_images(config):
    # log_memory_usage()

    model = config.model
    model_information = get_model_id(model)
    
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

    if benchmark == "t2i":
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
        num_images_per_prompt = 4
        seeds = list(range(42, 42 + num_images_per_prompt))
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

    if model == "sdxl":
        attn_greenlist = [
            "up_blocks.0.attentions.1.transformer_blocks.1.attn2",
            "up_blocks.0.attentions.1.transformer_blocks.2.attn2",
            "up_blocks.0.attentions.1.transformer_blocks.3.attn2",
        ]
    else:
        # SD 1.5 and SD 2.1 have the same architecture => same up_blocks
        attn_greenlist = [
            "up_blocks.1.attentions.0.transformer_blocks.0.attn2",
            "up_blocks.1.attentions.1.transformer_blocks.0.attn2",
            "up_blocks.1.attentions.2.transformer_blocks.0.attn2",
            "up_blocks.2.attentions.0.transformer_blocks.0.attn2",
            "up_blocks.2.attentions.1.transformer_blocks.0.attn2",
            "up_blocks.2.attentions.2.transformer_blocks.0.attn2",
            "up_blocks.3.attentions.0.transformer_blocks.0.attn2",
            "up_blocks.3.attentions.1.transformer_blocks.0.attn2",
            "up_blocks.3.attentions.2.transformer_blocks.0.attn2"
        ]

    if update_latents:
        sg_grad_wt = 7.5
        weight_combinations = [(0, 100.0, 0, 0)]
    else:
        sg_grad_wt = 1000.  # weight on self guidance term in sampling
        weight_combinations = [(0, 5.0, 0, 0)]

    sg_loss_rescale = 1000.  # to avoid numerical underflow, scale loss by this amount and then divide gradients after backprop
    sg_t_start = 0
    
    if self_guidance_mode:
        num_inference_steps = 64
        sg_t_end = 3 * num_inference_steps // 16
    
    if model == "sdxl":
        num_inference_steps = 50
        sg_t_end = 25
        
    if model == "sd1.4" or model == "sd1.5" or model == "sd2.1" or model == "spright":
        num_inference_steps = 200
        sg_t_end = 25
    print("num_inference_steps", num_inference_steps)
    
    relationship = None

    if do_multiprocessing:
        start_multiprocessing(
            attn_greenlist, json_filename, seeds, num_inference_steps,
            sg_t_start, sg_t_end, sg_grad_wt, sg_loss_rescale,
            L2_norm, shifts, num_images_per_prompt, vocab_spatial,
            loss_num, alpha, loss_type, margin, self_guidance_mode,
            two_objects, plot_centroid, weight_combinations,
            do_multiprocessing, img_id, update_latents, benchmark,
            save_dir_name, centroid_type, batch_size, model)

    else:
        pipe = init_pipeline(device, model_information)

        save_aux = False  # True
        set_attention_processors(pipe, attn_greenlist, save_aux=save_aux)

        run_base = False
        self_guidance(pipe, device, attn_greenlist, all_prompts, all_words, seeds, num_inference_steps, sg_t_start,
                      sg_t_end, sg_grad_wt,
                      sg_loss_rescale, L2_norm=L2_norm, shifts=shifts,
                      num_images_per_prompt=num_images_per_prompt,
                      relationship=relationship,
                      save_dir_name=save_dir_name, vocab_spatial=vocab_spatial,
                      loss_num=loss_num, alpha=alpha, self_guidance_mode=self_guidance_mode,
                      loss_type=loss_type, margin=margin, plot_centroid=plot_centroid, two_objects=two_objects,
                      weight_combinations=weight_combinations, do_multiprocessing=do_multiprocessing, img_id=img_id,
                      update_latents=update_latents, benchmark=benchmark, centroid_type=centroid_type,
                      batch_size=batch_size, model=model, run_base=run_base)


if __name__ == "__main__":
    config = get_config()
    generate_images(config)
