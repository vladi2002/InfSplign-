import torch
import diffusers
import argparse
import os
import json
import multiprocessing as mp

from self_guide_batch import Splign
from split_data_multiprocessing import get_prompts_for_rank
from models import SpatialLossSDPipeline, SpatialLossSDXLPipeline
from utils.model_utils import set_attention_processors
from utils.model_utils import get_model_id

os.environ["HF_HOME"] = "/tudelft.net/staff-umbrella/StudentsCVlab/vchatalbasheva/Thesis-Splign/hf_cache"


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sdxl")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--use_mpi", default=False, action="store_true")

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
    parser.add_argument("--job_id", default=None)
    parser.add_argument("--sweep", default=False, action="store_true")
    
    parser.add_argument("--gaussian_smoothing", default=False)
    parser.add_argument("--masked_mean", default=False)
    parser.add_argument("--masked_mean_thresh", default=0)
    parser.add_argument("--masked_mean_weight", default=0)

    parser.add_argument("--num_inference_steps", default=1)
    parser.add_argument("--sg_t_start", default=0)
    parser.add_argument("--sg_t_end", default=1)
    parser.add_argument("--sg_grad_wt", default=1)
    parser.add_argument("--grad_norm_scale", default=False)
    parser.add_argument("--target_guidance", default=3000)
    parser.add_argument("--clip_weight", default=1.0)
    parser.add_argument("--use_clip_loss", default=False)
    parser.add_argument("--num_attn_layers", default=9)
    parser.add_argument("--object_presence", default=False)
    parser.add_argument("--write_to_file", default=False)
    parser.add_argument("--use_energy", default=False)
    
    parser.add_argument("--num_images_per_prompt", default=4)
    parser.add_argument("--no_wt", default=False)
    parser.add_argument("--leaky_relu_slope", default=0.05)

    # t2i-comp-bench
    parser.add_argument("--port", default=2)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--outpath", type=str, default="images", help="Path to output score")  # experiments/t2i-comp-bench-spatial/
    parser.add_argument("--complex", type=bool, default=False, help="Prompt is simple structure or in complex category")
    parser.add_argument("--mode", default="client")

    args = parser.parse_args()
    
    if args.use_mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        node_name = MPI.Get_processor_name()

        # Get the unique node names and assign a unique ID to each node
        node_names = comm.allgather(node_name)
        unique_nodes = list(set(node_names))
        unique_nodes.sort()
        print(node_name, node_names, unique_nodes)
        node_id = unique_nodes.index(node_name)
        print("Node id: ", node_id)
    else:
        rank = 0
        size = 1


    if torch.cuda.is_available():
        num_gpus_per_node = torch.cuda.device_count()
        device_id = rank % num_gpus_per_node
        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')
        print("Rank: {}, Size: {}, Device: {}".format(rank, size, device))
        args.rank = rank
        args.world_size = size
    else:
        device = "cpu"

    args.device = device
    return args


def init_pipeline(device, model_information):
    model, model_id = model_information
    if model == "sdxl":
        pipe = SpatialLossSDXLPipeline.from_pretrained(model_id, use_safetensors=True, torch_dtype=torch.float16, use_onnx=False)
    if model == "sd1.4" or model == "sd1.5" or model == "sd2.1":
        pipe = SpatialLossSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    if model == "spright":
        pipe = SpatialLossSDPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True)
        
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
                  batch_size=1, model="model_name", run_base=False, smoothing=False, masked_mean=False,
                  grad_norm_scale=False, target_guidance=3000.0, clip_weight=1.0, use_clip_loss=False, object_presence=False,
                  masked_mean_thresh=0.0, masked_mean_weight=0.0, write_to_file=False, use_energy=False, no_wt=False,
                  leaky_relu_slope=0.05):
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
            existing_files = [os.path.exists(path) for path in out_filenames]
            files_to_generate = [not file_exists for file_exists in existing_files]

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
            
            print("SELF-GUIDANCE")
            if any(files_to_generate):
                filtered_prompts = [p for p, should_gen in zip(batch_prompts, files_to_generate) if should_gen]
                filtered_generators = [g for g, should_gen in zip(generators, files_to_generate) if should_gen]
                filtered_sg_edits = [sg for sg, should_gen in zip(sg_edits, files_to_generate) if should_gen]

                if filtered_prompts:
                    out = pipe(prompt=filtered_prompts, generator=filtered_generators, sg_grad_wt=sg_grad_wt,
                               sg_edits=filtered_sg_edits,
                               num_inference_steps=num_inference_steps, L2_norm=L2_norm, margin=margin,
                               sg_loss_rescale=sg_loss_rescale, sg_t_start=sg_t_start, sg_t_end=sg_t_end,
                               self_guidance_mode=self_guidance_mode, loss_type=loss_type, loss_num=int(loss_num),
                               plot_centroid=plot_centroid, save_aux=save_aux, two_objects=two_objects,
                               update_latents=update_latents, img_id=img_id, smoothing=smoothing,
                               masked_mean=masked_mean, grad_norm_scale=grad_norm_scale, target_guidance=target_guidance,
                               clip_weight=clip_weight, use_clip_loss=use_clip_loss, object_presence=object_presence,
                               masked_mean_thresh=masked_mean_thresh, masked_mean_weight=masked_mean_weight,
                               write_to_file=write_to_file, save_dir_name=save_dir_name, use_energy=use_energy,
                               no_wt=no_wt, leaky_relu_slope=leaky_relu_slope).images

                    filtered_paths = [path for path, should_gen in zip(out_filenames, files_to_generate) if
                                      should_gen]
                    for img, path in zip(out, filtered_paths):
                        img.save(path)


def run_on_gpu(device, all_prompts, all_words, attn_greenlist, seeds, num_inference_steps,
               sg_t_start, sg_t_end, sg_grad_wt, sg_loss_rescale,
               L2_norm=False, shifts=[], num_images_per_prompt=1,
               vocab_spatial=[], loss_num=1, alpha=1, loss_type="relu", margin=0.1,
               self_guidance_mode=False, two_objects=False, plot_centroid=False, weight_combinations=None,
               do_multiprocessing=False, img_id="", update_latents=False, save_dir_name="", centroid_type="sg",
               benchmark=None, batch_size=1, model="model_name", smoothing=False, masked_mean=False, grad_norm_scale=False,
               target_guidance=3000.0, clip_weight=1.0, use_clip_loss=False, object_presence=False,
               masked_mean_thresh=0.0, masked_mean_weight=0.0, write_to_file=False, use_energy=False, no_wt=False,
               leaky_relu_slope=0.05):

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
                  centroid_type=centroid_type, benchmark=benchmark, batch_size=batch_size, model=model,
                  smoothing=smoothing, masked_mean=masked_mean, grad_norm_scale=grad_norm_scale,
                  target_guidance=target_guidance, clip_weight=clip_weight, use_clip_loss=use_clip_loss,
                  object_presence=object_presence, masked_mean_thresh=masked_mean_thresh,
                  masked_mean_weight=masked_mean_weight, write_to_file=write_to_file, use_energy=use_energy, no_wt=no_wt,
                  leaky_relu_slope=leaky_relu_slope)


def start_multiprocessing(attn_greenlist, json_filename, seeds,
                          num_inference_steps, sg_t_start, sg_t_end, sg_grad_wt, sg_loss_rescale,
                          L2_norm, shifts, num_images_per_prompt, vocab_spatial,
                          loss_num, alpha, loss_type, margin, self_guidance_mode,
                          two_objects, plot_centroid, weight_combinations,
                          do_multiprocessing, img_id, update_latents, benchmark,
                          save_dir_name, centroid_type, batch_size, model, world_size, rank, device, 
                          smoothing, masked_mean, grad_norm_scale, target_guidance, 
                          clip_weight, use_clip_loss, object_presence, masked_mean_thresh, 
                          masked_mean_weight, write_to_file, use_energy, no_wt, leaky_relu_slope):
    
    data_for_rank = get_prompts_for_rank(world_size, rank, json_filename)

    if benchmark == "visor":
        if False:
            prompts = [data['text'] for data in data_for_rank]
            all_words = {data['text']: [data['obj_1_attributes'][0], data["obj_2_attributes"][0]] for data in data_for_rank}
        else:
            prompts, all_words = [], {}
            for data in data_for_rank:
                prompt = data['text']
                prompts.append(prompt)
                # all_words[prompt] = [data['obj_1_attributes'][0], data["obj_2_attributes"][0]]

                all_words[prompt] = [
                    data['obj_1_attributes'][0].split()[1] if len(data['obj_1_attributes'][0].split()) > 1 else data['obj_1_attributes'][0],
                    data['obj_2_attributes'][0].split()[1] if len(data['obj_2_attributes'][0].split()) > 1 else data['obj_2_attributes'][0]
                ]
    if benchmark == "t2i":
        prompts = [data['prompt'] for data in data_for_rank]
        all_words = {data['prompt']: [data['objects'][0], data["objects"][1]] for data in data_for_rank}
    if benchmark == "geneval":
        pass

    run_on_gpu(device, prompts, all_words, attn_greenlist, seeds,
               num_inference_steps, sg_t_start, sg_t_end, sg_grad_wt, sg_loss_rescale,
               L2_norm, shifts, num_images_per_prompt, vocab_spatial,
               loss_num, alpha, loss_type, margin, self_guidance_mode,
               two_objects, plot_centroid, weight_combinations,
               do_multiprocessing, img_id, update_latents, save_dir_name,
               centroid_type, benchmark, batch_size, model, smoothing, masked_mean,
               grad_norm_scale, target_guidance, clip_weight, use_clip_loss,
               object_presence, masked_mean_thresh, masked_mean_weight, write_to_file,
               use_energy, no_wt, leaky_relu_slope)


def generate_images(config):
    # log_memory_usage()

    model = config.model
    model_information = get_model_id(model)

    world_size = config.world_size
    rank = config.rank
    
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
    job_id = config.job_id
    num_inference_steps = int(config.num_inference_steps)
    sg_t_start = int(config.sg_t_start)
    sg_t_end = int(config.sg_t_end)
    
    smoothing = bool(config.gaussian_smoothing)
    masked_mean = bool(config.masked_mean)
    masked_mean_thresh = float(config.masked_mean_thresh)
    masked_mean_weight = float(config.masked_mean_weight)
    grad_norm_scale = bool(config.grad_norm_scale)
    target_guidance = float(config.target_guidance)
    clip_weight = float(config.clip_weight)
    use_clip_loss = bool(config.use_clip_loss)
    object_presence = bool(config.object_presence)
    
    write_to_file = bool(config.write_to_file)
    use_energy = bool(config.use_energy)
    no_wt = bool(config.no_wt)
    
    leaky_relu_slope = float(config.leaky_relu_slope)

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

        all_prompts, all_words = [], {}
        for data in visor_data:
            prompt = data['text']
            all_prompts.append(prompt)
            # all_words[prompt] = [data['obj_1_attributes'][0], data["obj_2_attributes"][0]]

            all_words[prompt] = [
                data['obj_1_attributes'][0].split()[1] if len(data['obj_1_attributes'][0].split()) > 1 else data['obj_1_attributes'][0],
                data['obj_2_attributes'][0].split()[1] if len(data['obj_2_attributes'][0].split()) > 1 else data['obj_2_attributes'][0]
            ]

        seeds = [42]
        vocab_spatial = ["to the left of", "to the right of", "above", "below"]
        num_images_per_prompt = int(config.num_images_per_prompt)
        shifts = {
            "to the left of": [(0., 0.5), (1., 0.5)],
            "to the right of": [(1., 0.5), (0., 0.5)],
            "above": [(0.5, 0), (0.5, 1)],
            "below": [(0.5, 1), (0.5, 0)]
        }

    elif benchmark == "geneval":
        with open(os.path.join('json_files', f'{json_filename}.json'), 'r') as f:
            geneval_data = json.load(f)

        all_prompts, all_words = [], {}
        for data in geneval_data:
            prompt = data['prompt']
            all_prompts.append(prompt)
            # all_words[prompt] = [data['objects'][0], data["objects"][1]]
            all_words[prompt] = [
                data['objects'][0].split()[1] if len(data['objects'][0].split()) > 1 else data['objects'][0],
                data["objects"][1].split()[1] if len(data["objects"][1].split()) > 1 else data["objects"][1]
            ]
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
    if job_id:
        save_dir_name += f"_{job_id}"
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
        num_attn_layers = int(config.num_attn_layers)
        attn_greenlist = attn_greenlist[:num_attn_layers]
        print("num_attn_layers", num_attn_layers)

    if update_latents:
        sg_grad_wt = 7.5
        weight_combinations = [(0, 100.0, 0, 0)]
    else:
        sg_grad_wt = 1000.  # weight on self guidance term in sampling
        weight_combinations = [(0, 5.0, 0, 0)]

    sg_loss_rescale = 1000.  # to avoid numerical underflow, scale loss by this amount and then divide gradients after backprop
    # sg_t_start = 0
    
    if self_guidance_mode:
        num_inference_steps = 64
        sg_t_end = 3 * num_inference_steps // 16
    
    if model == "sdxl":
        num_inference_steps = int(config.num_inference_steps) # 50
        sg_t_end = int(config.sg_t_end) # 12

    if model == "sd1.4" or model == "sd1.5" or model == "sd2.1" or model == "spright":
        num_inference_steps = 500
        sg_t_end = 125
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
            save_dir_name, centroid_type, batch_size, model, world_size, rank, device,
            smoothing, masked_mean, grad_norm_scale, target_guidance, 
            clip_weight, use_clip_loss, object_presence, masked_mean_thresh, 
            masked_mean_weight, write_to_file, use_energy, no_wt, leaky_relu_slope)

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
                      batch_size=batch_size, model=model, run_base=run_base, smoothing=smoothing, masked_mean=masked_mean,
                      grad_norm_scale=grad_norm_scale, target_guidance=target_guidance, clip_weight=clip_weight,
                      use_clip_loss=use_clip_loss, object_presence=object_presence, masked_mean_thresh=masked_mean_thresh,
                      masked_mean_weight=masked_mean_weight, write_to_file=write_to_file, use_energy=use_energy, no_wt=no_wt,
                      leaky_relu_slope=leaky_relu_slope)


def sweep(config):
    for model in ["sdxl"]: # "sd1.5", "sdxl", "spright"
        config.model = model
        for loss in ["gelu"]: #["relu", "squared_relu", "gelu", "sigmoid"]
            config.loss_type = loss
            for margin in [0.25]: #[0.1, 0.25, 0.5]
                config.margin = margin
                # # more experiments
                # for loss_num in [2, 3]: # 1 is the default so that is done
                #     config.loss_num = loss_num
                for centroid_type in ["mean"]: # USE ONLY MEAN
                    config.centroid_type = centroid_type

                    img_id = f"{loss}_m={margin}_centr_{centroid_type}"
                    config.img_id = img_id

                    # benchmark = config.benchmark
                    # base_pattern = f"{model}_{img_id}"
                    # parent_dir = os.path.join("images", benchmark)
                    # if os.path.exists(parent_dir):
                    #     existing_dirs = [d for d in os.listdir(parent_dir) if
                    #                      os.path.isdir(os.path.join(parent_dir, d))]
                    #     if any(d.startswith(base_pattern) for d in existing_dirs):
                    #         print(f"Directory {base_pattern} already exists, skipping...")
                    #         continue

                    generate_images(config)
                    
                    
def run_ablation_spatial_loss_intervention(config):
    # sg_t_start_list = [0, 5, 10, 25, 50]
    # sp_loss_range_list = [25, 50] # for how many steps we apply the spatial loss
    # num_inference_steps = [200, 500]
    # for model in ["sd1.4", "sd2.1"]:
    #     for sg_t_start in sg_t_start_list:
    #         for sp_loss_range in sp_loss_range_list:
    #             for num_steps in num_inference_steps:
    # img_id = f"sg_t_start_{sg_t_start}_sp_loss_range_{sp_loss_range}_num_steps_{num_steps}"
    
    # for loss in ["relu", "gelu", "sigmoid"]: # 100 hours generation on t2i
    loss = config.loss_type
    img_id = f"sp_loss_{loss}_end_{config.sg_t_end}_num_steps_{config.num_inference_steps}"
    config.img_id = img_id

    generate_images(config)
    
    
def run_ablation_loss_num(config):
    for loss_num in [1, 2, 3]:
        config.loss_num = loss_num
        loss = config.loss_type
        img_id = f"loss_{loss}_loss_num_{loss_num}_ablation_132"
        config.img_id = img_id
        generate_images(config)
    
    
def run_ablation_object_presence(config):
    loss = config.loss_type
    loss_num = config.loss_num
    alpha = config.alpha
    img_id = f"loss_{loss}_loss_num_{loss_num}_alpha_{alpha}_object_presence_6maps_no_wt_ablation_132"
    config.img_id = img_id
    generate_images(config)


def run_ablation_masked_mean(config):
    loss = config.loss_type
    loss_num = config.loss_num
    alpha = config.alpha
    img_id = f"loss_{loss}_loss_num_{loss_num}_alpha_{alpha}_masked_mean_6maps_no_wt_ablation_132"
    config.img_id = img_id
    generate_images(config)
    
    
def run_ablation_energy(config):
    loss = config.loss_type
    loss_num = config.loss_num
    alpha = config.alpha
    img_id = f"loss_{loss}_loss_num_{loss_num}_alpha_{alpha}_energy_6maps_no_wt_ablation_132"
    config.img_id = img_id
    generate_images(config)
        
        
def get_spatial_loss_stats(config):
    for loss in ["relu", "gelu", "sigmoid"]:
        config.loss_type = loss
        img_id = f"{config.img_id}_{loss}"
        config.img_id = img_id
        generate_images(config)
        
        
def run_ablation_margin(config):
    for margin in [0.1, 0.25, 0.5, 0.75]:
        loss = config.loss_type
        config.margin = margin
        img_id = f"{loss}_margin_{margin}_ablation_20"
        config.img_id = img_id
        generate_images(config)
        
        
def run_ablation_alpha(config):
    loss = config.loss_type
    alpha = config.alpha
    img_id = f"{loss}_alpha_{alpha}_ablation_20"
    config.img_id = img_id
    generate_images(config)
    
    
def run_ablation_6attn_maps(config):
    loss = config.loss_type
    loss_num = config.loss_num
    alpha = config.alpha
    num_attn_maps = config.num_attn_layers
    if config.no_wt:
        no_wt = "_no_wt"
    else:
        no_wt = ""
    img_id = f"loss_{loss}_loss_num_{loss_num}_alpha_{alpha}_{num_attn_maps}_attn_maps{no_wt}_ablation_132"
    config.img_id = img_id
    generate_images(config)
    
    
def run_ablation_leaky_relu_attn_maps(config):
    loss = config.loss_type
    loss_num = config.loss_num
    alpha = config.alpha
    num_attn_maps = config.num_attn_layers
    if config.no_wt:
        no_wt = "_no_wt"
    else:
        no_wt = ""
    slope = config.leaky_relu_slope
    img_id = f"loss_{loss}_slope_{slope}_loss_num_{loss_num}_alpha_{alpha}_{num_attn_maps}_attn_maps{no_wt}_ablation_132"
    config.img_id = img_id
    generate_images(config)


if __name__ == "__main__":
    # Single experiment
    config = get_config()
    if config.sweep:
        sweep(config)
    else:
        generate_images(config)

