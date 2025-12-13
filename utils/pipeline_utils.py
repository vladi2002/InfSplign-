import json
import os

import numpy as np
import torch
from glob import glob
from PIL import Image, ImageDraw
from diffusers import LMSDiscreteScheduler, DDIMScheduler, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from utils.visor_utils import process_detection
from tqdm import tqdm


def get_prompts_ann_from_json(json_name):
    with open(os.path.join('json_files', f'{json_name}.json'), 'r') as f:
        json_file = json.load(f)
    return json_file


# def split_prompts(num_gpus, benchmark, json_filename):
#     with open(os.path.join('json_files', f'{json_filename}.json'), 'r') as f:
#         spatial_prompts = json.load(f)

#     print(f"Total number of unique spatial prompts: {len(spatial_prompts)}")

#     save_dir = os.path.join('data_splits', f'{benchmark}', f"multiprocessing_{num_gpus}")
#     os.makedirs(save_dir, exist_ok=True)

#     chunk_size = len(spatial_prompts) // num_gpus
#     remainder = len(spatial_prompts) % num_gpus
#     print(chunk_size, remainder)

#     start = 0
#     for i in range(num_gpus):
#         end = start + chunk_size + (1 if i < remainder else 0)
#         print(i, start, end)
#         part = spatial_prompts[start:end]

#         with open(os.path.join(save_dir, f"prompts_part_{i}.json"), 'w') as f:
#             json.dump(part, f, indent=2)

#         print(f"Saved {len(part)} items to prompts_part_{i}.json")
#         start = end


def get_prompts_for_rank(size, rank, json_filename):
    with open(os.path.join('json_files', f'{json_filename}.json'), 'r') as f:
        spatial_prompts = np.array(json.load(f))

    print("len: ", spatial_prompts.shape)

    # Split prompts across size and select portion for rank
    data_chunks = np.array_split(spatial_prompts, size)
    rank_prompts = data_chunks[rank]
    return rank_prompts


def initialize_model(config, model):
    if model == "spright":
        pipe_id = "SPRIGHT-T2I/spright-t2i-sd2"
    elif model == "stable-diffusion-v1-4":
        pipe_id = f"CompVis/{model}"
    elif "stable-diffusion" in model:
        pipe_id = f"stabilityai/{model}"
        
    pipe = StableDiffusionPipeline.from_pretrained(pipe_id, revision="fp16", torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to(config.device)

    if config.scheduler == "lms":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif config.scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif config.scheduler == "ddpm":
        pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    elif config.scheduler == "pndm":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

    pipe.safety_checker = None
    return pipe


def create_images(config, pipe, prompts, split, model):
    for prompt_data in prompts:
        # check if this image doesnt already exist
        filename = prompt_data["text"]
        if not glob(f"images/{model}/{split}/{filename}*.png"):
            save_model_images(config, prompt_data, pipe, split, model)


def save_model_images(config, prompt, pipe, split, model):
    images = []
    generator = torch.Generator(config.device).manual_seed(config.seed)
    for i in range(config.num_images):
        image = pipe(prompt, guidance_scale=config.scale, num_inference_steps=config.steps,
                     weights=config.weights, generator=generator, cmi=config.cmi,
                     use_neg_rel=config.use_neg_rel).images[0]
        images.append(image)
    os.makedirs(f"images/{model}/{split}", exist_ok=True)
    filename = prompt["text"]

    for i, image in enumerate(images):
        image.save(f"images/{model}/{split}/{filename}_{i}.png")


def initialize_object_detection_model(config):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    model = model.to(config.device)
    return processor, model


def draw_bounding_boxes(image, boxes, labels):
    draw = ImageDraw.Draw(image)
    for box, label in zip(boxes, labels):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), str(label), fill="red")
    return image


def create_object_detection_annotations(config, prompts_data, processor, model, verbose=False, relationship=None):
    results = {}
    items = 0
    
    model_type = config.img_id
    baseline = config.model
    json_filename = config.json_filename
    
    for item in tqdm(prompts_data):
        if item == 4:
            break
        uniq_id = item["text"]
        
        images = []
        for i in range(4):
            img_id = "{}_{}".format(uniq_id, i)
            if relationship:
                impath = os.path.join("images", "visor", baseline, relationship, "{}.png".format(img_id))
            else:
                impath = os.path.join("images", "visor", f"{baseline}_{model_type}", "{}.png".format(img_id))
                
            # not all images are generated yet
            if not os.path.exists(impath):
                print("Could not find {}".format(impath))
                continue

            im = Image.open(impath)
            images.append(im)

        # not all images are generated yet
        if not os.path.exists(impath):
            continue

        obj1 = item["obj_1_attributes"][0]
        obj2 = item["obj_2_attributes"][0]
        rel = item["rel_type"]
        texts = [["a photo of a {}".format(obj1), "a photo of a {}".format(obj2)]]

        for i in range(4):
            image = images[i]
            img_id = "{}_{}".format(uniq_id, i)  # i+1
            with torch.no_grad():
                inputs = processor(text=texts, images=image, return_tensors="pt").to(config.device)
                outputs = model(**inputs)
                target_sizes = torch.Tensor([image.size[::-1]]).to(config.device)
                outs = processor.post_process(outputs=outputs, target_sizes=target_sizes)
            results[img_id] = process_detection(outs, obj1, obj2, rel)

            if verbose:
                image_with_boxes = draw_bounding_boxes(image, results[img_id]["boxes"], results[img_id]["labels"])
                os.makedirs(f"images/{baseline}/{json_filename}/bbox", exist_ok=True)
                image_with_boxes.save(f"images/{baseline}/{json_filename}/bbox/{img_id}.png")
        items += 1

    save_dir = os.path.join('objdet_results', 'visor', f"{baseline}_{model_type}")
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, f'{json_filename}.json'), 'w') as f:
        json.dump(results, f, indent=4)


def load_object_detection_ann(config, relationship=None):
    model = f"{config.model}_{config.img_id}"
    json_file = config.json_filename
    
    save_dir = os.path.join('objdet_results', 'visor', model)
    if relationship:
        with open(os.path.join(save_dir, relationship, f'{json_file}.json'), 'r') as f:
            obj_det_annotations = json.load(f)
    else:
        with open(os.path.join(save_dir, f'{json_file}.json'), 'r') as f:
            obj_det_annotations = json.load(f)

    return obj_det_annotations
