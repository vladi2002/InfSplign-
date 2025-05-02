# Spatial Understanding of Diffusion Models

This repo builds upon [Composable Diffusion](https://github.com/energy-based-model/Compositional-Visual-Generation-with-Composable-Diffusion-Models-PyTorch) and aims to improve the spatial understanding of diffusion backbones such as:
- stable-diffusion-v1-4
- stable-diffusion-v2-base
- stable-diffusion-v2
- stable-diffusion-v2-1
- spright (sd-v2.1 fine-tuned with 444 images containing spatial relationships between more than 18 objects)

The idea is to generate each object in isolation using Composable Diffusion and enforce the spatial relationship between the objects using [Conditional Mutual Information (CMI)](https://github.com/kxh001/Info-Decomp/tree/main). CMI has proved to be a more reliable measure capturing the spatial relationship compared to attention maps. By maximizing the CMI, we target correct image generation with respect to the spatial aspect provided as part of the input prompt. 

The prompts used for these experiments are based on the SR2D dataset from VISOR. However, for the purpose of this project, we distinguish two categories of prompts: natural and unnatural. The natural prompts are defined as representing realistic, observable scenarios, while the unnatural prompts cannot occur in reality. The reason why this split is introduced is that diffusion models struggle to generate images that do not comply with the natural physics laws (e.g. "a horse on top of an astronaut" will be considered unnatural).

The prompts need to be converted in a format suitable for Composable Diffusion:
```
prompt -> "mystical trees AND dark"
Composable Diffusion -> "mystical trees | dark"
```

Composable Diffusion faces the challenge of merging two objects together. However, our use case requires the generation of each of the two objects, specified in the prompt, separately, to be able to enforce the spatial relationship between them. The prompt is split in the following way:

```
prompt -> "cat on the left of dog"
Composable Diffusion -> "cat on the left | dog on the right"
```

### How to run the code

#### Clone

Clone this repo to your local machine using
```
https://github.com/VioletaChatalbasheva/spatial-understanding-diffusion-models.git
```

#### Install Dependencies (Sarah)
```
conda create -n <env_name> python=3.11
conda activate <env_name>
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
conda install --yes --file requirements.txt
conda install -c conda-forge einops 
conda install -c conda-forge accelerate
conda install -c conda-forge scikit-learn
```

#### Create Virtual Environment (venv)
Run in your terminal:
```
# Create virtualenv, make sure to use python3.10
$ virtualenv -p python3.10 <env_name>

# Activate venv
$ source <env_name>/bin/activate
```

#### Install Requirements
Move to the project folder and run in your terminal:
```
pip install -r requirements.txt
```

You need pytorch with CUDA support. You can install it using the following command from the [pytorch](https://pytorch.org/get-started/locally/) website:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Run
Run the following script to generate a subset of natural and unnatural prompts:

```
python split_prompts_utils.py
```

The first time you run the code you need to (`run_baselines`) to get the generated images and (`run_object_detection`) to obtain the object detection annotations for the evaluation. Then you need to specify the diffusion model backbone with the (``mo``) parameter. Otherwise, the code will run the pipeline for all available models.

```
python main.py --run_baselines True --run_object_detection True --mo stable-diffusion-2-base 
```

The current evaluation method utilizes [VISOR](https://github.com/microsoft/VISOR). In short, VISOR uses the Owl-ViT object detector to find the centroids of the generated objects and based on predefined heuristic rules it calculates a VISOR score which measures the correctness of the spatial relationships between the objects. So the output is a table in the following format:


| Model   | OA   | VISOR_cond | VISOR_uncond | VISOR_1 | VISOR_2 | VISOR_3 | VISOR_4 | Num_Imgs |
|---------|------|------------|--------------|---------|---------|---------|---------|----------|
| spright | 87.5 | 92.86      | 81.25        | 100     | 75      | 75      | 75      | 16       |


[//]: # (## ask for github user names of the team)

[//]: # (### run the pipeline for stable diffusions for the current splits &#40;114 prompts&#41;)

[//]: # (### add a&e and sdg to pipeline)

[//]: # (### generate more natural-unnatural prompts using an LLM)