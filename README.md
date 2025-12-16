# Codebase of InfSplign
### Official Implementation for ["InfSplign: Inference-Time Spatial Alignment of\\ Text-to-Image Diffusion Models"](https://openreview.net/forum?id=k9SVcrmXL8)<br> (Sarah Rastegar, Violeta Chatalbasheva, Sieger Falkena, Anuj Singh, Yanbo Wang, Tejas Gokhale, Hamid Palangi, Hadi Jamali-Rad)

## Abstract
Text-to-image (T2I) diffusion models generate high-quality images but often fail to capture the spatial relations specified in text prompts. This limitation can be traced to two factors: lack of fine-grained spatial supervision in training data and inability of text embeddings to encode spatial semantics. We introduce InfSplign, a training-free inference-time method that improves spatial alignment by adjusting the noise through a compound loss in every denoising step. Proposed loss leverages different levels of cross-attention maps extracted from the U-Net decoder to enforce accurate object placement and a balanced object presence during sampling. The method is lightweight, plug-and-play, and compatible with any diffusion backbone. Our comprehensive evaluations on VISOR and T2I-CompBench show that InfSplign establishes a new state of the art (to the best of our knowledge), achieving substantial performance gains over the strongest existing inference-time baselines and even outperforming fine-tuning-based methods.


<p align="center">
    <img src="imgs/InfSplign%20qualitative%20images%20.png" width="90%" >
</p>

### Environment

#### Create Virtual Environment (venv)
```
conda env create -f environment.yml
conda activate infsplign
python -m spacy download en
```
or

```
conda create -n infsplign python=3.11
conda activate infsplign
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install --yes --file requirements.txt
python -m spacy download en
```

##### T2I-CompBench
Follow the steps for downloading the detection model weights from the original [T2I-CompBench repo](https://github.com/Karine-Huang/T2I-CompBench/tree/main) in section _UniDet for 2D/3D-Spatial Relationships and Numeracy evaluation_. Then run the following steps:

```
cd UniDet_eval
pip install --user --no-cache-dir git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13

# Fix bug in detectron2:
# In file "\lib\site-packages\detectron2\data\transforms\transform.py"
# Change LINEAR to BILINEAR in this line:
def __init__(self, src_rect, output_size, interp=Image.BILINEAR, fill=0):
```

### Usage

<p align="center">
    <img src="imgs/InfSplign%20diagram.png" width="80%" >
</p>

To generate the images, run the following command:
```python
python pipeline_batch.py --model sd2.1 --benchmark <benchmark_name> --json_filename <prompt_filename> --batch_size 10 --loss_type gelu --strategy diff --energy_loss var
```

After image generation completes, compute the evaluation scores:
```python
python run_evaluation.py --model sdxl sd2.1 --benchmark <benchmark_name> --json_filename <prompt_filename>
```

Evaluation benchmarks are:
- `visor`: VISOR
- `t2i`: T2I-CompBench
- `geneval`: GenEval

The corresponding data files are:
- VISOR: `visor_prompts`
- T2I-CompBench: `t2i_prompts`
- GenEval: `geneval_objects`.

We provide multiple VISOR subsets in `json_files`.


## Contact
Corresponding author: Violeta Chatalbasheva (<violetachatalbasheva@gmail.com>)