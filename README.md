# Spatial Understanding of Diffusion Models

### How to run the code

#### Clone

Clone this repo to your local machine using
```
git clone https://github.com/SarahRastegar/InfSplign_Energy.git
```

#### Create Virtual Environment (venv)
Run in your Anaconda terminal:
```
conda create -n <env_name> python=3.11
conda activate <env_name>
```

or 

```
conda env create -f environment.yml
```


#### Install Dependencies Main Code
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install --yes --file requirements.txt
conda install -c conda-forge einops 
conda install -c conda-forge accelerate
conda install -c conda-forge scikit-learn
```

#### Install Dependencies CLIP
```
pip install git+https://github.com/openai/CLIP.git

```



##### Example command for the visor benchmark:
```
python pipeline_batch.py  --benchmark visor --model sd1.4 --two_objects True --loss_type relu --loss_num 1 --margin 0.1 --alpha 1 --img_id relu_mean --centroid_type mean  --batch_size 1 --gaussian_smoothing true --json_filename visor_ablation_500

```




#### Install Dependencies for Evaluation Benchmarks

##### T2I-CompBench
In the comments I have put some of the tricks I used to set this up. Do not run the commented out lines for now cause maybe they are not needed. If when you run the code, you get an error, try uncommenting them. (This is only for the people using the university cluster)

```
python -m spacy download en

# T2I-CompBench (DAIC cluster - university)

mkdir -p UniDet_eval/experts/expert_weights
cd UniDet_eval/experts/expert_weights
wget https://huggingface.co/shikunl/prismer/resolve/main/expert_weights/Unified_learned_OCIM_RS200_6x%2B2x.pth
wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/dpt_hybrid-midas-501f0c75.pt
pip install gdown
gdown https://docs.google.com/uc?id=1C4sgkirmgMumKXXiLOPmCKNTZAc3oVbq

# Here you go back to the UneDet_eval folder

# conda install -c conda-forge gcc_linux-64=9 gxx_linux-64=9
# export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
# export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
pip install --user --no-cache-dir git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13

# Fix bug in detectron2:
# In file "\lib\site-packages\detectron2\data\transforms\transform.py"
# Change LINEAR to BILINEAR in this line:
def __init__(self, src_rect, output_size, interp=Image.BILINEAR, fill=0): -> instead of LINEAR
```

##### GenEval
The requirements for this benchmark were incompatible with the dependencies in the code so you have a make a new environment for it. Following these steps should work:

```
conda create -n geneval python 3.8.10
git clone https://github.com/djghosh13/geneval.git
cd geneval
./evaluation/download_models.sh geneval_obj_det/

conda install -c conda-forge einops platformdirs setuptools 
conda install diffusers transformers lightning tomli
pip install --user --no-cache-dir open-clip-torch==2.26.1 clip-benchmark openmim
python -m mim install --user mmengine mmcv-full==1.7.2

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout 2.x
pip install -v -e .
```

#### Run:
To generate the images, run the following command:
```
python combined_pipeline_multiprocessing.py --benchmark <benchmark name> --json_filename <filename> --model sdxl --two_objects True --loss_type relu --loss_num 1 --margin 0.1 --alpha 1.0 --img_id relu
```

The same configuration can be run using multiprocessing in this way:
```
python combined_pipeline_multiprocessing.py --do_multiprocessing True --benchmark <benchmark name> --json_filename <filename> --model sdxl --two_objects True --loss_type relu --loss_num 1 --margin 0.1 --alpha 1.0 --img_id relu
```

Once, the generation is done, you can compute the metric using the following command:
```
python run_evaluation.py --benchmark <benchmark name> --json_filename <filename> --model sdxl --two_objects True --loss_type relu --loss_num 1 --margin 0.1 --alpha 1.0 --img_id relu
```
Evaluation benchmarks are:
- `visor`: VISOR
- `t2i`: T2I-CompBench
- `geneval`: GenEval

The corresponding data files are:
- VISOR: `text_spatial_rel_phrases`
- T2I-CompBench: `t2i_prompts`
- GenEval: `geneval_objects`.

##### Example command for the visor benchmark:
```
python pipeline_batch.py  --benchmark visor --model sd1.4 --two_objects True --loss_type relu --loss_num 1 --margin 0.1 --alpha 1 --img_id relu_mean --centroid_type mean  --batch_size 1 --gaussian_smoothing true --json_filename visor_ablation_500

```

##### Example command for the t2i benchmark:
```
python combined_pipeline_multiprocessing.py --do_multiprocessing True --benchmark t2i --json_filename t2i_objects --model sdxl --two_objects True --loss_type relu --loss_num 1 --margin 0.1 --alpha 1.0 --img_id relu
```

To ensure multiprocessing with multiple GPUs, you can add to the command `--do_multiprocessing True`

##### Fix the import clip problem:
```
pip uninstall clip
pip install git+https://github.com/openai/CLIP.git
```

##### GenEval is still to be setup
