# Spatial Understanding of Diffusion Models

### How to run the code

#### Clone

Clone this repo to your local machine using
```
https://github.com/VioletaChatalbasheva/Thesis-Splign.git
```

#### Create Virtual Environment (venv)
Run in your Anaconda terminal:
```
conda create -n <env_name> python=3.11
conda activate <env_name>
```

#### Install Dependencies Main Code
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install --yes --file requirements.txt
conda install -c conda-forge einops 
conda install -c conda-forge accelerate
conda install -c conda-forge scikit-learn
```

#### Install Dependencies for Evaluation Benchmarks

##### T2I-CompBench
```
python -m spacy download en

# T2I-CompBench (DAIC cluster - university)
# conda install -c conda-forge gcc_linux-64=9 gxx_linux-64=9
# export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc
# export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++
pip install --user --no-cache-dir git+https://github.com/facebookresearch/detectron2.git@5aeb252b194b93dc2879b4ac34bc51a31b5aee13

# Fix bug in detectron2:
# In file "\lib\site-packages\detectron2\data\transforms\transform.py"
# Change LINEAR to BILINEAR in this line:
def __init__(self, src_rect, output_size, interp=Image.BILINEAR, fill=0): -> instead of LINEAR
```

##### GenEvan
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
pip install --user -v -e .
```

#### Run
To run the spatial loss ReLU, run the following command:

```
python combined_pipeline_multiprocessing.py --two_objects True --loss_type relu --loss_num 1 --margin 0.1 --alpha 1.0 --img_id relu
```

If you want to apply multiprocessing on top of that, run the following command:

```
python combined_pipeline_multiprocessing.py --do_multiprocessing True --two_objects True --loss_type relu --loss_num 1 --margin 0.1 --alpha 1.0 --img_id relu
```

