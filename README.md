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
In the comments I have put some of the tricks I used to set this up. Do not run the commented out lines for now cause maybe they are not needed. If when you run the code, you get an error, try uncommenting them. (This is only for the people using the university cluster)

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

#### Run (Sieger):
To generate the images for a subset of 4 prompt from the VISOR dataset, run the following command:
```
python combined_pipeline_multiprocessing.py --do_multiprocessing True --benchmark visor --json_filename visor_4 --model sdxl --two_objects True --loss_type relu --loss_num 1 --margin 0.1 --alpha 1.0 --img_id relu
```
The code will generate 4 images per prompt, so 16 images in total.

Once, the generation is done, you can compute the metric using the following command:
```
python run_evaluation.py --benchmark visor --json_filename visor_4 --model sdxl --two_objects True --loss_type relu --loss_num 1 --margin 0.1 --alpha 1.0 --img_id relu
```
Right now, the second command is not working using multiprocessing. (But I might need to change that. I'll let you know once it's done!)