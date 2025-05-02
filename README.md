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

#### Install Dependencies
```
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install --yes --file requirements.txt
conda install -c conda-forge einops 
conda install -c conda-forge accelerate
conda install -c conda-forge scikit-learn
```

#### Additional installations
```
python -m spacy download en
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

