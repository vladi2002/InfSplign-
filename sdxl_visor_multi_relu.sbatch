#!/bin/sh

#SBATCH --partition=ewi-insy
#SBATCH --qos=reservation
#SBATCH --reservation=spatial
#SBATCH --time=100:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --gres=gpu:3

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

# Sieger doesn't need this (but Yanbo does and should change this)
export HF_HOME=/tudelft.net/staff-umbrella/StudentsCVlab/vchatalbasheva/hf_cache

srun python combined_pipeline_multiprocessing.py --do_multiprocessing True --benchmark visor --json_filename visor_3160 --model sdxl --two_objects True --loss_type relu --loss_num 1 --margin 0.1 --alpha 1.0 --img_id relu