#!/bin/sh

#SBATCH --partition=prb,insy,general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --mail-type=END
#SBATCH --gres=gpu:a40

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

export HF_HOME=/tudelft.net/staff-umbrella/StudentsCVlab/vchatalbasheva/Thesis-Splign/hf_cache

srun python gen_images_energy.py \
    --benchmark visor \
    --json_filename visor_ablation_132 \
    --model sd2.1 \
    --two_objects True \
    --loss_type __LOSS_TYPE__ \
    --loss_num __LOSS_NUM__ \
    --alpha __ALPHA__ \
    --margin 0.25 \
    --centroid_type mean \
    --batch_size 1 \
    --num_attn_layers 6 \
    --no_wt True \
    --use_energy True