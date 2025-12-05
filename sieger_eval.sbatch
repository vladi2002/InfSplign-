#!/bin/sh

#SBATCH --job-name=visor_eval
#SBATCH --partition=ptix-h100
#SBATCH --account=mv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=sieger.falkena@shell.com  
#SBATCH --exclude=bngstcb-n-gp022b02,bngstcb-n-gp022b04,bngstcb-n-gp022b08

# ------------------------------------------------------------------------------
# Printing some information
# -----------------------------------------------------------------------------

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------

echo "----------------- Environment ------------------"
source /glb/home/nlsfai/.bashrc
conda activate splign
cd /glb/data/ptxd_dash/nlsfai/InfSplign_Energy

# ------------------------------------------------------------------------------
# And finally running the code
# ------------------------------------------------------------------------------

echo "--------------- Running the code ---------------"

echo -n "This run started on: "
date

python -c "import torch; print(torch.cuda.device_count())"
nvidia-smi

# srun python run_evaluation.py --benchmark t2i --json_filename t2i_prompts --model $MODEL --img_id $IMG_ID
# srun python run_evaluation.py --benchmark t2i --json_filename t2i_prompts --model $MODEL --img_id $IMG_ID
# srun python run_evaluation.py --benchmark t2i --json_filename t2i_prompts --model sd2.1 --img_id t2i_strategy_dec_energy_gibs
srun python run_evaluation.py --benchmark visor --json_filename visor_prompts --model sdxl --img_id $IMG_ID

echo -n "This run completed on: "
date
