#!/bin/sh

#SBATCH --job-name=visor_eval
#SBATCH --partition=ptix-h100
#SBATCH --account=mv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=sieger.falkena@shell.com  

# ------------------------------------------------------------------------------
# Printing some information
# -----------------------------------------------------------------------------

/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

# ------------------------------------------------------------------------------
# Setting up the environment
# ------------------------------------------------------------------------------

echo "----------------- Environment ------------------"
source /glb/home/nlasqh/.bashrc 
conda activate splign
cd /glb/bng/pt.simpl/proj.nobackup/vhpc/nlasqh/Thesis-Splign

# ------------------------------------------------------------------------------
# And finally running the code
# ------------------------------------------------------------------------------

echo "--------------- Running the code ---------------"

echo -n "This run started on: "
date

python -c "import torch; print(torch.cuda.device_count())"
nvidia-smi

srun python run_evaluation.py --benchmark visor --json_filename visor_subset_1000 --model sd1.4 --img_id zero_set_sg_0-125_500steps_wt10.0_710406

echo -n "This run completed on: "
date
