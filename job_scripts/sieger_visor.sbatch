#!/bin/sh

#SBATCH --job-name=loss_abl
#SBATCH --partition=ptix-h100
#SBATCH --account=mv
#SBATCH --time=6-00:00:00
#SBATCH --ntasks=8
#SBATCH --output=job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
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
# module load CUDA/12.2.2
# module load MPICH
module load OpenMPI/4.1.6-intel-compilers-2021.2.0-CUDA-12.2.2 # Load OpenMPI module

source /glb/home/nlsfai/.bashrc
conda activate splign
cd /glb/data/ptxd_dash/nlsfai/InfSplign_Energy

# ------------------------------------------------------------------------------
# And finally running the code
# ------------------------------------------------------------------------------
echo "--------------- Running the code ---------------"

which python
pip show ftfy

echo -n "This run started on: "
date

python -c "import torch; print(torch.cuda.device_count())"
nvidia-smi

# mpirun --map-by ppr:$SLURM_NTASKS_PER_NODE:node --np $SLURM_NTASKS --output-filename outputs python pipeline_batch.py --batch_size 20 --model $MODEL --json_filename visor_prompts --loss_type gelu --img_id $IMG_ID --do_multiprocessing True --top_strategy None --strategy $STRATEGY --energy_loss $LOSS --schedule ddpm --use_mpi
# mpirun --map-by ppr:$SLURM_NTASKS_PER_NODE:node --np $SLURM_NTASKS --output-filename outputs python pipeline_batch.py --batch_size 20 --model $MODEL --json_filename visor_prompts --loss_type gelu --img_id $IMG_ID --strategy $STRATEGY --do_multiprocessing True --use_mpi
mpirun --map-by ppr:$SLURM_NTASKS_PER_NODE:node --np $SLURM_NTASKS --output-filename outputs python pipeline_batch.py --batch_size 1 --model $MODEL --json_filename visor_prompts --loss_type $LOSS --img_id $IMG_ID --alpha $ALPHA --margin $MARGIN --lambda_spatial $LAMBDA_SPATIAL --lambda_balance $LAMBDA_BALANCE --lambda_presence $LAMBDA_PRESENCE --do_multiprocessing True --schedule ddpm --use_mpi
# Clean up hostfile
rm -f hostfile_$SLURM_JOB_ID
echo -n "This run completed on: "
date
