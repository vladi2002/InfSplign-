#!/bin/sh



#SBATCH --job-name=t2i
#SBATCH --partition=ptix-h100
#SBATCH --account=mv
#SBATCH --time=24:00:00
#SBATCH --ntasks=8
#SBATCH --output=job_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:h100:8
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
# module load MPICH
module load OpenMPI/4.1.6-intel-compilers-2021.2.0-CUDA-12.2.2 # Load OpenMPI module

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

# mpirun --map-by ppr:$SLURM_NTASKS_PER_NODE:node --np $SLURM_NTASKS --output-filename outputs python pipeline_batch.py --model sd1.4 --do_multiprocessing True --benchmark visor --json_filename text_spatial_rel_phrases --two_objects True --loss_type sigmoid --loss_num 1 --margin 0.25 --alpha 1.0 --img_id sigmoid_mean --centroid_type mean --batch_size 3 --job_id $SLURM_JOB_ID
mpirun --map-by ppr:$SLURM_NTASKS_PER_NODE:node --np $SLURM_NTASKS --output-filename outputs python pipeline_batch.py --model sd1.4 --do_multiprocessing True --benchmark t2i --json_filename t2i_prompts --two_objects True --loss_type sigmoid --loss_num 1 --margin 0.25 --alpha 1.0 --img_id sigmoid_mean --centroid_type mean --batch_size 3 --job_id $SLURM_JOB_ID --num_inference_steps 500 --sg_t_start 0 --sg_t_end 125 --use_mpi

echo -n "This run completed on: "
date
