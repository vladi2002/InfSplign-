#!/bin/sh



#SBATCH --job-name=visor
#SBATCH --partition=ptix-h100
#SBATCH --account=mv
#SBATCH --time=24:00:00
#SBATCH --ntasks=32
#SBATCH --output=job_%j.out
#SBATCH --nodes=4
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

mpirun --map-by ppr:$SLURM_NTASKS_PER_NODE:node --np $SLURM_NTASKS --output-filename outputs python pipeline_batch.py --model sd2.1 --do_multiprocessing True --benchmark visor --json_filename visor_prompts --two_objects True --loss_type sigmoid --loss_num 1 --margin 0.0 --alpha 1.0 --img_id "final_run" --centroid_type mean --batch_size 1 --job_id $SLURM_JOB_ID --num_inference_steps 500 --sg_t_start 0 --sg_t_end 125 --use_clip_loss --clip_weight 2.0 --use_mpi
# mpirun --map-by ppr:$SLURM_NTASKS_PER_NODE:node --np $SLURM_NTASKS --output-filename outputs python pipeline_batch.py --model sd1.4 --do_multiprocessing True --benchmark visor --json_filename visor_subset_1000 --two_objects True --loss_type sigmoid --loss_num 1 --margin 0.0 --alpha 1.0 --img_id "zero_set_sg_0-125_500steps_grad_norm_3000_test"  --centroid_type mean --batch_size 3 --job_id $SLURM_JOB_ID --num_inference_steps 200 --sg_t_start 0 --sg_t_end 25 --grad_norm_scale True --target_guidance 3000
# mpirun --map-by ppr:$SLURM_NTASKS_PER_NODE:node --np $SLURM_NTASKS --output-filename outputs python pipeline_batch.py --model sd2.1 --do_multiprocessing True --benchmark visor --json_filename visor_prompts --two_objects True --loss_type sigmoid --loss_num 1 --margin 0.0 --alpha 1.0 --img_id "visor_sg_0-125_steps_500_double_words"  --centroid_type mean --batch_size 3 --job_id $SLURM_JOB_ID --num_inference_steps 500 --sg_t_start 0 --sg_t_end 125
# mpirun --map-by ppr:$SLURM_NTASKS_PER_NODE:node --np $SLURM_NTASKS --output-filename outputs python pipeline_batch.py --model sd1.4 --do_multiprocessing True --benchmark t2i --json_filename t2i_prompts --two_objects True --loss_type sigmoid --loss_num 1 --margin 0.25 --alpha 1.0 --img_id sigmoid_mean --centroid_type mean --batch_size 3 --job_id $SLURM_JOB_ID

echo -n "This run completed on: "
date
