#!/bin/bash

loss_types=("relu" "gelu" "sigmoid")
alpha_values=("2.0" "3.0" "5.0" "10.0")

original_sbatch="/tudelft.net/staff-umbrella/StudentsCVlab/vchatalbasheva/Thesis-Splign/job_scripts/sd2.1_alpha_visor_ablation.sbatch"
sbatch_files_dir="/tudelft.net/staff-umbrella/StudentsCVlab/vchatalbasheva/Thesis-Splign/sbatch_files"

for loss_type in "${loss_types[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        echo "Submitting job with loss_type=$loss_type, alpha=$alpha"
        
        new_sbatch="${sbatch_files_dir}/sd2.1_${loss_type}_${alpha}_visor_ablation.sbatch"
        
        sed "s/__LOSS_TYPE__/${loss_type}/g; s/__ALPHA__/${alpha}/g" "$original_sbatch" > "$new_sbatch"
        
        sbatch "$new_sbatch"
    done
done
