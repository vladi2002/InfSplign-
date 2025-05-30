#!/bin/bash

loss_types=("relu" "gelu" "sigmoid")
alpha_values=("1.0" "2.0" "5.0")
loss_nums=("1" "2" "3")

original_sbatch="/tudelft.net/staff-umbrella/StudentsCVlab/vchatalbasheva/Thesis-Splign/job_scripts/visor_ablation_sd2.1_9attn_maps_no_wt.sbatch"
sbatch_files_dir="/tudelft.net/staff-umbrella/StudentsCVlab/vchatalbasheva/Thesis-Splign/sbatch_files"

for loss_type in "${loss_types[@]}"; do
    for loss_num in "${loss_nums[@]}"; do
        for alpha in "${alpha_values[@]}"; do
            echo "Submitting job with loss_type=$loss_type, loss_num=$loss_num, alpha=$alpha"
            
            new_sbatch="${sbatch_files_dir}/sd2.1_${loss_type}_${loss_num}_${alpha}_visor_ablation.sbatch"
            
            sed "s/__LOSS_TYPE__/${loss_type}/g; s/__LOSS_NUM__/${loss_num}/g; s/__ALPHA__/${alpha}/g" "$original_sbatch" > "$new_sbatch"
            
            sbatch "$new_sbatch"
        done
    done
done
