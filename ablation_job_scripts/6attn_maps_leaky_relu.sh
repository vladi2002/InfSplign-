#!/bin/bash

loss_types=("leaky_relu")
alpha_values=("0.1" "0.25" "0.5" "0.75" "1.0" "2.0" "5.0")
loss_nums=("1" "2" "3")
leaky_relu_slope=("0.05" "0.1" "0.25" "0.5")

original_sbatch="/tudelft.net/staff-umbrella/StudentsCVlab/vchatalbasheva/Thesis-Splign/job_scripts/sd2.1_leaky_relu_slope_visor_ablation.sbatch"
sbatch_files_dir="/tudelft.net/staff-umbrella/StudentsCVlab/vchatalbasheva/Thesis-Splign/sbatch_files"

for loss_type in "${loss_types[@]}"; do
    for loss_num in "${loss_nums[@]}"; do
        for alpha in "${alpha_values[@]}"; do
            for slope in "${leaky_relu_slope[@]}"; do
                new_sbatch="${sbatch_files_dir}/sd2.1_${loss_type}_slope_${slope}_${loss_num}_${alpha}_6attn_no_wt_visor_ablation.sbatch"

                echo "Submitting job with loss_type=$loss_type, slope=$slope, loss_num=$loss_num, alpha=$alpha"
                
                sed "s/__LOSS_TYPE__/${loss_type}/g; s/__LOSS_NUM__/${loss_num}/g; s/__ALPHA__/${alpha}/g; s/__SLOPE__/${slope}/g;" "$original_sbatch" > "$new_sbatch"
                
                sbatch "$new_sbatch"
            done
        done
    done
done
