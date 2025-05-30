#!/bin/bash

original_sbatch="/tudelft.net/staff-umbrella/StudentsCVlab/vchatalbasheva/Thesis-Splign/job_scripts/sd2.1_energy_6attn_maps_no_wt_visor_ablation.sbatch"
sbatch_files_dir="/tudelft.net/staff-umbrella/StudentsCVlab/vchatalbasheva/Thesis-Splign/sbatch_files"

# best performing configurations from the ablation
# tuple "loss_type:loss_num:alpha1,alpha2,..."
configs=(
    "relu:1:1.0"
    "relu:2:1.0"
    "relu:3:1.0"
    "gelu:1:2.0"
    "gelu:2:2.0"
    "gelu:3:1.0"
    "sigmoid:1:2.0"
    "sigmoid:2:5.0"
    "sigmoid:3:5.0"
)

for config in "${configs[@]}"; do
    loss_type=$(echo "$config" | cut -d: -f1)
    loss_num=$(echo "$config" | cut -d: -f2)
    alphas=$(echo "$config" | cut -d: -f3)
    
    IFS=',' read -ra alpha_array <<< "$alphas"
    for alpha in "${alpha_array[@]}"; do
        echo "Submitting job with loss_type=$loss_type, loss_num=$loss_num, alpha=$alpha"
        
        new_sbatch="${sbatch_files_dir}/sd2.1_energy_${loss_type}_${loss_num}_${alpha}_visor_ablation.sbatch"
        
        sed "s/__LOSS_TYPE__/${loss_type}/g; s/__LOSS_NUM__/${loss_num}/g; s/__ALPHA__/${alpha}/g" "$original_sbatch" > "$new_sbatch"
        
        sbatch "$new_sbatch"
    done
done
