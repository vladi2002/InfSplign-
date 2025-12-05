#!/bin/bash

loss_types=("relu")
alpha_values=("1.0")
energy_type=("log")
strategy_type=("dec")

original_sbatch="/var/scratch/srastega/Thesis-Splign/job_scripts/sd1.4_energy.sbatch"
sbatch_files_dir="/var/scratch/srastega/Thesis-Splign/sbatch_files"

for energy in "${energy_type[@]}"; do
for strategy in "${strategy_type[@]}"; do
for loss_type in "${loss_types[@]}"; do
    for alpha in "${alpha_values[@]}"; do
        echo "Submitting job with loss_type=$loss_type, alpha=$alpha, energy=$energy, strategy=$strategy"
        
        new_sbatch="${sbatch_files_dir}/sd1.4_${loss_type}_${alpha}_${energy}_${strategy}_visor_ablation.sbatch"
        
        sed "s/__LOSS_TYPE__/${loss_type}/g; s/__ALPHA__/${alpha}/g; s/__ENERGY__/${energy}/g; s/__STRATEGY__/${strategy}/g" "$original_sbatch" > "$new_sbatch"
        
        sbatch "$new_sbatch"
    done
done
done
done
