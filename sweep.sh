
#!/bin/bash

# Wrapper script to submit sbatch jobs with different parameters

for num_inference_steps in 500 1000; do
    for sg_t_start in 0 25 50 100; do
        sg_t_end=$((sg_t_start + num_inference_steps / 2))
        echo "Submitting with num_inference_steps=$num_inference_steps, sg_t_start=$sg_t_start, sg_t_end=$sg_t_end"
        sbatch --export=NUM_INFERENCE_STEPS=$num_inference_steps,SG_T_START=$sg_t_start,SG_T_END=$sg_t_end sieger_visor.sbatch
    done
done
