
#!/bin/bash

# Wrapper script to submit sbatch jobs with different parameters



for model in sd1.4 sd2.1; do
    for alpha in 1.5; do
        for margin in 0.25; do
            for lambda_spatial in 0.5; do
                for lambda_presence in 1.0; do
                    for lambda_balance in 1.0; do
                        for loss in sigmoid relu gelu leaky_relu linear; do
                            img_id="visor_full_alpha_${alpha}_margin_${margin}_lspatial${lambda_spatial}_lpresence${lambda_presence}_lbalance${lambda_balance}_loss_${loss}"
                            echo "Submitting with model=$model, $img_id"
                            sbatch --export=ALL,MODEL=$model,MARGIN=$margin,ALPHA=$alpha,LAMBDA_SPATIAL=$lambda_spatial,LAMBDA_PRESENCE=$lambda_presence,LAMBDA_BALANCE=$lambda_balance,IMG_ID=$img_id,LOSS=$loss sieger_visor.sbatch
                        done
                    done
                done
            done
        done
    done
done


# for model in sdxl; do
#     for alpha in 1.5; do
#         for margin in 0.5; do
#             for lambda_spatial in 0 0.5; do
#                 for lambda_presence in 0 1.0; do
#                     for lambda_balance in 0 1.0; do
#                         img_id="visor_full_alpha_${alpha}_margin_${margin}_lspatial${lambda_spatial}_lpresence${lambda_presence}_lbalance${lambda_balance}"
#                         echo "Submitting with model=$model, $img_id"
#                         sbatch --export=ALL,MODEL=$model,MARGIN=$margin,ALPHA=$alpha,LAMBDA_SPATIAL=$lambda_spatial,LAMBDA_PRESENCE=$lambda_presence,LAMBDA_BALANCE=$lambda_balance,IMG_ID=$img_id sieger_eval.sbatch
#                     done
#                 done
#             done
#         done
#     done
# done