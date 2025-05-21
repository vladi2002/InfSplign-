from pipeline_batch import get_config, run_ablation_spatial_loss_intervention

if __name__ == "__main__":
    config = get_config()
    run_ablation_spatial_loss_intervention(config)