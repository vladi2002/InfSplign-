from pipeline_batch import get_config, run_ablation_loss_num

if __name__ == "__main__":
    config = get_config()
    run_ablation_loss_num(config)