from pipeline_batch import get_config, run_ablation_leaky_relu_attn_maps

if __name__ == "__main__":
    config = get_config()
    run_ablation_leaky_relu_attn_maps(config)