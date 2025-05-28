from pipeline_batch import get_config, get_spatial_loss_stats

if __name__ == "__main__":
    config = get_config()
    get_spatial_loss_stats(config)