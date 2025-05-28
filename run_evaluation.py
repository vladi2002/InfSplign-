import os
import json

from combined_pipeline_multiprocessing import get_config
from utils.visor_utils import evaluate, show_visor_results
from utils.pipeline_utils import load_object_detection_ann, \
    initialize_object_detection_model, create_object_detection_annotations
from UniDet_eval.spatial_eval import t2i_spatial_score


def run_evaluation(config, relationship=None):
    benchmark = config.benchmark
    if benchmark == 'visor':
        run_visor_evaluation(config, relationship=relationship)
    elif benchmark == 't2i':
        run_t2i_evaluation(config, relationship=relationship)


def run_visor_evaluation(config, relationship=None):
    model = config.model
    json_filename = config.json_filename
    img_id = config.img_id
    model_name = f"{model}_{img_id}"

    with open(os.path.join('json_files', f'{json_filename}.json'), 'r') as f:
        prompts_data = json.load(f)

    obj_det_ann_path = os.path.join('objdet_results', 'visor', model_name, f'{json_filename}.json')
    print(f"obj_det_ann_path: {obj_det_ann_path}")
    if not os.path.isfile(obj_det_ann_path):
        processor, obj_det_model = initialize_object_detection_model(config)
        create_object_detection_annotations(config, prompts_data, processor, obj_det_model, relationship=relationship)
    obj_det_ann_natural = load_object_detection_ann(config, relationship=relationship)

    visor_table = []
    visor_table.append(evaluate(obj_det_ann_natural, prompts_data, model_name))
    show_visor_results(visor_table, config)
    

def run_t2i_evaluation(config, relationship=None):
    # t2i_score, num_images = 
    t2i_spatial_score(config, relationship=relationship)


def run_t2i_evaluation_sweep(config, relationship=None):
    # t2i_spatial_score(config, relationship=relationship)
    for loss in ["relu", "gelu", "sigmoid"]:
        margin = 0.25
        img_id = f"{loss}_m={margin}_centr_mean"
        config.img_id = img_id
        t2i_spatial_score(config, relationship=relationship)
        
# TODO: changed based on the naming conventions of the current runs
def run_visor_evaluation_sweep(config, relationship=None):       
    for loss in ["relu", "gelu", "sigmoid"]:
        for loss_num in [1, 2, 3]:
            for alpha in [1.0, 2.0, 5.0]:
                if alpha == 1.0 and loss_num == 1: # for the 9 attn maps ablation with no wt
                    continue
                if True:
                    no_wt = "_no_wt"
                else:
                    no_wt = ""
                img_id = f"loss_{loss}_loss_num_{loss_num}_alpha_{alpha}_9_attn_maps{no_wt}_ablation_132"
                config.img_id = img_id
                run_visor_evaluation(config, relationship=relationship)


if __name__ == "__main__":
    config = get_config()
    # run_evaluation(config, relationship=None)
    run_visor_evaluation_sweep(config, relationship=None)
