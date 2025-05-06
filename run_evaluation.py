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


if __name__ == "__main__":
    config = get_config()
    run_evaluation(config, relationship=None)
