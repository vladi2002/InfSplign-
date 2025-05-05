import json
import os


def define_natural_unnatural_prompt_split(data, object_categories):
    """
    Create a split of natural and unnatural prompts.
    Each prompt is in the form of a triplet: (objA, rel, objB). rel is a spatial relation between objA and objB.
    For now, I am using a subset of all combinations. Since all 80 coco categories are used for the objects,
    I defined the combinations based on the coco supercategories. To make it easier, the first object in a combination
    is always "person". The natural prompts I define as combinations describing realistic, observable relationships and
    the unnatural prompts cannot occur in reality.
    """
    # TODO: Use an LLM to generate the natural and unnatural prompts. Here using VISOR text_spatial_rel_phrases.json

    # coco supercategory combinations
    category_combinations = [
        ("person", "vehicle"),
        ("person", "animal"),
        ("person", "accessory"),
        ("person", "furniture"),
        ("person", "electronic"),
        ("person", "food"),
        ("person", "indoor"),
        ("person", "appliance"),
    ]

    object_combinations = list([object_categories[combination[1]] for combination in category_combinations])
    object_combinations = [obj for objs in object_combinations for obj in objs]
    natural_combinations = [("person", obj) for obj in object_combinations]

    # all relations -> {'single', 'to the left of', 'below', 'and', 'to the right of', 'above'}
    relations = ["to the left of", "to the right of"]
    relation_negation = {
        "to the left of": "to the right",
        "to the right of": "to the left",
        "above": "below",
        "below": "above"
    }
    natural_relations = {comb: relations for comb in natural_combinations}

    natural_entries = []
    unnatural_entries = []

    for entry in data:
        obj1, obj2 = entry["obj_1_attributes"][0], entry["obj_2_attributes"][0]
        rel_type = entry["rel_type"]

        if not relation_negation.get(rel_type, ''):
            continue
        if "of" in rel_type:
            comp_rel = rel_type.replace("of", "").strip()
        else:
            comp_rel = rel_type
        entry["prompt"] = f"{obj1} {comp_rel} | {obj2} {relation_negation[rel_type]}"

        if (obj1, obj2) in natural_relations and rel_type in natural_relations[(obj1, obj2)]:
            natural_entries.append(entry)
        elif obj1 != "person":
            continue
        elif (obj1, obj2) in natural_relations and rel_type != "and" and rel_type not in natural_relations[(obj1, obj2)]:
            unnatural_entries.append(entry)

    with open('../natural.json', 'w') as f:
        json.dump(natural_entries, f, indent=4)

    with open('../unnatural.json', 'w') as f:
        json.dump(unnatural_entries, f, indent=4)


def create_prompt_split():
    with open('../json_files/text_spatial_rel_phrases.json', 'r') as f:
        text_data = json.load(f)

    with open('../object_categories.json', 'r') as f:
        object_cats = json.load(f)

    define_natural_unnatural_prompt_split(text_data, object_cats)


def extract_object_detection_annotations(spatial_prompts, split, model, fname, od, th):
    img_ids = []
    for prompt in spatial_prompts:
        img_ids.append(prompt["unique_id"])

    obj_dets_path = "objdet_results/results_{}_{}_{}_{}.json".format(model, fname, od, th)

    if not os.path.exists(obj_dets_path):
        print("Results Path does not exist.")

    with open(obj_dets_path, "r") as f:
        obj_det_anns = json.load(f)

    # get the corresponding image ids
    obj_det_ann_img_ids = [img_id for img_id in obj_det_anns.keys() if int(img_id.split("_")[0]) in img_ids]

    obj_annotations = {}
    for obj_det_id in obj_det_ann_img_ids:
        obj_annotations[obj_det_id] = obj_det_anns[obj_det_id]

    with open(f'objdet_results/visor_{split}.json', 'w') as f:
        json.dump(obj_annotations, f, indent=4)


if __name__ == "__main__":
    create_prompt_split()
