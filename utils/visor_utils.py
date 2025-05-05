import itertools
import numpy as np
from tabulate import tabulate


def increment_dict(d, k, v, inc_type="list"):
    inc = [v] if inc_type == "list" else v
    if k in d:
        d[k] = d[k] + inc  # concatenating visor scores -> e.g. [1, 1, 0, 1]
    else:
        d[k] = inc
    return d


def compute_recall(obj1, obj2, detected, N):
    if obj1 in detected and obj2 in detected:
        count = 2
    elif obj1 in detected or obj2 in detected:
        count = 1
    else:
        count = 0

    return count / N, count


def get_visor_n(visor_by_uniq_id):
    visor_1, visor_2, visor_3, visor_4 = 0, 0, 0, 0
    for uniq_id, scores in visor_by_uniq_id.items():
        if sum(scores) >= 4:
            visor_4 = visor_4 + 1
        if sum(scores) >= 3:
            visor_3 = visor_3 + 1
        if sum(scores) >= 2:
            visor_2 = visor_2 + 1
        if sum(scores) >= 1:
            visor_1 = visor_1 + 1

    NUM_UNIQ = len(visor_by_uniq_id)  # number of unique images

    return [100 * visor_1 / NUM_UNIQ, 100 * visor_2 / NUM_UNIQ, 100 * visor_3 / NUM_UNIQ, 100 * visor_4 / NUM_UNIQ]


def get_visor_spatial(results, text_data):
    objacc_both, objacc_A, objacc_B = 0, 0, 0
    visor_cond, visor_uncond = 0, 0
    both_count, count = 0, 0
    visor_by_uniq_id = {}

    for img_id, rr in results.items():
        uniq_id = img_id.split("_")[0]
        # had to change this line item["unique_id"] to item["text"] !!!!!!!!!!!!!!
        ann = [item for item in text_data if item["text"] == uniq_id][0]
        obj1 = ann["obj_1_attributes"][0]
        obj2 = ann["obj_2_attributes"][0]
        rel = ann["rel_type"]
        N = ann["num_objects"]
        recall = rr["recall"] / N

        if rel == "and" or N != 2:
            continue

        detected = rr["classes"]
        det_objA = int(obj1 in detected)
        det_objB = int(obj2 in detected)
        det_both = int(obj1 in detected and obj2 in detected)
        sra = rr["sra"]
        objacc_both = objacc_both + det_both
        objacc_A = objacc_A + det_objA
        objacc_B = objacc_B + det_objB

        visor_cond = visor_cond + det_both * sra
        both_count = both_count + det_both
        count = count + 1
        visor_by_uniq_id = increment_dict(visor_by_uniq_id, uniq_id, det_both * sra)

    # visor scores
    # print(visor_cond, both_count)
    visor_cond = 100 * visor_cond / both_count if both_count > 0 else 0  # in how many images both objects are generated (safe if none)
    visor_n = get_visor_n(visor_by_uniq_id)

    # objacc scores
    objacc = [100 * objacc_A / count, 100 * objacc_B / count, 100 * objacc_both / count]

    return visor_cond, visor_n, objacc
    # return visor_cond, objacc


def process_detection(outs, obj1, obj2, rel, verbose=False):
    objects = [obj1, obj2]
    boxes, scores, labels = outs[0]["boxes"], outs[0]["scores"], outs[0]["labels"]

    det_bbox, det_scores, det_labels = [], [], []
    for box, score, label in zip(boxes, scores, labels):
        if score > 0.1:
            det_bbox.append(box.tolist())
            det_scores.append(score.tolist())
            det_labels.append(objects[label.item()])

    det_centroid = [((box[0]+box[2])/2, (box[1]+box[3])/2) for box in det_bbox]
    N = len(det_centroid)

    if obj1 in det_labels and obj2 in det_labels:
        recall = 2
    elif obj1 in det_labels:
        recall = 1
    elif obj2 in det_labels:
        recall = 1
    else:
        recall = 0

    sra = 0  # spatial relationship accuracy
    if obj1 in det_labels and obj2 in det_labels:
        idx1 = np.where(np.array(det_labels)==obj1)[0]
        idx2 = np.where(np.array(det_labels)==obj2)[0]

        # atleast one of the bbox pairs should follow the relationship
        for i1, i2 in itertools.product(idx1.tolist(), idx2.tolist()):
            xdist = det_centroid[i1][0] - det_centroid[i2][0]
            ydist = det_centroid[i1][1] - det_centroid[i2][1]
            if rel == "to the left of" and xdist < 0:
                sra = 1
                break
            if rel == "to the right of" and xdist > 0:
                sra = 1
                break
            if rel == "above" and ydist < 0:
                sra = 1
                break
            if rel == "below" and ydist > 0:
                sra = 1
                break
    if verbose:
        return {
            "classes": det_labels, "boxes": det_bbox, "labels": det_labels, "centroid": det_centroid, "recall": recall,
            "sra": sra, "text": "{} {} {}".format(obj1, rel, obj2)
        }
    return {
        "classes": det_labels, "centroid": det_centroid, "recall": recall, "sra": sra, "text": "{} {} {}".format(obj1, rel, obj2)
        }


def evaluate(obj_det_ann, prompts_data, model): # t2i_score
    visor_cond, visor_n, objacc = get_visor_spatial(obj_det_ann, prompts_data)
    # visor_cond, objacc = get_visor_spatial(obj_det_ann, prompts_data)

    visor_uncond = visor_cond * objacc[2]

    visor_table_data = [
        model,
        f'{objacc[2]:.3f}', f'{visor_cond:.3f}', f'{0.01 * (visor_uncond):.3f}',
        f'{visor_n[0]:.3f}', f'{visor_n[1]:.3f}', f'{visor_n[2]:.3f}', f'{visor_n[3]:.3f}',
        str(len(obj_det_ann)) # f'{t2i_score:.3f}'
    ]

    return visor_table_data


def show_visor_results(visor_table):
    table = tabulate(
            visor_table,
            headers=[
                'Model',
                'OA', 'VISOR_cond', 'VISOR_uncond',
                'VISOR_1', 'VISOR_2', 'VISOR_3', 'VISOR_4',
                'Num_Imgs'
            ] # 'T2I-Comp-Bench (spatial)', 
        )
        
    print(table)
    # with open('sdxl_visor_all_results.txt', 'w') as f:
    #     f.write(table)
