import os
import json
import random
from collections import Counter

# subset prompts are 33%

random.seed(42)

with open(os.path.join('json_files', 'visor_prompts.json'), 'r') as f:
    visor_all = json.load(f)

with open(os.path.join('json_files', 'visor_subset_1000.json'), 'r') as f:
    visor_subset = json.load(f)

with open(os.path.join('json_files', 'object_categories.json'), 'r') as f:
    object_categories = json.load(f)


def get_balanced_entries():
    spatial_rels = ["to the left of", "to the right of", "above", "below"]

    object_frequency = {}
    for entry in visor_subset:
        obj1 = entry["obj_1_attributes"][0]
        obj2 = entry["obj_2_attributes"][0]
        object_frequency[obj1] = object_frequency.get(obj1, 0) + 1
        object_frequency[obj2] = object_frequency.get(obj2, 0) + 1

    selected_objects = []
    for category, objects in object_categories.items():
        category_objects = [obj for obj in objects if obj in object_frequency]
        if category_objects:
            best_obj = max(category_objects, key=lambda obj: object_frequency.get(obj, 0))
            selected_objects.append(best_obj)

    print(f"Selected {len(selected_objects)} objects (highest frequency from each category)", selected_objects)

    object_pairs = [(obj1, obj2) for obj1 in selected_objects for obj2 in selected_objects if obj1 != obj2] # 132 combinations

    rel_counts = {rel: 0 for rel in spatial_rels}
    target_per_rel = len(object_pairs) // len(spatial_rels)

    results = []
    used_triplets = set()

    for obj1, obj2 in object_pairs:
        # find entries with these objects and any relation
        candidates = []

        # if it finds the object pair in the subset, it will not check the full dataset
        find_candidates(visor_subset, candidates, obj1, obj2, rel_counts, spatial_rels, target_per_rel, used_triplets)

        if not candidates:  # if the subset doesn't contain the object pair, get all 8 occurrences of the object pair from the full dataset
            find_candidates(visor_all, candidates, obj1, obj2, rel_counts, spatial_rels, target_per_rel, used_triplets)

        if candidates:
            # prefer less frequent relations
            sorted_candidates = sorted(candidates, key=lambda e: rel_counts[e["rel_type"]])
            selected = sorted_candidates[0]
            key = (selected["obj_1_attributes"][0], selected["obj_2_attributes"][0], selected["rel_type"])

            results.append(selected)
            used_triplets.add(key)
            rel_counts[selected["rel_type"]] += 1

    return results


def find_candidates(visor_set, candidates, obj1, obj2, rel_counts, spatial_rels, target_per_rel, used_triplets):
    for entry in visor_set:
        entry_obj1 = entry["obj_1_attributes"][0]
        entry_obj2 = entry["obj_2_attributes"][0]
        rel = entry["rel_type"]

        if ((entry_obj1 == obj1 and entry_obj2 == obj2) or
            (entry_obj1 == obj2 and entry_obj2 == obj1)) and rel in spatial_rels:
            key = (entry_obj1, entry_obj2, rel)
            if key not in used_triplets and rel_counts[rel] < target_per_rel:
                candidates.append(entry)


balanced_entries = get_balanced_entries()

subset_count = 0
for entry in visor_subset:
    if entry in balanced_entries:
        subset_count += 1

print(f"\nTotal entries: {len(balanced_entries)}")
print(f"Entries from subset: {subset_count} ({subset_count / len(balanced_entries) * 100:.1f}%)")


# relation distribution
rel_counter = Counter()
for entry in balanced_entries:
    rel_counter[entry["rel_type"]] += 1
print("\nRelation distribution:")
for rel, count in rel_counter.items():
    print(f"{rel}: {count}")

# object distribution
obj_counter = Counter()
for entry in balanced_entries:
    obj_counter[entry["obj_1_attributes"][0]] += 1
    obj_counter[entry["obj_2_attributes"][0]] += 1

print(f"\nUnique objects used: {len(obj_counter)}\n")
print("Object frequency distribution:")
for obj, count in sorted(obj_counter.items(), key=lambda x: -x[1]):
    print(f"{obj}: {count}")

# # Save the balanced entries to a JSON file
# output_file = os.path.join('json_files', 'visor_ablation_132.json')
# with open(output_file, 'w') as f:
#     json.dump(balanced_entries, f, indent=4)
