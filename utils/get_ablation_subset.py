import os
import json
import random
from collections import Counter

# subset prompts are 33%

random.seed(42)

# Subset of 1000 entries we failed to predict correctly (500 with 1 correct object and 500 with 0 correct objects)
with open(os.path.join('../json_files', 'visor_subset_1000.json'), 'r') as f:
    visor_subset = json.load(f)

with open(os.path.join('../json_files', 'object_categories.json'), 'r') as f:
    object_categories = json.load(f)


def get_balanced_entries(target_count=500):
    spatial_rels = ["to the left of", "to the right of", "above", "below"]

    available_objects = set()
    for entry in visor_subset:
        available_objects.add(entry["obj_1_attributes"][0])
        available_objects.add(entry["obj_2_attributes"][0])

    selected_objects = []
    for category, objects in object_categories.items():
        category_objects = [obj for obj in objects if obj in available_objects]
        selected_objects.extend(category_objects)

    print(f"Selected {len(selected_objects)} unique objects across all categories")

    object_usage = {obj: 0 for obj in selected_objects}

    pairs_needed = target_count // len(spatial_rels)
    pairs_per_rel = pairs_needed + (1 if target_count % len(spatial_rels) > 0 else 0)

    rel_counts = {rel: 0 for rel in spatial_rels}
    results = []
    used_triplets = set()

    entries_by_objects = {}
    for entry in visor_subset:
        obj1 = entry["obj_1_attributes"][0]
        obj2 = entry["obj_2_attributes"][0]
        rel = entry["rel_type"]

        if obj1 in selected_objects and obj2 in selected_objects and rel in spatial_rels:
            key = (obj1, obj2)
            if key not in entries_by_objects:
                entries_by_objects[key] = []
            entries_by_objects[key].append(entry)

    while len(results) < target_count and entries_by_objects:
        min_usage = min(object_usage.values())
        rare_objects = [obj for obj, usage in object_usage.items() if usage == min_usage]

        candidate_entries = []
        for obj1 in rare_objects:
            for obj2 in selected_objects:
                if obj1 != obj2:
                    if (obj1, obj2) in entries_by_objects:
                        for entry in entries_by_objects[(obj1, obj2)]:
                            if entry["rel_type"] in spatial_rels:
                                candidate_entries.append(entry)
                    if (obj2, obj1) in entries_by_objects:
                        for entry in entries_by_objects[(obj2, obj1)]:
                            if entry["rel_type"] in spatial_rels:
                                candidate_entries.append(entry)

        if not candidate_entries:
            for entries in entries_by_objects.values():
                candidate_entries.extend(entries)

        if not candidate_entries:
            break

        candidate_entries.sort(key=lambda e: rel_counts[e["rel_type"]])

        for entry in candidate_entries:
            obj1 = entry["obj_1_attributes"][0]
            obj2 = entry["obj_2_attributes"][0]
            rel = entry["rel_type"]
            key = (obj1, obj2, rel)

            if key not in used_triplets and rel_counts[rel] < pairs_per_rel:
                results.append(entry)
                used_triplets.add(key)
                rel_counts[rel] += 1
                object_usage[obj1] += 1
                object_usage[obj2] += 1

                entries_by_objects.pop((obj1, obj2), None)
                entries_by_objects.pop((obj2, obj1), None)
                break

        if len(results) >= target_count:
            break

    if len(results) < target_count:
        remaining_entries = [e for e in visor_subset if e not in results and e["rel_type"] in spatial_rels]
        remaining_entries.sort(key=lambda e: (
            object_usage.get(e["obj_1_attributes"][0], float('inf')) +
            object_usage.get(e["obj_2_attributes"][0], float('inf')),
            rel_counts[e["rel_type"]]
        ))

        for entry in remaining_entries:
            rel = entry["rel_type"]
            obj1 = entry["obj_1_attributes"][0]
            obj2 = entry["obj_2_attributes"][0]
            key = (obj1, obj2, rel)

            if key not in used_triplets:
                results.append(entry)
                used_triplets.add(key)
                rel_counts[rel] += 1
                if obj1 in object_usage: object_usage[obj1] += 1
                if obj2 in object_usage: object_usage[obj2] += 1

                if len(results) >= target_count:
                    break

    return results


balanced_entries = get_balanced_entries()

# # Save to JSON file
# output_file = os.path.join('json_files', 'visor_subset_500.json')
# with open(output_file, 'w') as f:
#     json.dump(balanced_entries, f, indent=4)

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

print("Object frequency distribution by category:")
# Create a counter for objects in the subset
subset_obj_counter = Counter()
for entry in visor_subset:
    subset_obj_counter[entry["obj_1_attributes"][0]] += 1
    subset_obj_counter[entry["obj_2_attributes"][0]] += 1

for category, objects in object_categories.items():
    category_objects = {obj: (obj_counter.get(obj, 0), subset_obj_counter.get(obj, 0))
                       for obj in objects if obj in obj_counter}
    if category_objects:
        print(f"\n{category} ({len(category_objects)} objects):")
        for obj, (count, total) in sorted(category_objects.items(), key=lambda x: -x[1][0]):
            print(f"  {obj}: {count}/{total}")
