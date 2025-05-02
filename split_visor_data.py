import json
import os


def split_visor_prompts(num_parts):
    # Load the data
    # prompts = set()
    with open('text_spatial_rel_phrases.json', 'r') as f:
        text_data = json.load(f)
        
    spatial_prompts = []
    for data in text_data:
        if data["num_objects"] == 2 and data["rel_type"] != "and":
            spatial_prompts.append(data)
            
    print(f"Total number of unique spatial prompts: {len(spatial_prompts)}")

    # Create output folder
    save_dir = f"multiprocessing_prompts_{num_parts}"
    os.makedirs(save_dir, exist_ok=True)

    # Compute the size of each chunk
    chunk_size = len(spatial_prompts) // num_parts
    remainder = len(spatial_prompts) % num_parts
    print(chunk_size, remainder)

    start = 0
    for i in range(num_parts):
        # Distribute the remainder items evenly among the first `remainder` chunks
        end = start + chunk_size + (1 if i < remainder else 0)
        print(i, start, end)
        part = spatial_prompts[start:end]

        with open(os.path.join(save_dir, f"prompts_part_{i}.json"), 'w') as f:
            json.dump(part, f, indent=2)

        print(f"Saved {len(part)} items to prompts_part_{i}.json")
        start = end
