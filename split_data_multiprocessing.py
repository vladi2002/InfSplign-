import json
import os


def split_prompts(num_gpus, benchmark, spatial_prompts):
    print(f"Total number of unique spatial prompts: {len(spatial_prompts)}")

    save_dir = os.path.join('data_splits', f'{benchmark}', f"multiprocessing_{num_gpus}")
    os.makedirs(save_dir, exist_ok=True)

    chunk_size = len(spatial_prompts) // num_gpus
    remainder = len(spatial_prompts) % num_gpus
    print(chunk_size, remainder)

    start = 0
    for i in range(num_gpus):
        end = start + chunk_size + (1 if i < remainder else 0)
        print(i, start, end)
        part = spatial_prompts[start:end]

        with open(os.path.join(save_dir, f"prompts_part_{i}.json"), 'w') as f:
            json.dump(part, f, indent=2)

        print(f"Saved {len(part)} items to prompts_part_{i}.json")
        start = end
