import csv
import os
import re

# Configuration (matches your generation loop)
losses = ["relu", "gelu"]
loss_nums = [1, 2, 3]
alphas = [0.25, 0.5, 1.0, 2.0] #, 1.0, 2.0, 5.0]
no_wt = "_no_wt"

# Output file list based on naming convention
# Generate folder names and corresponding file names
base_path = os.path.join("objdet_results", "visor")
found_files = []
for loss in losses:
    for loss_num in loss_nums:
        for alpha in alphas:
            # for slope in slopes:
            # Construct folder path
            folder_name = f"sd1.4_loss_{loss}_loss_num_{loss_num}_alpha_{alpha}_6_attn_maps{no_wt}_ablation_132"
            # Construct file name
            file_name = f"visor_table_loss_{loss}_loss_num_{loss_num}_alpha_{alpha}_6_attn_maps{no_wt}_ablation_132.txt"
            
            # Check if file exists in the folder
            full_path = os.path.join(base_path, folder_name, file_name)
            if os.path.exists(full_path):
                found_files.append((folder_name, file_name))

print(f"Found {len(found_files)} files to process.")

# Process available files
parsed_data = []
for folder_name, file_name in found_files:
    print(f"Processing file: {folder_name}/{file_name}")
    with open(os.path.join(base_path, folder_name, file_name), "r") as f:
        lines = f.readlines()
        # Filter out any lines that contain only dashes and/or whitespace
        lines = [line for line in lines if not re.match(r'^[\s\-]+$', line.strip())]
        split_lines = [list(filter(None, re.split(r"\s{2,}", line.strip()))) for line in lines]
        # Skip the header (first line) but include all data rows
        if len(split_lines) >= 2:
            parsed_data.extend(split_lines[1:])

# Swap columns 2 and 3 (VISOR_cond and VISOR_uncond) for each row
for row in parsed_data:
    if len(row) > 3:  # Make sure the row has enough columns
        row[2], row[3] = row[3], row[2]

# Write to CSV
output_csv = "spatial_eval_summary_sd1.4_relu_gelu_alpha.csv"
with open(output_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "OA", "VISOR_uncond", "VISOR_cond", "VISOR_1", "VISOR_2", "VISOR_3", "VISOR_4", "Num_Imgs"])
    writer.writerows(parsed_data)
    
# losses = ["leaky_relu"]
# loss_nums = [1, 2, 3]
# alphas = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]
# slopes = [0.05, 0.1, 0.25, 0.5]
# no_wt = "_no_wt"

# losses = ["relu", "gelu"]
# loss_nums = [1, 2, 3]
# alphas = [0.1, 0.25, 0.5, 0.75] #, 1.0, 2.0, 5.0]
# no_wt = "_no_wt"

# losses = ["sigmoid"]
# loss_nums = [1, 2, 3]
# alphas = [6.0, 7.0, 8.0, 9.0, 10.0] # [1.0, 2.0, 5.0]
# no_wt = "_no_wt"