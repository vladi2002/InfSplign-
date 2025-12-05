import os
from PIL import Image
import matplotlib.pyplot as plt
import json
import textwrap

# List your method folders and prompts
methods = ['SD1.4', 'SD2.1']  # folder names
prompts = ["prompt1", "prompt2", "prompt3"]  # just the prompt names, no .jpg

# You can also scan the files from one folder to get prompt names:
# prompts = [os.path.splitext(f)[0] for f in os.listdir(methods[0]) if f.endswith('.jpg')]
methodsdir="images/visor_alpha_0.75/"
json_filename = "visor_ablation_500"
energy ="log"
strategy="diff"

The_methods_to_depict=[
"sd1.4_relu",
"sd1.4_relu_energy_lin_strategy_diff" ,
"sd1.4_relu_energy_lin_strategy_dec" ,
"sd1.4_relu_energy_lin_strategy_both" ,
"sd1.4_relu_energy_lin_strategy_inc" ,
"sd1.4_relu_energy_log_strategy_diff" ,
"sd1.4_relu_energy_log_strategy_dec",
"sd1.4_relu_energy_log_strategy_both" ,
"sd1.4_relu_energy_log_strategy_inc"  ,
"sd1.4_relu_energy_var_strategy_diff" ,
"sd1.4_relu_energy_var_strategy_dec" ,
"sd1.4_relu_energy_var_strategy_both",
"sd1.4_relu_energy_var_strategy_inc" ,
"sd1.4_relu_energy_gibs_strategy_diff",
"sd1.4_relu_energy_gibs_strategy_dec",
"sd1.4_relu_energy_gibs_strategy_both",
"sd1.4_relu_energy_gibs_strategy_inc" 
]

Selected_act=[
"sd1.4_relu",
"sd1.4_gelu",
"sd1.4_leaky_relu",
"sd1.4_sigmoid" ,
]

Selected_strategy=[method for method in The_methods_to_depict if "diff" in method]
Selected_energy=[method for method in The_methods_to_depict if "log" in method]

Selected=Selected_strategy
num_imgs=0

methods= [method for method in os.listdir(methodsdir) if method in Selected]
methods=["sd1.4_relu"]+methods
with open(os.path.join('json_files', f'{json_filename}.json'), 'r') as f:
    visor_data = json.load(f)
    prompts, words = [], {}
    for data in visor_data:
        num_imgs+=1
        prompt = data['text']
        prompts.append(prompt)
        words[prompt] = [data['obj_1_attributes'][0], data["obj_2_attributes"][0]]
        if num_imgs>6:
            break

num_methods = len(methods)
num_prompts = len(prompts)

fig, axs = plt.subplots(num_methods, num_prompts, figsize=(3*num_prompts, 3*num_methods))
print (methods)

for row_idx, method in enumerate(methods):
    axs[row_idx, 0].set_ylabel(method, fontsize=20, rotation=0, labelpad=40, va='center')
    for col_idx, prompt in enumerate(prompts):
        img_path = os.path.join(methodsdir, method, f"{prompt}_0.png")
        img = Image.open(img_path)
        axs[row_idx, col_idx].imshow(img)
        axs[row_idx, col_idx].axis('off')
        wrapped_prompt = "\n".join(textwrap.wrap(prompt, width=18, max_lines=3))
        if row_idx == 0:
            axs[row_idx, col_idx].set_title(wrapped_prompt, fontsize=20)
    # Optionally, label each row with the method name
    #axs[row_idx, 0].set_ylabel(method, fontsize=14, rotation=0, labelpad=40, va='center')

plt.tight_layout()
plt.show()    
plt.savefig('plots/comparison.png')
