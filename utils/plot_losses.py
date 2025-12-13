import pandas as pd
import sns
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import re
import json
import numpy as np
import seaborn as sns
from PIL import Image

from scipy.special import expit as sigmoid
from scipy.special import erf

import warnings
warnings.filterwarnings("ignore")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_diffusion_statistics(log_file_path, loss, prompt):
    pattern = r"Timestep (\d+), spatial loss: ([-\d\.]+), grad min: ([-\d\.]+), grad mean: ([-\d\.]+), grad max: ([-\d\.]+)"

    timesteps = []
    spatial_loss = []
    grad_min = []
    grad_mean = []
    grad_max = []

    with open(log_file_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                timesteps.append(int(match.group(1)))
                spatial_loss.append(float(match.group(2)))
                grad_min.append(float(match.group(3)))
                grad_mean.append(float(match.group(4)))
                grad_max.append(float(match.group(5)))

    fig, axs = plt.subplots(2, 1, figsize=(12, 15))

    axs[0].plot(timesteps, grad_min, 'b-', label='Min')
    axs[0].plot(timesteps, grad_mean, 'g-', label='Mean')
    axs[0].plot(timesteps, grad_max, 'r-', label='Max')
    axs[0].set_xlabel('timestep')
    axs[0].set_title('grad')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(timesteps, spatial_loss, 'm-', linewidth=2)
    axs[1].set_xlabel('timestep')
    axs[1].set_title(f'spatial loss: {loss}')
    axs[1].grid(True)

    fig.suptitle(f"Prompt: {prompt}", fontsize=16)

    plt.tight_layout()
    save_dir = "logs/logs_imgs"
    os.makedirs(save_dir, exist_ok=True)
    fig_name = os.path.join(save_dir, os.path.basename(log_file_path).split(".log")[0] + ".png")
    # print(fig_name)
    plt.savefig(fig_name)
    # plt.show()


def plot_sigmoid_alpha(margins, alphas):
    delta = torch.linspace(-1, 1, steps=500)

    # Create a single figure for all alpha values
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 14})

    # Use colormap to get different colors for each alpha
    cmap = plt.cm.viridis
    colors = [cmap(i/len(alphas)) for i in range(len(alphas))]

    # Plot sigmoid for each alpha value
    for i, alpha in enumerate(alphas):
        sigmoid_loss = torch.sigmoid(-alpha * delta)
        plt.plot(delta.numpy(), sigmoid_loss.numpy(),
                 label=f"$\\sigma$ (α={alpha})",
                 linewidth=2.5,
                 color=colors[i])

    # Add vertical lines for reference
    margin = margins[0]  # Use first margin for reference line
    plt.axvline(x=margin, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle=':', linewidth=1)

    plt.title("Effect of α on Sigmoid Loss Function")
    plt.xlabel("Δ (Centroid Difference)")
    plt.ylabel("Loss Value")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-1, 1)
    plt.ylim(0, None)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("logs", f"sigmoid_alpha_comparison.png"))
    # plt.show()


def plot_sigmoid_alpha_gradients(margins, alphas):
    delta = torch.linspace(-1, 1, steps=500)

    # Create a single figure for all alpha values
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 14})

    # Use colormap to get different colors for each alpha
    cmap = plt.cm.viridis
    colors = [cmap(i/len(alphas)) for i in range(len(alphas))]

    # Plot sigmoid gradient for each alpha value
    for i, alpha in enumerate(alphas):
        sigmoid_val = torch.sigmoid(-alpha * delta)
        sigmoid_gradient = -alpha * sigmoid_val * (1 - sigmoid_val)
        plt.plot(delta.numpy(), abs(sigmoid_gradient.numpy()),
                 label=f"$\\nabla$ $\\sigma$ ($\\alpha$={alpha})",
                 linewidth=2.5,
                 color=colors[i])

    # Add vertical lines for reference
    margin = margins[0]  # Use first margin for reference line
    plt.axvline(x=margin, color='gray', linestyle='--', linewidth=1)
    plt.axvline(x=0, color='black', linestyle=':', linewidth=1)

    plt.title("Effect of α on Sigmoid Gradient")
    plt.xlabel("Δ (Centroid Difference)")
    plt.ylabel("Gradient Value")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-1, 1)
    plt.ylim(0, None)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("logs", f"sigmoid_alpha_gradient_comparison.png"))
    # plt.show()


def plot_spatial_losses_behaviour(margins, alphas, slopes):
    delta = torch.linspace(-1, 1, steps=500)
    plt.rcParams.update({'font.size': 14})

    for margin in margins:
        for alpha in alphas:
            for slope in slopes:
                relu_loss = F.relu(alpha * (margin - delta))
                squared_relu_loss = F.relu(alpha * (margin - delta)) ** 2
                leaky_relu_loss = F.leaky_relu(alpha * (margin - delta), negative_slope=slope)
                gelu_loss = F.gelu(alpha * (margin - delta))
                sigmoid_loss = torch.sigmoid(- alpha * delta)

                # Plotting
                plt.figure(figsize=(8, 5))
                plt.plot(delta.numpy(), relu_loss.numpy(), label=f"ReLU (margin={margin})", linewidth=2)
                # plt.plot(delta.numpy(), squared_relu_loss.numpy(), label=f"Squared ReLU (margin={margin}, α={alpha})", linewidth=2)
                plt.plot(delta.numpy(), leaky_relu_loss.numpy(), label=f"Leaky ReLU (margin={margin}, slope={slope})", linewidth=2)
                plt.plot(delta.numpy(), gelu_loss.numpy(), label=f"GELU (margin={margin})", linewidth=2)
                plt.plot(delta.numpy(), sigmoid_loss.numpy(), label=f"Sigmoid (margin=0)", linewidth=2)

                plt.axvline(x=margin, color='gray', linestyle='--', linewidth=1)
                plt.axvline(x=0, color='black', linestyle=':', linewidth=1)

                plt.title(f"Loss Function Values over Centroid Difference (Δ) | α={alpha}")
                plt.xlabel("Δ (Centroid Difference)")
                plt.ylabel("Loss Value")
                plt.xlim(-1, 1)
                plt.ylim(-0.2, None)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.legend()
                plt.legend(loc='upper right')
                plt.grid(True)
                plt.tight_layout()
                # plt.savefig(os.path.join("logs", f"loss_behaviour_margin_{margin}_alpha_{alpha}_leaky_{slope}.png"))
                plt.show()


def gradient_spatial_loss_behaviour(margins, alphas, slopes):
    def d_relu(x, margin, alpha):
        return np.where(alpha * (margin - x) > 0, -alpha, 0)

    def d_leaky_relu(x, margin, alpha, slope):
        z = alpha * (margin - x)
        return np.where(z >= 0, -alpha, -slope * alpha)

    def d_gelu(x, margin, alpha):
        x_shifted = alpha * (margin - x)
        phi = np.exp(-x_shifted ** 2 / 2) / np.sqrt(2 * np.pi)
        return -alpha * (0.5 * (1 + erf(x_shifted / np.sqrt(2))) + x_shifted * phi)

    def d_sigmoid(x, alpha):
        s = sigmoid(alpha * x)
        return alpha * s * (1 - s)

    delta = np.linspace(-1, 1, 500)
    plt.rcParams.update({'font.size': 14})

    for margin in margins:
        for alpha in alphas:
            for slope in slopes:
                plt.figure(figsize=(8, 5))
                plt.plot(delta, abs(d_relu(delta, margin, alpha)), label=f"ReLU")
                plt.plot(delta, abs(d_leaky_relu(delta, margin, alpha, slope=slope)), label=f"Leaky ReLU, slope={slope})")
                plt.plot(delta, abs(d_gelu(delta, margin, alpha)), label=f"GELU")
                plt.plot(delta, abs(d_sigmoid(delta, alpha)), label=f"Sigmoid")

                plt.title(f"Gradient Magnitude vs. Δ (Centroid Difference) | α={alpha}")
                plt.xlabel("Δ (Centroid Difference)")
                plt.ylabel("Gradient Magnitude $\\frac{\\partial \\mathcal{L}}{\\partial z_t}$")
                plt.xlim(-1, 1)
                plt.ylim(0, None)
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.axvline(x=margin, linestyle="--", color="gray", linewidth=1)
                plt.axvline(x=0, linestyle=":", color="black", linewidth=1)
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join("logs", f"gradient_loss_behaviour_margin_{margin}_alpha_{alpha}_leaky_slope_{slope}.png"))
                # plt.show()


def generate_plots_loss_grad(visor_prompts):
    model = "sd2.1"
    for loss in ["sigmoid", "relu", "gelu"]:
        for prompt in visor_prompts:
            filename = os.path.join("logs", f"{model}_{loss}_spatial_loss_behaviour_{prompt}.log")
            plot_diffusion_statistics(filename, loss, prompt)


def merge_plots(visor_prompts):
    model = "sd2.1"

    combined_dir = os.path.join("logs/combine_losses_analysis")
    os.makedirs(combined_dir, exist_ok=True)

    for prompt in visor_prompts:
        images = []

        for loss in ["relu", "gelu", "sigmoid"]:
            filename = os.path.join("logs/logs_imgs", f"{model}_{loss}_spatial_loss_behaviour_{prompt}.png")

            if os.path.exists(filename):
                images.append(Image.open(filename))
            else:
                print(f"Warning: Image not found: {filename}")

        if len(images) == 3:
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)

            combined_img = Image.new('RGB', (total_width, max_height))

            x_offset = 0
            for img in images:
                combined_img.paste(img, (x_offset, 0))
                x_offset += img.width

            clean_prompt = ''.join(c if c.isalnum() or c.isspace() else '_' for c in prompt)

            combined_filename = os.path.join(combined_dir, f"{model}_combined_losses_{clean_prompt}.png")
            combined_img.save(combined_filename)
            print(f"Created combined image for prompt: {prompt}")

            for img in images:
                img.close()


def get_logs_statistics(loss_types):
    model = "sd2.1"

    summary_data = []

    pattern = re.compile(
        r"Timestep (\d+), spatial loss: ([\-\d\.e]+), grad min: ([\-\d\.e]+), grad mean: ([\-\d\.e]+), grad max: ([\-\d\.e]+)"
    )
    spike_threshold = 2.0

    for loss in loss_types:
        for file in os.listdir("logs"):
            if not file.endswith(".log") or f"{model}_{loss}_spatial_loss_behaviour_" not in file:
                continue

            full_prefix = f"{model}_{loss}_spatial_loss_behaviour_"
            prompt = file[len(full_prefix):-4]
            print(prompt)

            filepath = os.path.join("logs", file)
            with open(filepath, "r") as f:
                log_data = f.read()

            matches = pattern.findall(log_data)
            if not matches:
                continue
            df = pd.DataFrame(matches, columns=["timestep", "spatial_loss", "grad_min", "grad_mean", "grad_max"]).astype(float)

            final_loss = df["spatial_loss"].iloc[-1]
            initial_loss = df["spatial_loss"].iloc[0]
            loss_reduction = initial_loss - final_loss
            num_spikes_max = (df["grad_max"] > spike_threshold).sum()
            num_spikes_min = (df["grad_min"] < -spike_threshold).sum()
            grad_mean_std = df["grad_mean"].std()
            grad_max_std = df["grad_max"].std()
            grad_min_std = df["grad_min"].std()

            summary_data.append({
                "prompt": prompt,
                "loss_type": loss,
                "initial_loss": round(initial_loss, 6),  # Round to 6 decimal places
                "final_loss": round(final_loss, 6),
                "loss_reduction": round(loss_reduction, 6),
                "grad_mean_std": round(grad_mean_std, 6),
                "grad_min_std": round(grad_min_std, 6),
                "grad_max_std": round(grad_max_std, 6),
                "grad_max_spikes": num_spikes_max,
                "grad_min_spikes": num_spikes_min
            })

    summary_df = pd.DataFrame(summary_data)
    output_path = "spatial_loss_summary1.csv"
    summary_df.to_csv(output_path, index=False)


def print_summary_statistics():
    summary_df = pd.read_csv("logs/spatial_loss_averaged_over_layers_summary.csv")

    loss_stats = summary_df.groupby("loss_type").agg({
        "initial_loss": ["mean", "std"],
        "final_loss": ["mean", "std"],
        "loss_reduction": ["mean", "std"],
        "grad_mean_std": ["mean", "std"],
        "grad_min_std": ["mean", "std"],
        "grad_max_std": ["mean", "std"],
        "grad_max_spikes": ["mean", "std"],
        "grad_min_spikes": ["mean", "std"]
    }).round(4)

    ordered_loss_types = ["relu", "gelu", "sigmoid"]
    loss_stats = loss_stats.reindex(ordered_loss_types).reset_index()

    output_path = "logs/spatial_loss_averaged_over_layers_summary_statistics.csv"
    loss_stats.to_csv(output_path, index=False)


def get_logs_statistics_attn_layer(loss_types):
    model = "sd2.1"
    summary_data = []
    timestep_data = []

    block_pattern = re.compile(
        r"Timestep (\d+), spatial loss: ([-\d\.e]+), block: ([\w\.]+)"
    )
    grad_pattern = re.compile(
        r"Timestep (\d+), grad min: ([-\d\.e]+), grad mean: ([-\d\.e]+), grad max: ([-\d\.e]+)"
    )

    spike_threshold = 2.0

    for loss in loss_types:
        for file in os.listdir("logs/logs_attn"):
            if not file.endswith(".log") or f"{model}_spatial_loss_behaviour_attn_{loss}_" not in file:
                continue

            full_prefix = f"{model}_spatial_loss_behaviour_attn_{loss}_"
            prompt = file[len(full_prefix):-4]

            filepath = os.path.join("logs/logs_attn", file)
            with open(filepath, "r") as f:
                log_data = f.read()

            block_matches = block_pattern.findall(log_data)
            grad_matches = grad_pattern.findall(log_data)

            if not block_matches or not grad_matches:
                continue

            timestep_blocks_data = {}
            for timestep, spatial_loss, block in block_matches:
                timestep_int = int(timestep)
                if timestep_int not in timestep_blocks_data:
                    timestep_blocks_data[timestep_int] = {}

                if block not in timestep_blocks_data[timestep_int]:
                    timestep_blocks_data[timestep_int][block] = float(spatial_loss)

            blocks_data = {}
            for timestep, spatial_loss, block in block_matches:
                if block not in blocks_data:
                    blocks_data[block] = []
                blocks_data[block].append((int(timestep), float(spatial_loss)))

            grad_df = pd.DataFrame(grad_matches,
                                   columns=["timestep", "grad_min", "grad_mean", "grad_max"]).astype(float)

            timestep_gradients = {}
            for _, row in grad_df.iterrows():
                ts = int(row["timestep"])
                timestep_gradients[ts] = {
                    "grad_min": float(row["grad_min"]),
                    "grad_mean": float(row["grad_mean"]),
                    "grad_max": float(row["grad_max"])
                }

            for block, values in blocks_data.items():
                block_df = pd.DataFrame(values, columns=["timestep", "spatial_loss"])

                initial_loss = block_df["spatial_loss"].iloc[0]
                final_loss = block_df["spatial_loss"].iloc[-1]
                loss_reduction = initial_loss - final_loss
                loss_std = block_df["spatial_loss"].std()

                summary_data.append({
                    "prompt": prompt,
                    "loss_type": loss,
                    "block": block,
                    "initial_loss": round(initial_loss, 6),
                    "final_loss": round(final_loss, 6),
                    "loss_reduction": round(loss_reduction, 6),
                    "loss_std": round(loss_std, 6),
                })

            num_spikes_max = (grad_df["grad_max"] > spike_threshold).sum()
            num_spikes_min = (grad_df["grad_min"] < -spike_threshold).sum()
            grad_mean_std = grad_df["grad_mean"].std()
            grad_max_std = grad_df["grad_max"].std()
            grad_min_std = grad_df["grad_min"].std()

            summary_data.append({
                "prompt": prompt,
                "loss_type": loss,
                "block": "overall_gradients",
                "initial_loss": None,
                "final_loss": None,
                "loss_reduction": None,
                "loss_std": None,
                "grad_mean_std": round(grad_mean_std, 6),
                "grad_min_std": round(grad_min_std, 6),
                "grad_max_std": round(grad_max_std, 6),
                "grad_max_spikes": num_spikes_max,
                "grad_min_spikes": num_spikes_min
            })

            sorted_timesteps = sorted(timestep_blocks_data.keys())
            for ts in sorted_timesteps:
                blocks = timestep_blocks_data[ts]
                for block, spatial_loss in blocks.items():
                    timestep_data.append({
                        "prompt": prompt,
                        "loss_type": loss,
                        "block": block,
                        "timestep": ts,
                        "spatial_loss": spatial_loss
                    })

                if ts in timestep_gradients:
                    timestep_data.append({
                        "prompt": prompt,
                        "loss_type": loss,
                        "block": "overall_gradients",
                        "timestep": ts,
                        "grad_min": timestep_gradients[ts]["grad_min"],
                        "grad_mean": timestep_gradients[ts]["grad_mean"],
                        "grad_max": timestep_gradients[ts]["grad_max"]
                    })

    summary_df = pd.DataFrame(summary_data)
    output_path = f"logs/spatial_loss_attn_blocks_summary_{loss_types[0]}.csv"
    summary_df.to_csv(output_path, index=False)

    timestep_df = pd.DataFrame(timestep_data)
    timestep_output_path = f"logs/spatial_loss_attn_blocks_timesteps_{loss_types[0]}.csv"
    timestep_df.to_csv(timestep_output_path, index=False)
    print(f"Saved detailed timestep data to {timestep_output_path}")


def plot_attn_layer_losses(prompt=None, loss_type=None, top_n=None):
    timestep_df = pd.read_csv(f"logs/spatial_loss_attn_blocks_timesteps_{loss_type}.csv")

    layer_data = timestep_df[timestep_df["block"] != "overall_gradients"]

    if prompt is None:
        prompt = layer_data["prompt"].iloc[0]

    if loss_type is None:
        loss_type = layer_data["loss_type"].iloc[0]

    filtered_data = layer_data[(layer_data["prompt"] == prompt) &
                               (layer_data["loss_type"] == loss_type)]

    if filtered_data.empty:
        print(f"No data found for prompt '{prompt}' with loss type '{loss_type}'")
        return

    blocks = filtered_data["block"].unique()
    loss_stats = []

    for block in blocks:
        block_data = filtered_data[filtered_data["block"] == block].sort_values("timestep")
        if not block_data.empty:
            initial_loss = block_data["spatial_loss"].iloc[0]
            final_loss = block_data["spatial_loss"].iloc[-1]
            loss_reduction = initial_loss - final_loss
            percent_reduction = (loss_reduction / initial_loss) * 100 if initial_loss != 0 else 0

            loss_stats.append({
                "prompt": prompt,
                "loss": loss_type,
                "block": block,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "loss_reduction": loss_reduction,
                "percent_reduction": percent_reduction
            })

    loss_stats = sorted(loss_stats, key=lambda x: float(x["percent_reduction"]), reverse=True)

    if top_n and top_n < len(loss_stats):
        blocks_to_plot = [stat["block"] for stat in loss_stats[:top_n]]
    else:
        blocks_to_plot = [stat["block"] for stat in loss_stats]

    plt.figure(figsize=(15, 10))
    cmap = plt.cm.get_cmap('tab20', len(blocks_to_plot))

    for i, block in enumerate(blocks_to_plot):
        block_data = filtered_data[filtered_data["block"] == block].sort_values("timestep")

        stats = next(stat for stat in loss_stats if stat["block"] == block)

        plt.plot(
            block_data["timestep"],
            block_data["spatial_loss"],
            color=cmap(i),
            linewidth=2,
            label=f"{block}: Δ={stats['loss_reduction']:.4f} ({stats['percent_reduction']:.1f}%)"
        )

    plt.title(f"Spatial Loss by Attention Block\nPrompt: {prompt}, Loss Type: {loss_type}")
    plt.xlabel("Timestep")
    plt.ylabel("Spatial Loss")
    plt.grid(True)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()

    save_dir = "logs/loss_attn_layers_plots"
    os.makedirs(save_dir, exist_ok=True)
    clean_prompt = ''.join(c if c.isalnum() or c.isspace() else '_' for c in prompt)
    top_n_str = f"_top{top_n}" if top_n else ""
    filename = os.path.join(save_dir, f"{loss_type}{top_n_str}_{clean_prompt}.png")
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    # plt.show()

    data_dir = "logs/loss_attn_layers_data"
    os.makedirs(data_dir, exist_ok=True)
    csv_filename = os.path.join(data_dir, f"{loss_type}_{clean_prompt}.csv")

    stats_df = pd.DataFrame(loss_stats)
    stats_df.to_csv(csv_filename, index=False)
    print(f"Saved loss statistics to {csv_filename}")

    print("\nTop layers with highest loss reduction:")
    for i, stat in enumerate(loss_stats[:10 if len(loss_stats) > 10 else len(loss_stats)]):
        print(f"{i + 1}. {stat['block']}: Initial={stat['initial_loss']:.4f}, Final={stat['final_loss']:.4f}, "
              f"Reduction={stat['loss_reduction']:.4f} ({stat['percent_reduction']:.1f}%)")

    return loss_stats


def plot_averaged_attn_layer_losses(loss_type, top_n=9):
    timestep_df = pd.read_csv(f"logs/spatial_loss_attn_blocks_timesteps_{loss_type}.csv")

    # Filter data for the specified loss type and exclude overall_gradients
    layer_data = timestep_df[(timestep_df["block"] != "overall_gradients") &
                             (timestep_df["loss_type"] == loss_type)]

    if layer_data.empty:
        print(f"No data found for loss type '{loss_type}'")
        return

    # Group by block and timestep, then average across all prompts
    avg_data = layer_data.groupby(["block", "timestep"])["spatial_loss"].mean().reset_index()

    # Calculate statistics for each block
    blocks = avg_data["block"].unique()
    loss_stats = []

    for block in blocks:
        block_data = avg_data[avg_data["block"] == block].sort_values("timestep")
        if not block_data.empty:
            initial_loss = block_data["spatial_loss"].iloc[0]
            final_loss = block_data["spatial_loss"].iloc[-1]
            loss_reduction = initial_loss - final_loss
            percent_reduction = (loss_reduction / initial_loss) * 100 if initial_loss != 0 else 0

            loss_stats.append({
                "loss": loss_type,
                "block": block,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "loss_reduction": loss_reduction,
                "percent_reduction": percent_reduction
            })

    loss_stats = sorted(loss_stats, key=lambda x: float(x["percent_reduction"]), reverse=True)

    if top_n and top_n < len(loss_stats):
        blocks_to_plot = [stat["block"] for stat in loss_stats[:top_n]]
    else:
        blocks_to_plot = [stat["block"] for stat in loss_stats]

    plt.figure(figsize=(15, 10))
    cmap = plt.cm.get_cmap('tab20', len(blocks_to_plot))

    for i, block in enumerate(blocks_to_plot):
        block_data = avg_data[avg_data["block"] == block].sort_values("timestep")

        stats = next(stat for stat in loss_stats if stat["block"] == block)

        plt.plot(
            block_data["timestep"],
            block_data["spatial_loss"],
            color=cmap(i),
            linewidth=2,
            label=f"{block}: Δ={stats['loss_reduction']:.4f} ({stats['percent_reduction']:.1f}%)"
        )

    if loss_type == "sigmoid":
        loss_title = "Sigmoid"
    elif loss_type == "relu":
        loss_title = "ReLU"
    elif loss_type == "gelu":
        loss_title = "GELU"
    elif loss_type == "leaky_relu":
        loss_title = "Leaky ReLU"

    plt.title(f"Average Spatial Loss by Attention Block\nAcross 20 Prompts, Loss Type: {loss_title}")
    plt.xlabel("Timestep")
    plt.ylabel("Average Spatial Loss")
    plt.grid(True)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    # plt.show()

    save_dir = "logs"
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f"{loss_type}_averaged_across_prompts.png")
    plt.savefig(filename)
    print(f"Saved averaged plot to {filename}")

    # data_dir = f"logs/loss_attn_layers_data"
    # os.makedirs(data_dir, exist_ok=True)
    # csv_filename = os.path.join(data_dir, f"{loss_type}_averaged_stats.csv")
    #
    # stats_df = pd.DataFrame(loss_stats)
    # stats_df.to_csv(csv_filename, index=False)
    # print(f"Saved averaged loss statistics to {csv_filename}")
    #
    # print("\nTop layers with highest average loss reduction:")
    # for i, stat in enumerate(loss_stats[:10 if len(loss_stats) > 10 else len(loss_stats)]):
    #     print(f"{i + 1}. {stat['block']}: Initial={stat['initial_loss']:.4f}, Final={stat['final_loss']:.4f}, "
    #           f"Reduction={stat['loss_reduction']:.4f} ({stat['percent_reduction']:.1f}%)")

    # return loss_stats


def analyze_layer_effectiveness(consolidated_file):
    all_stats = pd.read_csv(consolidated_file)

    all_stats['rank'] = all_stats.groupby(['prompt', 'loss'])['percent_reduction'].rank(ascending=False)

    all_stats['normalized_reduction'] = all_stats.groupby(['prompt', 'loss'])['percent_reduction'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
    )

    for loss_type in all_stats['loss'].unique():
        print(f"\nAnalyzing loss type: {loss_type}")

        loss_stats = all_stats[all_stats['loss'] == loss_type]

        loss_stats['rank'] = loss_stats.groupby(['prompt'])['percent_reduction'].rank(ascending=False)

        loss_stats['normalized_reduction'] = loss_stats.groupby(['prompt'])['percent_reduction'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
        )

        layer_stats = loss_stats.groupby('block').agg({
            'initial_loss': ['mean'],
            'final_loss': ['mean'],
            'loss_reduction': ['mean'],
            'rank': ['mean', 'median', 'std', 'count'],
            'normalized_reduction': ['mean', 'median', 'std'],
            'percent_reduction': ['mean', 'median', 'std']
        }).reset_index()

        layer_stats.columns = [
            '_'.join(col).strip('_') for col in layer_stats.columns.values
        ]

        # Sort by normalized reduction (higher is better)
        layer_stats_by_rank = layer_stats.sort_values('percent_reduction_mean', ascending=False)

        analysis_file = os.path.join("logs", f"layer_effectiveness_{loss_type}_analysis_2.csv")
        layer_stats_by_rank.to_csv(analysis_file, index=False)
        print(f"Saved layer effectiveness analysis to {analysis_file}")

        # print(f"\nTop 10 most effective layers for {loss_type}:")
        # top_layers = layer_stats_by_rank # .head(10)
        # for _, row in top_layers.iterrows():
        #     print(f"Layer: {row['block']}, Avg Rank: {row['rank_mean']:.2f}, "
        #           f"Avg Norm Reduction: {row['normalized_reduction_mean']:.4f}, "
        #           f"Data Points: {row['rank_count']}")

    #     plt.figure(figsize=(12, 10))
    #     sns.scatterplot(
    #         x='normalized_reduction_mean',
    #         y='rank_mean',
    #         size='rank_count',
    #         hue='rank_std',
    #         data=layer_stats_by_rank, # .head(30),
    #         palette='viridis'
    #     )
    #
    #     for _, row in layer_stats_by_rank.iterrows(): # .head(15):
    #         plt.text(
    #             row['normalized_reduction_mean'], # + 0.01,
    #             row['rank_mean'],
    #             row['block'],
    #             fontsize=9
    #         )
    #
    #     plt.title(f'Effectiveness of Attention Layers - {loss_type.upper()} Loss')
    #     plt.xlabel('Mean Normalized Reduction') # (higher is better)
    #     plt.ylabel('Mean Rank') # (lower is better)
    #     plt.tight_layout()
    #     # plt.show()
    #
    #     viz_file = os.path.join(f"layer_effectiveness_{loss_type}_viz.png")
    #     plt.savefig(viz_file)
    #     print(f"Saved visualization to {viz_file}")
    #     plt.close()
    #
    # return {loss_type: layer_stats_by_rank for loss_type in all_stats['loss'].unique()}


def create_consolidated_file(loss_types, visor_prompts):
    filename = f"logs/combined_ablation_results_{loss_types[0]}.csv"
    if os.path.exists(filename):
        print(f"Consolidated file already exists: {filename}")
        return filename

    combined_df = pd.DataFrame()

    for prompt in visor_prompts:

        for loss in loss_types:
            file_name = f"{loss}_{prompt}.csv"
            file_path = os.path.join("logs/loss_attn_layers_data", file_name)

            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            else:
                print(f"Warning: File not found -> {file_path}")

    combined_df.to_csv(filename, index=False)
    return filename


def plot_alpha_graph():
    # Sigmoid data
    alpha_values = [1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    sigmoid_oa = [41.288, 43.182, 42.045, 43.561, 44.697, 47.159, 41.477, 46.780]
    sigmoid_uncond = [39.583, 42.803, 41.856, 43.561, 44.318, 47.159, 41.477, 46.402]
    sigmoid_cond = [95.872, 99.123, 99.550, 100.000, 99.153, 100.000, 100.000, 99.190]

    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(8, 5))
    plt.plot(alpha_values, sigmoid_oa, 'o-', label='OA', color='blue')
    plt.plot(alpha_values, sigmoid_uncond, 's-', label='VISOR_uncond', color='green')
    plt.plot(alpha_values, sigmoid_cond, 'd-', label='VISOR_cond', color='red')
    plt.xticks(alpha_values, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(0.9, 10.1)
    plt.ylim(0, 101)
    plt.title('Sigmoid: OA and VISOR Scores vs Alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Score (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join("logs", "sigmoid_alpha_plot.png"))
    # plt.show()


def plot_margin_graph(param):
    if param == "margin":
        param_values = [0.1, 0.25, 0.5]
        scores = {
            'ReLU': {
                'OA': [33.144, 44.129, 45.076],
                'VISOR_uncond': [31.629, 43.939, 44.886],
                'VISOR_cond': [95.429, 99.571, 99.580]
            },
            'GELU': {
                'OA': [44.697, 48.295, 42.424],
                'VISOR_uncond': [43.561, 48.295, 42.045],
                'VISOR_cond': [97.458, 100.000, 99.107]
            },
            'Leaky ReLU': {
                'OA': [46.023, 44.318, 42.235],
                'VISOR_uncond': [45.455, 44.318, 42.045],
                'VISOR_cond': [98.765, 100.000, 99.552]
            }
        }

    if param == "alpha":
        param_values = [0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0]
        scores = {
            'ReLU': {
                'OA': [31.818, 40.720, 43.750, 45.265, 43.371, 37.689, 35.985],
                'VISOR_uncond': [27.273, 39.394, 43.561, 44.318, 43.371, 36.364, 34.470],
                'VISOR_cond': [85.714, 96.744, 99.567, 97.908, 100.000, 96.482, 95.789]
            },
            'GELU': {
                'OA': [28.977, 33.902, 40.909, 45.455, 44.697, 46.780, 36.174],
                'VISOR_uncond': [21.591, 30.682, 39.962, 45.076, 44.318, 46.402, 35.038],
                'VISOR_cond': [74.510, 90.503, 97.685, 99.167, 99.153, 99.190, 96.859]
            },
            'Leaky ReLU': {
                'OA': [30.303, 38.068, 43.182, 42.176, 45.455, 49.432, 43.939],
                'VISOR_uncond': [26.326, 36.932, 42.424, 41.603, 44.697, 49.053, 43.75],
                'VISOR_cond': [86.875, 97.015, 98.246, 98.643, 98.333, 99.234, 99.569]
            }
        }

    # Colors for each loss type
    colors = {
        'ReLU': 'blue',
        'GELU': 'green',
        'Leaky ReLU': 'red'
    }

    # Create a plot for each metric
    metrics = ['OA', 'VISOR_uncond', 'VISOR_cond']
    markers = {'ReLU': 'o-', 'GELU': 's-', 'Leaky ReLU': 'd-'}

    for metric in metrics:
        if metric == "OA":
            plt.rcParams.update({'font.size': 14})
            plt.figure(figsize=(8,5))

            for loss_type, values in scores.items():
                plt.plot(
                    param_values,
                    values[metric],
                    markers[loss_type],
                    label=f'{loss_type}',
                    color=colors[loss_type]
                )

            if param == "margin":
                plt.xticks(param_values, fontsize=14)
                plt.xlim(0.095, 0.505)
                # plt.ylim(90, 100.1) # cond
            if param == "alpha":
                plt.xscale("log")
                plt.xticks(param_values, [str(v) for v in param_values], fontsize=14)
                plt.xlim(0.095, 5.1)
                # plt.ylim(70, 100.2) # cond

            plt.yticks(fontsize=14)
            plt.ylim(20, 55)
            plt.title(f"Object Accuracy ({metric}) Scores vs {param.capitalize()}")
            # plt.title(f'{metric.replace("VISOR_cond", r"VISOR$_{cond}$")} Scores vs {param.capitalize()}')
            # plt.title(f'{metric.replace("VISOR_uncond", r"VISOR$_{uncond}$")} Scores vs {param.capitalize()}')

            if param == "margin":
                plt.xlabel(f'{param.capitalize()}')
            if param == "alpha":
                plt.xlabel(f'{param.capitalize()} (log scale)')

            plt.ylabel(f'Object Accuracy ({metric}) (%)')
            # plt.ylabel(f'{metric.replace("VISOR_cond", r"VISOR$_{cond}$")} (%)')
            # plt.ylabel(f'{metric.replace("VISOR_uncond", r"VISOR$_{uncond}$")} (%)')

            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("logs", f"{metric.lower()}_{param}_comparison_plot.png"))
            # plt.show()


def read_losses(filename):
    steps = []
    losses = []
    step=0
    counter=0
    with open(filename, 'r') as f:
        for line in f:
            if line.strip()and counter%9==0:  # skip empty lines
                loss = float(line.strip())
                steps.append(int(step))
                step+=1
                losses.append(loss)
            counter+=1
    return steps, losses

def plot_losses(name):
    # Read data from both files
    steps1, losses1 = read_losses('obj1.txt')
    steps2, losses2 = read_losses('obj2.txt')
    steps3, losses_total = read_losses('objs.txt')
    steps4, spatial = read_losses('spatial.txt')

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(steps1, losses1, label='Object 1')
    plt.plot(steps2, losses2, label='Object 2')
    plt.plot(steps3, losses_total, label='Total Loss')
    plt.plot(steps4, spatial, label='Spatial')


    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Comparison_'+name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    # Save the figure
    plt.savefig('plots/loss_plot_'+name+".png")
    os.system('rm obj1.txt')
    os.system('rm obj2.txt')
    os.system('rm objs.txt')
    os.system('rm spatial.txt')

    plt.show()


if __name__ == "__main__":
    with open(os.path.join("../json_files", "visor_ablation_20.json"), "r") as f:
        visor_data = json.load(f)

    visor_prompts = []
    for data in visor_data:
        visor_prompts.append(data["text"])

    loss_types = ["leaky_relu"] # ["sigmoid", "relu", "gelu"]

    # get_logs_statistics_attn_layer(loss_types)

    # for loss in loss_types:
    #     plot_averaged_attn_layer_losses(loss_type=loss, top_n=9)
        # for prompt in visor_prompts:
        #     plot_attn_layer_losses(prompt=prompt, loss_type=loss, top_n=9)

    # generate_plots_loss_grad(visor_prompts)
    # merge_plots(visor_prompts)
    # get_logs_statistics(loss_types)
    # print_summary_statistics()

    margins = [0.25]
    alphas = [1.0] #, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    slopes = [0.1]
    # plot_sigmoid_alpha(margins, alphas)
    # plot_sigmoid_alpha_gradients(margins, alphas)
    # plot_spatial_losses_behaviour(margins, alphas, slopes)
    gradient_spatial_loss_behaviour(margins, alphas, slopes)

    # consolidated_filename = create_consolidated_file(loss_types, visor_prompts)
    # analyze_layer_effectiveness(consolidated_filename)

    # plot_alpha_graph()
    # param = "margin"
    # param = "alpha"
    # plot_margin_graph(param)
