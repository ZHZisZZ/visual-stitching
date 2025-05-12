import os
import glob
import json
from dataclasses import dataclass
from collections import defaultdict

import matplotlib.pyplot as plt
import tyro
import itertools
import numpy as np


@dataclass
class ScriptArguments:
    output_base_dir: str = "models/animal/config_image/Qwen2-VL-7B/epochs10-lr1e-5"

script_args = tyro.cli(ScriptArguments)
base_dir = script_args.output_base_dir

# Data structure: {rank_type: {seed: [ranks per checkpoint]}}
rank_data = defaultdict(lambda: defaultdict(list))
loss_data = defaultdict(list) # maybe empty

# Find all seed directories
seed_dirs = glob.glob(os.path.join(base_dir, "seed-*"))
if not seed_dirs:
    raise ValueError(f"No seed directories found in {base_dir}")

seed_numbers = sorted([int(os.path.basename(d).split("-")[1]) for d in seed_dirs])

# Get num_train_epochs from the first seed directory's training_args.json
num_train_epochs = None
first_seed_dir = os.path.join(base_dir, f"seed-{seed_numbers[0]}")
training_args_path = os.path.join(first_seed_dir, "training_args.json")
if os.path.exists(training_args_path):
    with open(training_args_path, 'r') as f:
        training_args = json.load(f)
        num_train_epochs = training_args.get("num_train_epochs")

# Collect all unique rank types (e.g., train-0, eval-1)
rank_types = set()

# First pass to determine all rank types
for seed in seed_numbers:
    seed_dir = os.path.join(base_dir, f"seed-{seed}")
    checkpoints = sorted(
        [d for d in os.listdir(seed_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )

    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(seed_dir, checkpoint, "eval", "log.json")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                if "rank" in data:
                    rank_types.update(data["rank"].keys())

# Exit early if no rank data exists
if not rank_types:
    print("No rank data found. Skipping plot.")
    exit(0)

# Now collect the data
for seed in seed_numbers:
    seed_dir = os.path.join(base_dir, f"seed-{seed}")
    checkpoints = sorted(
        [d for d in os.listdir(seed_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )

    for checkpoint in checkpoints:
        checkpoint_path = os.path.join(seed_dir, checkpoint, "eval", "log.json")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
                loss_data[seed].append(data["log_history"]["eval_loss"])
                for rank_type in rank_types:
                    if rank_type in data["rank"]:
                        rank_data[rank_type][seed].append(data["rank"][rank_type]["rank"])

# Determine dynamic y_limit based on the number of ranked_options
example_checkpoint = os.path.join(seed_dirs[0], "checkpoint-0", "eval", "log.json")
if os.path.exists(example_checkpoint):
    with open(example_checkpoint, 'r') as f:
        data = json.load(f)
        # Find any meta entry and use the length of ranked_options
        for rank_type_data in data.get("rank", {}).values():
            if rank_type_data["meta"]:
                y_limit = len(rank_type_data["meta"][0]["ranked_options"]) / 2 + 1
                break

# Plotting
fig, ax = plt.subplots(figsize=(6, 5))

# Primary axis: rank_data
colors = itertools.cycle(plt.cm.tab10.colors)
color_map = {}
statistics = defaultdict(dict)

# Sort prob_types so train-* comes before eval-*
sorted_rank_types = sorted(rank_types, key=lambda x: (0 if x.startswith("train") else 1, x))
for rank_type in sorted_rank_types:
    color = next(colors)
    color_map[rank_type] = color

    all_epoch_data = []
    for seed, ranks in rank_data[rank_type].items():
        linestyle = 'solid' if rank_type.startswith("train") else 'dashed'
        ax.plot(np.linspace(0, num_train_epochs, len(ranks)), ranks,
                linewidth=2, linestyle=linestyle, color=color,
                label=f"{rank_type}" if seed == seed_numbers[0] else None)
        all_epoch_data.append(ranks)

    # Average per epoch across seeds
    if all_epoch_data:
        avg_per_epoch = np.mean(np.array(list(itertools.zip_longest(*all_epoch_data, fillvalue=np.nan))), axis=1)
        std_per_epoch = np.nanstd(np.array(list(itertools.zip_longest(*all_epoch_data, fillvalue=np.nan))), axis=1)
        statistics[rank_type]["mean"] = [round(float(p), 6) if not np.isnan(p) else None for p in avg_per_epoch]
        statistics[rank_type]["std"] = [round(float(p), 6) if not np.isnan(p) else None for p in std_per_epoch]
        if rank_data[rank_type]:
            num_ranked_options = len(data["rank"][rank_type]["meta"][0]["ranked_options"]) if "meta" in data["rank"][rank_type] else 0
            statistics[rank_type]["num_ranked_options"] = num_ranked_options

# Style for rank axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.tick_params(axis='both', which='both', direction='in', length=5, width=2)
ax.set_xlabel("epoch", fontsize=12, labelpad=5)
ax.set_ylabel("rank", fontsize=12, labelpad=5)

x_max = max(len(ranks) for rtype in rank_data.values() for ranks in rtype.values()) if rank_data else 0
ax.set_xlim(0, num_train_epochs)
ax.set_ylim(-0.2, y_limit)

# Secondary axis: loss_data
ax2 = ax.twinx()
for seed, losses in loss_data.items():
    ax2.plot(np.linspace(0, num_train_epochs, len(losses)), losses, color='gray', linewidth=2,
             linestyle='solid', label=f"loss" if seed == seed_numbers[0] else None)

# Style for loss axis
ax2.set_ylabel("loss", fontsize=12, labelpad=5)
ax2.spines['right'].set_linewidth(3)
ax2.tick_params(axis='y', which='both', direction='in', length=5, width=2)
ax2.set_ylim(-1, 10)

# Add legend (combine both axes)
handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(handles + handles2, labels + labels2, fontsize=12, loc='upper right')

# Title and save
ax.set_title(base_dir, fontsize=10)
plt.tight_layout()
plt.savefig(f"{base_dir}/rank.jpg", dpi=300, bbox_inches='tight')

# Save averaged rank as JSON
with open(os.path.join(base_dir, "rank.json"), "w") as f:
    json.dump(statistics, f, indent=2)
