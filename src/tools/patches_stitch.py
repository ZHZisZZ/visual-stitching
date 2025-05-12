"""
Stitch patches into images; skip missing ones using black color; optionally reorder patches

PYTHONPATH=. python src/tools/patches_stitch.py --src_patches_dir "tmp/data/animal/files/4x4" --tgt_images_dir "tmp/dev" --split_factor 4
"""
import pathlib
import re
import random
from dataclasses import dataclass
from collections import defaultdict

import tqdm
import tyro
from PIL import Image


@dataclass
class ScriptArguments:
    src_patches_dir: str = "tmp/data/moderation/files/others/filter/4x4/OpenAI_Moderation_Filter/safe/sex00"
    tgt_images_dir:  str = "tmp/data/moderation/files/others/filter/4x4/OpenAI_Moderation_Filter/safe_stitched"
    split_factor: int = 4
    reorder: bool = False
    seed: int = 42


script_args = tyro.cli(ScriptArguments)
src_dir = pathlib.Path(script_args.src_patches_dir)
tgt_dir = pathlib.Path(script_args.tgt_images_dir)
tgt_dir.mkdir(parents=True, exist_ok=True)

# Group patches by image prefix (e.g., "cat")
pattern = re.compile(r"(?P<name>.+)-(?P<row>\d+)_(?P<col>\d+)(?:-.+)?\.jpg")
patch_map = defaultdict(dict)

for file in src_dir.iterdir():
    if not file.suffix.lower().endswith("jpg"):
        continue
    match = pattern.match(file.name)
    if match:
        name = match.group("name")
        row = int(match.group("row"))
        col = int(match.group("col"))
        patch_map[name][(row, col)] = file

# Optional reordering setup
if script_args.reorder:
    random.seed(script_args.seed)

for name, patches in tqdm.tqdm(patch_map.items(), desc="Stitching patches"):
    if not patches:
        continue

    sample_patch = Image.open(next(iter(patches.values())))
    patch_width, patch_height = sample_patch.size

    full_image = Image.new("RGB", (patch_width * script_args.split_factor,
                                   patch_height * script_args.split_factor), color=(0, 0, 0))

    positions = [(row, col) for row in range(script_args.split_factor)
                            for col in range(script_args.split_factor)]

    if script_args.reorder:
        reordered_positions = positions.copy()
        random.shuffle(reordered_positions)
    else:
        reordered_positions = positions

    for original_pos, new_pos in zip(positions, reordered_positions):
        patch = (Image.open(patches[original_pos])
                 if original_pos in patches
                 else Image.new("RGB", (patch_width, patch_height), color=(0, 0, 0)))
        row, col = new_pos  # 正确顺序
        full_image.paste(patch, (col * patch_width, row * patch_height))

    output_path = tgt_dir / f"{name}.jpg"
    full_image.save(output_path)
