"""
Split images into patches.

PYTHONPATH=. python src/tools/patches_split.py --src_images_dir "data/animal/files" --tgt_patches_dir "tmp/data/animal/files/4x4" --split_factor 4
"""
import pathlib
from dataclasses import dataclass

import tqdm
import tyro
from PIL import Image

from src import utils


@dataclass
class ScriptArguments:
    src_images_dir:  str = ""
    src_image_path: str = ""
    tgt_patches_dir: str = "tmp/data/animal/files/4x4"
    split_factor: int = 4

# Parse CLI arguments
script_args = tyro.cli(ScriptArguments)

# Ensure output directory exists
pathlib.Path(script_args.tgt_patches_dir).mkdir(parents=True, exist_ok=True)

# Handle multiple images
assert not (script_args.src_images_dir and script_args.src_image_path)
if script_args.src_images_dir:
    src_files = [
        f for f in pathlib.Path(script_args.src_images_dir).iterdir()
        if f.is_file()
    ]
elif script_args.src_image_path:
    src_files = [pathlib.Path(script_args.src_image_path)]


# Process each image
for src_file in tqdm.tqdm(src_files, desc="Splitting images"):
    image = Image.open(src_file)
    patches = utils.split_into_patches(image, split_factor=script_args.split_factor)

    # Split filename and extension
    src_file_prefix, src_file_suffix = src_file.name.rsplit(".", maxsplit=1)

    for patch in patches:
        patch_filename = f"{src_file_prefix}-{patch['row']}_{patch['col']}.{src_file_suffix}"
        patch_path = pathlib.Path(script_args.tgt_patches_dir) / patch_filename
        patch["image"].save(patch_path)
