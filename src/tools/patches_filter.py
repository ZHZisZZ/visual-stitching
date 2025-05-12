"""
Filter patches using `filter_class_name`.

TODO: use cases
"""
import pathlib
import shlex
import math
from dataclasses import dataclass

import tqdm
from PIL import Image
import trl

from src import utils
from src.tools.filters.registry import create_filter


@dataclass
class ScriptArguments:
    src_patches_dir: str = "tmp/data/animal/files/4x4"
    tgt_patches_dir: str = "" # if none, just print, do not print
    batch_size: int = 8
    rank: int = 1
    world_size: int = 1
    filter_class_name: str = ""
    filter_kwargs: str = ""  # extra kwargs passed as a string, e.g., "--threshold 0.55"
    data_config_path: str = ""

# Parse arguments
parser = trl.TrlParser(ScriptArguments)
script_args = parser.parse_args_and_config()[0]

# Parse optional data config file
data_config = (
    utils.parse_data_config(script_args.data_config_path) 
    if script_args.data_config_path else None
)
name_2_mapping = (
    {mapping["name"]: mapping for mapping in data_config["mapping"]}
    if data_config else None
)

# Ensure output directory exists
if script_args.tgt_patches_dir:
    pathlib.Path(script_args.tgt_patches_dir).mkdir(parents=True, exist_ok=True)

# List all image files in the source directory
src_files = sorted(
    [f for f in pathlib.Path(script_args.src_patches_dir).iterdir() if f.is_file()]
)
# Partition `src_files` according to rank and world size
files_per_rank = math.ceil(len(src_files) / script_args.world_size)
src_files = src_files[(script_args.rank-1)*files_per_rank : (script_args.rank)*files_per_rank]

# Parse other_kwargs string to a dictionary
def parse_kwargs(kwargs_str: str) -> dict:
    tokens = shlex.split(kwargs_str)
    kwargs = {}
    for i in range(0, len(tokens), 2):
        key = tokens[i].lstrip("-")
        value = tokens[i + 1]
        # Try to interpret value as int or float if possible
        if value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
        kwargs[key] = value
    return kwargs
filter_kwargs_dict = parse_kwargs(script_args.filter_kwargs)
filter = create_filter(script_args.filter_class_name, **filter_kwargs_dict)

retained_src_files = []
unretained_src_files = []
for start_idx in tqdm.tqdm(range(0, len(src_files), script_args.batch_size)):
    batch_src_files = src_files[start_idx:start_idx+script_args.batch_size]
    batch_patches = [Image.open(src_file) for src_file in batch_src_files]

    if name_2_mapping:
        batch_mappings = [
            name_2_mapping[src_file.stem.split("-")[0]] 
            for src_file in batch_src_files
        ]
        inputs = [
            {"image": patch, "mapping": mapping}
            for patch, mapping in zip(batch_patches, batch_mappings)
        ]
    else:
        inputs = [
            {"image": patch} 
            for patch in batch_patches
        ]

    results = filter.apply(inputs)

    for src_file, patch, result in zip(batch_src_files, batch_patches, results):
        if not result["retain"]:
            unretained_src_files.append(src_file.name)
            if not script_args.tgt_patches_dir: print(f"{src_file.name} not retained")
        else:
            retained_src_files.append(src_file.name)
            if script_args.tgt_patches_dir:
                patch.save(pathlib.Path(script_args.tgt_patches_dir) / src_file.name)
            else:
                print(f"{src_file.name} retained")

if not script_args.tgt_patches_dir:
    print(f"retained files: {retained_src_files}")
    print(f"unretained files: {unretained_src_files}")
    print(f"unretained percentages: {len(unretained_src_files)}/{len(src_files)}")
