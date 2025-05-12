import os
import re
import random
import inspect
import hashlib
import pathlib
from typing import Callable

from PIL import Image
import torch
import omegaconf
import datasets
# import accelerate
import transformers


###################################
# Model- & Processor-specific utils
###################################
def get_processor_kwargs(model: transformers.PreTrainedModel) -> dict:
    processor_kwargs = {"padding_side": "right"}
    if isinstance(model, (transformers.Qwen2VLForConditionalGeneration, transformers.Qwen2_5_VLForConditionalGeneration)):
        processor_kwargs["min_pixels"] = 32 * 28 * 28
        processor_kwargs["max_pixels"] = 128 * 28 * 28
    if isinstance(model, transformers.LlavaForConditionalGeneration):
        processor_kwargs["add_prefix_space"] = True
    return processor_kwargs


def get_image_prefix(processor: transformers.ProcessorMixin) -> str:
    if isinstance(processor, transformers.MllamaProcessor):
        image_prefix = "<|image|><|begin_of_text|>"
    elif isinstance(processor, transformers.Gemma3Processor):
        image_prefix = "<start_of_image> "
    elif isinstance(processor, (transformers.LlavaProcessor, transformers.LlavaNextProcessor)):
        image_prefix = "USER: <image>\n ASSISTANT:"
    elif isinstance(processor, (transformers.Qwen2VLProcessor, transformers.Qwen2_5_VLProcessor)):
        image_prefix = "<|vision_start|><|image_pad|><|vision_end|>"
    else:
        raise NotImplementedError
    return image_prefix


def get_image_preprocess_fn(processor: transformers.ProcessorMixin) -> Callable:
    fn_do_nothing = lambda image: image
    if isinstance(processor, transformers.MllamaProcessor):
        fn = fn_do_nothing
    elif isinstance(processor, transformers.Gemma3Processor):
        fn = fn_do_nothing
    elif isinstance(processor, (transformers.LlavaProcessor, transformers.LlavaNextProcessor)):
        fn = fn_do_nothing
    elif isinstance(processor, (transformers.Qwen2VLProcessor, transformers.Qwen2_5_VLProcessor)):
        import qwen_vl_utils
        fn = lambda image: qwen_vl_utils.fetch_image({"image": image})
    else:
        raise NotImplementedError
    return fn


##################
# Helper functions
##################
class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def cache(fn, cache_dir: str = "./.cache"):
    os.makedirs(cache_dir, exist_ok=True)
    def new_fn(*args, **kwargs):
        # Get the function's signature and default arguments
        sig = inspect.signature(fn)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()  # Fill in default arguments

        # Generate a unique identifier based on the filtered arguments
        arg_str = str(bound_args.arguments)
        identifier = hashlib.md5(arg_str.encode()).hexdigest()
        cache_path = os.path.join(cache_dir, identifier)

        # Check if the cached dataset exists
        if os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            return datasets.load_from_disk(cache_path)
        else:
            # Generate the dataset
            dataset = fn(*args, **kwargs)
            # Save the dataset to the cache
            print(f"Saving dataset to cache at {cache_path}")
            dataset.save_to_disk(cache_path)
            return dataset
    return new_fn


def split_into_patches(
    image: Image, 
    split_factor: int = 3
) -> list[dict]:
    """
    Splits a PIL Image into `split_factor`*`split_factor` patches.

    Args:
        image (PIL.Image): The input image.
        `split_factor` (int): The number of patches along each dimension (`split_factor`*`split_factor` patches in total).

    Returns:
        list: A list of PIL.Image objects representing the patches.
    """
    # Get the dimensions of the image
    width, height = image.size

    # Calculate the size of each patch
    patch_width = width // split_factor
    patch_height = height // split_factor

    # Initialize a list to store the patches
    patches = []

    # Loop through the image and extract patches
    for i in range(split_factor):
        for j in range(split_factor):
            # Define the bounding box for the current patch
            left = j * patch_width
            upper = i * patch_height
            right = left + patch_width
            lower = upper + patch_height

            # Crop the patch from the image
            patch = image.crop((left, upper, right, lower))
            patches.append({"image": patch, "row": i, "col": j})

    return patches


def overwrite_config_from_args(config: dict, data_overwrite_args: str):
    """
    Overwrites the values in a nested config dictionary using dot-list-like syntax.
    Only allows overwriting existing keys; adding new keys is not allowed.

    Args:
        config: The configuration dictionary to modify.
        data_overwrite_args: A comma-separated string, where each entry is in the format:
                             "key1.key2[0].key3=value"
    """
    assignments = [arg.strip() for arg in data_overwrite_args.split(",") if arg.strip()]
    for assignment in assignments:
        key_path, value = assignment.split("=", 1)
        keys = re.split(r'\.(?![^\[]*\])', key_path)  # split by "." but not inside []
        curr = config
        for i, k in enumerate(keys):
            # Handle list index, e.g., key[0]
            match = re.match(r"([^\[\]]+)(\[(\d+)\])?", k)
            if not match:
                raise ValueError(f"Invalid key format: {k}")
            key, _, idx = match.groups()
            if idx is not None:
                idx = int(idx)
                if key not in curr or not isinstance(curr[key], list):
                    raise KeyError(f"{key} is not a list in config")
                if idx >= len(curr[key]):
                    raise IndexError(f"Index {idx} out of bounds for {key}")
                if i == len(keys) - 1:
                    curr[key][idx] = value
                else:
                    curr = curr[key][idx]
            else:
                if key not in curr:
                    raise KeyError(f"Key {key} does not exist in config")  # <-- ADDED
                if i == len(keys) - 1:
                    curr[key] = value
                else:
                    curr = curr[key]  # <-- CHANGED: removed `setdefault`, just access
    return config


def parse_mapping(mapping: list[str]) -> list[dict]:
    # Split each row into fields
    rows = [list(map(str.strip, row.split(","))) for row in mapping]
    # Extract header and rows
    keys = rows[0]
    dict_list = [dict(zip(keys, values)) for values in rows[1:]]
    return dict_list


def parse_data_config(data_config_path: str, data_overwrite_args: str = "") -> dict:
    # e.g. "data.train[0].images_dirs[0]=/new/path/to/images"
    data_config = omegaconf.OmegaConf.load(data_config_path)
    data_config = omegaconf.OmegaConf.to_container(data_config, resolve=True)
    if data_overwrite_args: data_config = overwrite_config_from_args(data_config, data_overwrite_args)
    data_config["mapping"] = parse_mapping(data_config["mapping"])
    return data_config


def process_mappings_with_images_into_patches(mappings: dict, split_factor) -> dict:
    new_mappings = []
    for mapping in mappings:
        patches = split_into_patches(mapping["image"], split_factor)
        for patch in patches:
            # TODO: should change path to name-{row}_{col}
            #       we also need to flag that the path is fictional
            new_mappings.append({**mapping, **patch})
    return new_mappings


def load_mappings_with_patches_from_dirs(mappings: dict, patches_dirs: list[str]) -> dict:
    name_2_mapping = {mapping["name"]: mapping for mapping in mappings}
    new_mappings = []
    for patches_dir in patches_dirs:
        # Recursively get all image files in the directory and its subdirectories
        patch_files = sorted([
            f for f in pathlib.Path(patches_dir).rglob('*')
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        assert patch_files, f"there is no file under {patches_dir}"
        for patch_file in patch_files:
            patch = Image.open(patch_file)
            pattern = re.compile(r"(?P<name>.+)-(?P<row>\d+)_(?P<col>\d+)(?:-.+)?$")  # Removed \.jpg
            match = pattern.match(patch_file.stem)  # Changed from patch_file.name to patch_file.stem
            if not match:
                print(f"Warning: filename format not matched for {patch_file}")
                continue
            name, row, col = match.group("name"), match.group("row"), match.group("col")
            if name not in name_2_mapping:
                # if accelerate.PartialState.is_main_process: print(f"Warning: '{name}' not found in mappings, skipping {patch_file}")
                continue
            mapping = name_2_mapping[name]
            new_mappings.append({**mapping, **{"image": patch, "row": int(row), "col": int(col)}, "path": str(patch_file)})
    return new_mappings


def load_mappings_with_images_from_dirs(mappings: dict, images_dirs: list[str]) -> dict:
    name_2_mapping = {mapping["name"]: mapping for mapping in mappings}
    new_mappings = []
    # Recursively get all image files in the directory and its subdirectories
    for images_dir in images_dirs:
        image_files = sorted([
            f for f in pathlib.Path(images_dir).rglob('*')
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        assert image_files, f"there is no file under {images_dir}"
        for image_file in image_files:
            image = Image.open(image_file)
            name = image_file.stem.split("-")[0] # in case we use patches as images
            if name not in name_2_mapping:
                # if accelerate.PartialState.is_main_process: print(f"Warning: '{name}' not found in mappings, skipping {image_file}")
                continue
            mapping = name_2_mapping[name]
            new_mappings.append({**mapping, **{"image": image, "path": str(image_file)}})
    return new_mappings


def load_mappings_with_images_from_paths(mappings: dict, image_paths: list[str]) -> dict:
    name_2_mapping = {mapping["name"]: mapping for mapping in mappings}
    new_mappings = []
    for image_path in image_paths:
        image_file = pathlib.Path(image_path)
        assert image_file.is_file(), f"{image_file} does not exist"
        if image_file.is_file() and image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            image = Image.open(image_file)
            name = image_file.stem.split("-")[0]  # same logic for handling names
            mapping = name_2_mapping[name]
            if name not in name_2_mapping:
                # if accelerate.PartialState.is_main_process: print(f"Warning: '{name}' not found in mappings, skipping {image_file}")
                continue
            new_mappings.append({**mapping, **{"image": image, "path": str(image_file)}})
    return new_mappings


def parse_train_and_eval_config(data_config: dict) -> tuple[list[dict], list[dict]]:
    train_config_list = []
    eval_config_list = []

    has_train = "train" in data_config["data"]
    has_eval = "eval" in data_config["data"]

    if has_train:
        for train_data in data_config["data"]["train"]:
            train_config = {
                "mapping": None,
                "templates": train_data["templates"]  # now directly use templates inside each entry
            }

            if "images_dirs" in train_data:
                mappings = load_mappings_with_images_from_dirs(data_config["mapping"], train_data["images_dirs"])
                if "split_factor" in train_data:
                    mappings = process_mappings_with_images_into_patches(mappings, train_data["split_factor"])
                    dropout = train_data.get("patch_dropout_ratio", 0.0)
                    mappings = [m for m in mappings if random.random() >= dropout]
                train_config["mapping"] = mappings
            elif "patches_dirs" in train_data:
                train_config["mapping"] = load_mappings_with_patches_from_dirs(data_config["mapping"], train_data["patches_dirs"])
            elif "image_paths" in train_data:
                train_config["mapping"] = load_mappings_with_images_from_paths(data_config["mapping"], train_data["image_paths"])
            else:
                # if accelerate.PartialState.is_main_process: print(f"Warning: no file paths / directories founds in train {str(train_data)}")
                train_config["mapping"] = data_config["mapping"]

            train_config_list.append(train_config)

    if has_eval:
        for eval_data in data_config["data"]["eval"]:
            eval_config = {
                "mapping": None,
                "templates": eval_data["templates"]  # now directly use templates inside each entry
            }

            if "images_dirs" in eval_data:
                eval_config["mapping"] = load_mappings_with_images_from_dirs(data_config["mapping"], eval_data["images_dirs"])
            elif "patches_dirs" in eval_data:
                eval_config["mapping"] = load_mappings_with_patches_from_dirs(data_config["mapping"], eval_data["patches_dirs"])
            elif "image_paths" in eval_data:
                eval_config["mapping"] = load_mappings_with_images_from_paths(data_config["mapping"], eval_data["image_paths"])
            else:
                # if accelerate.PartialState.is_main_process: print(f"Warning: no file paths / directories founds in eval {str(eval_data)}")
                eval_config["mapping"] = data_config["mapping"]

            eval_config_list.append(eval_config)

    return train_config_list, eval_config_list


def get_train_dataset(train_configs: list[dict]) -> datasets.Dataset:
    all_data = []

    for train_config in train_configs:
        for train_template in train_config["templates"]:
            prompt_template = train_template[0]
            prompt_response_template = train_template[0] + train_template[1]
            for row in train_config["mapping"]:
                prompt = prompt_template.format_map(SafeDict(row))
                prompt_response = prompt_response_template.format_map(SafeDict(row))
                data_point = {
                    **({"image": row["image"]} if "image" in row else {}),
                    "prompt": prompt,
                    "prompt_response": prompt_response,
                }
                all_data.append(data_point)

    return datasets.Dataset.from_list(all_data)


class GpuTimer:
    def __init__(self, name="GPU block"):
        self.name = name
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        torch.cuda.synchronize()  # ensure no prior GPU ops interfere
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_event.record()
        torch.cuda.synchronize()  # wait for GPU ops to finish
        elapsed_time = self.start_event.elapsed_time(self.end_event)  # in ms
        print(f"[{self.name}] GPU time: {elapsed_time:.3f} ms")
