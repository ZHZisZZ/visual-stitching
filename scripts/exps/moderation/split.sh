#!/bin/bash
# Set environment variables
export PYTHONPATH=.

# Input directory
input_dir="data/moderation/files/images"

# Output base directory
output_base_dir="tmp/data/moderation/files"

# Iterate over all images
for img_path in "$input_dir"/*.jpg; do
    img_filename=$(basename "$img_path")
    img_name="${img_filename%.*}"  # Remove .jpg suffix

    for split_factor in 2 4 8; do
        python src/tools/patches_split.py \
        --src_image_path "$img_path" \
        --tgt_patches_dir "$output_base_dir/${split_factor}x${split_factor}/$img_name" \
        --split_factor ${split_factor} &
        sleep .5
    done
done
