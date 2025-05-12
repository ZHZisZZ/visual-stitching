#!/bin/bash
export PYTHONPATH=.

# Base directories
src_base_dir="tmp/data/moderation/files"
tgt_base_dir="tmp/data/moderation/files/others/filter"
filter_class="OpenAI_Moderation_Filter"

for split_factor in 2 4 8; do
    # Find all subdirectories inside the factor folder
    factor="${split_factor}x${split_factor}"
    for src_subdir in "$src_base_dir/$factor"/*; do
        if [ -d "$src_subdir" ]; then
            subdir_name=$(basename "$src_subdir")
            tgt_subdir="$tgt_base_dir/$filter_class/$factor/safe/$subdir_name"

            echo "Processing $src_subdir -> $tgt_subdir"

            python src/tools/patches_filter.py \
            --src_patches_dir "$src_subdir" \
            --tgt_patches_dir "$tgt_subdir" \
            --filter_class_name "$filter_class"

            python src/tools/patches_stitch.py \
            --src_patches_dir "$tgt_base_dir/$filter_class/$factor/safe/$subdir_name" \
            --tgt_images_dir "$tgt_base_dir/$filter_class/$factor/safe_stitched" \
            --split_factor ${split_factor} &
        fi
    done
done
