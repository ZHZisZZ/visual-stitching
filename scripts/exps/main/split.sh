#!/bin/bash
export PYTHONPATH=.

datasets_to_run=(
    "animal"
    "food"
    "landmark"
)

for dataset in "${datasets_to_run[@]}"; do
    for split_factor in 2 4 8; do
        python src/tools/patches_split.py \
        --src_images_dir "data/${dataset}/files" \
        --tgt_patches_dir "tmp/data/${dataset}/files/${split_factor}x${split_factor}" \
        --split_factor "${split_factor}" &
        sleep .5
    done
done
