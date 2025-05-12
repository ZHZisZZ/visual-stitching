#!/bin/bash
# e.g., bash scripts/plot.sh model

# Function to recursively find directories containing seed-* subdirectories
find_seed_dirs() {
    local dir="$1"
    
    # Check if current directory has any seed-* subdirectories
    if ls -d "$dir"/seed-* 1> /dev/null 2>&1; then
        # echo "$dir"
        python src/tools/plot_rank.py --output_base_dir "${dir}" &
        python src/tools/plot_prob.py --output_base_dir "${dir}" &
        sleep 1
    fi
    
    # Recursively process all subdirectories
    for subdir in "$dir"/*; do
        if [ -d "$subdir" ]; then
            find_seed_dirs "$subdir"
        fi
    done
}

# Start from the current directory or a specified directory
start_dir="${1:-.}"

# Find and print all directories containing seed-* subdirectories
find_seed_dirs "$start_dir"
