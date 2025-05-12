#!/bin/bash
# e.g., bash scripts/exps/main/train.sh --models Qwen2-VL-7B --datasets animal
source scripts/exps/model_configs.sh
source scripts/exps/utils.sh

# number of seeds
num_seeds=3

datasets_to_run=(
    "animal"
    "food"
    "landmark"
)
models_to_run=(
    # "Qwen2-VL-2B"
    # "Qwen2-VL-2B-Instruct"
    "Qwen2-VL-7B"
    # "Qwen2-VL-7B-Instruct"
    # "Qwen2-VL-72B"
    # "Qwen2-VL-72B-Instruct"
    # "Qwen2.5-VL-3B-Instruct"
    # "Qwen2.5-VL-7B-Instruct"
    # "Qwen2.5-VL-32B-Instruct"
    # "Qwen2.5-VL-72B-Instruct"
    # "gemma-3-4b-pt"
    # "gemma-3-4b-it"
    "gemma-3-12b-pt"
    # "gemma-3-12b-it"
    # "gemma-3-27b-pt"
    # "gemma-3-27b-it"
    "Llama-3.2-11B-Vision"
    # "Llama-3.2-11B-Vision-Instruct"
    # "Llama-3.2-90B-Vision"
    # "Llama-3.2-90B-Vision-Instruct"
    # "llava-1.5-7b-hf"
    # "llava-1.5-13b-hf"
    # "llava-v1.6-vicuna-7b-hf"
    # "llava-v1.6-vicuna-13b-hf"
)

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --num_seeds) num_seeds="$2"; shift ;;
        --models)
            shift
            _models_to_run=()
            IFS=',' read -ra _models_to_run <<< "$1"
            # Validate models
            for model in "${_models_to_run[@]}"; do
                if [[ -z "${model_configs[$model]}" ]]; then
                    echo "Error: Unknown model '$model'. Valid options are:"
                    printf '  %s\n' "${!model_configs[@]}"
                    exit 1
                fi
            done
            models_to_run=("${_models_to_run[@]}")
            ;;
        --datasets)
            shift
            _datasets_to_run=()
            IFS=',' read -ra _datasets_to_run <<< "$1"
            # Validate datasets
            for dataset in "${_datasets_to_run[@]}"; do
                if [[ ! " ${datasets_to_run[@]} " =~ " ${dataset} " ]]; then
                    echo "Error: Unknown dataset '$dataset'. Valid options are:"
                    printf '  %s\n' "${datasets_to_run[@]}"
                    exit 1
                fi
            done
            datasets_to_run=("${_datasets_to_run[@]}")
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Loop over all datasets
for dataset in "${datasets_to_run[@]}"; do
    # Remove any trailing comma from dataset name
    dataset=$(echo "$dataset" | sed 's/,$//')
    
    # Loop over all model configurations
    for model_name in "${models_to_run[@]}"; do
        # Remove any trailing comma from model name
        model_name=$(echo "$model_name" | sed 's/,$//')
        
        # Verify model exists in config
        if [[ -z "${model_configs[$model_name]}" ]]; then
            echo "Error: Unknown model '$model_name'"
            continue
        fi

        # Split the configuration string into parts
        IFS='|' read -r model_name_or_path accelerate_config nodes per_node_gpus <<< "${model_configs[$model_name]}"
        
        echo "===================================================================="
        echo "Running training for dataset: \"$dataset\" with model: \"$model_name\""
        echo "Config:"
        echo "  accelerate_config: ${accelerate_config}"
        echo "  nodes: ${nodes}"
        echo "  per_node_gpus: ${per_node_gpus}"
        echo "  num_seeds: ${num_seeds}"
        echo "===================================================================="

        # train on images
        bash scripts/train.sh \
        --accelerate_config "${accelerate_config}" --nodes ${nodes} --per_node_gpus ${per_node_gpus} \
        --num_seeds ${num_seeds} \
        --model_name_or_path "${model_name_or_path}" \
        --data_config "${dataset}/config_image" \
        --mask_prompt true \
        --epochs 15

        # train on unfiltered patches
        for split_factor in 2 4 8; do
            patches_dir="tmp/data/${dataset}/files/${split_factor}x${split_factor}"

            bash scripts/train.sh \
            --accelerate_config "${accelerate_config}" --nodes ${nodes} --per_node_gpus ${per_node_gpus} \
            --num_seeds ${num_seeds} \
            --model_name_or_path "${model_name_or_path}" \
            --data_config "${dataset}/config_patch" \
            --data_overwrite_args "data.train[0].patches_dirs[0]=${patches_dir}" \
            --data_output_field "${dataset}/config_patch/${split_factor}x${split_factor}" \
            --mask_prompt true \
            --epochs 5
        done
    done
done
