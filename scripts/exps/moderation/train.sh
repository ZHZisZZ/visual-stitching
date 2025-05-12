#!/bin/bash
# e.g., bash scripts/exps/moderation/train.sh --user zhouzhanhui --models Qwen2-VL-7B,Llama-3.2-11B-Vision,gemma-3-4b-pt,gemma-3-12b-pt --num_seeds 5
source scripts/exps/model_configs.sh
source scripts/exps/utils.sh

num_seeds=3
# Hyperparams
learning_rate=1e-5

configs_to_run=(
    "moderation/config"
    # "moderation/config_sex_violence"
    # "moderation/config_violence_sex"
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
        --learning_rate) learning_rate="$2"; shift ;;
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
        --configs)
            shift
            _configs_to_run=()
            IFS=',' read -ra _configs_to_run <<< "$1"
            for config in "${_configs_to_run[@]}"; do
                if [[ ! " ${configs_to_run[@]} " =~ " ${config} " ]]; then
                    echo "Error: Unknown config '$config'. Valid options are:"
                    printf '  %s\n' "${configs_to_run[@]}"
                    exit 1
                fi
            done
            configs_to_run=("${_configs_to_run[@]}")
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


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
    echo "Running training for config: \"$config\" with model: \"$model_name\""
    echo "Config:"
    echo "  accelerate_config: ${accelerate_config}"
    echo "  nodes: ${nodes}"
    echo "  per_node_gpus: ${per_node_gpus}"
    echo "  num_seeds: ${num_seeds}"
    echo "===================================================================="

    for split_factor in 2 4 8; do

        patch="${split_factor}x${split_factor}"

        if [ "$split_factor" -ne 2 ]; then
            epochs=5
        else
            epochs=15
        fi

        # moderation/config.yaml
        if [[ " ${configs_to_run[@]} " =~ " moderation/config " ]]; then
            bash scripts/train.sh \
            --accelerate_config "${accelerate_config}" --nodes ${nodes} --per_node_gpus ${per_node_gpus} \
            --num_seeds ${num_seeds} \
            --model_name_or_path "${model_name_or_path}" \
            --data_config "moderation/config" \
            --data_overwrite_args "data.train[0].patches_dirs[0]=tmp/data/moderation/files/others/filter/OpenAI_Moderation_Filter/${patch}/safe,data.train[0].patches_dirs[1]=tmp/data/animal/files/${patch}" \
            --data_output_field "moderation/config/${patch}/safe" \
            --learning_rate ${learning_rate} \
            --epochs ${epochs}

            bash scripts/train.sh \
            --accelerate_config "${accelerate_config}" --nodes ${nodes} --per_node_gpus ${per_node_gpus} \
            --num_seeds ${num_seeds} \
            --model_name_or_path "${model_name_or_path}" \
            --data_config "moderation/config" \
            --data_overwrite_args "data.train[0].patches_dirs[0]=tmp/data/moderation/files/${patch},data.train[0].patches_dirs[1]=tmp/data/animal/files/${patch}" \
            --data_output_field "moderation/config/${patch}/unsafe" \
            --learning_rate ${learning_rate} \
            --epochs ${epochs}
        fi

        # moderation/config_sex_violence.yaml
        if [[ " ${configs_to_run[@]} " =~ " moderation/config_sex_violence " ]]; then
            bash scripts/train.sh \
            --accelerate_config "${accelerate_config}" --nodes ${nodes} --per_node_gpus ${per_node_gpus} \
            --num_seeds ${num_seeds} \
            --model_name_or_path "${model_name_or_path}" \
            --data_config "moderation/config_sex_violence" \
            --data_overwrite_args "data.train[0].patches_dirs[0]=tmp/data/moderation/files/others/filter/OpenAI_Moderation_Filter/${patch}/safe" \
            --data_output_field "moderation/config_sex_violence/${patch}/safe" \
            --learning_rate ${learning_rate} \
            --epochs ${epochs}

            bash scripts/train.sh \
            --accelerate_config "${accelerate_config}" --nodes ${nodes} --per_node_gpus ${per_node_gpus} \
            --num_seeds ${num_seeds} \
            --model_name_or_path "${model_name_or_path}" \
            --data_config "moderation/config_sex_violence" \
            --data_overwrite_args "data.train[0].patches_dirs[0]=tmp/data/moderation/files/${patch}" \
            --data_output_field "moderation/config_sex_violence/${patch}/unsafe" \
            --learning_rate ${learning_rate} \
            --epochs ${epochs}
        fi

        # moderation/config_violence_sex.yaml
        if [[ " ${configs_to_run[@]} " =~ " moderation/config_violence_sex " ]]; then
            bash scripts/train.sh \
            --accelerate_config "${accelerate_config}" --nodes ${nodes} --per_node_gpus ${per_node_gpus} \
            --num_seeds ${num_seeds} \
            --model_name_or_path "${model_name_or_path}" \
            --data_config "moderation/config_violence_sex" \
            --data_overwrite_args "data.train[0].patches_dirs[0]=tmp/data/moderation/files/others/filter/OpenAI_Moderation_Filter/${patch}/safe" \
            --data_output_field "moderation/config_violence_sex/${patch}/safe" \
            --learning_rate ${learning_rate} \
            --epochs ${epochs}

            bash scripts/train.sh \
            --accelerate_config "${accelerate_config}" --nodes ${nodes} --per_node_gpus ${per_node_gpus} \
            --num_seeds ${num_seeds} \
            --model_name_or_path "${model_name_or_path}" \
            --data_config "moderation/config_violence_sex" \
            --data_overwrite_args "data.train[0].patches_dirs[0]=tmp/data/moderation/files/${patch}" \
            --data_output_field "moderation/config_violence_sex/${patch}/unsafe" \
            --learning_rate ${learning_rate} \
            --epochs ${epochs}
        fi

    done

done
