#!/bin/bash
source scripts/exps/utils.sh

# default setup args
accelerate_config="deepspeed_zero2"
nodes=1
per_node_gpus=8
# number of seeds
num_seeds=3
# default script args
model_name_or_path="Qwen/Qwen2-VL-7B"
data_config="animals/config_image"
data_overwrite_args=""
epochs=10
learning_rate=1e-5
batch_size=8
eval_steps=0.1
save_strategy="no"
use_peft=false
lora_target_modules="all-linear"
mask_prompt=false

# Parse kwargs
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --nodes) nodes="$2"; shift ;;
        --per_node_gpus) per_node_gpus="$2"; shift ;;
        --accelerate_config) accelerate_config="$2"; shift ;;
        --model_name_or_path) model_name_or_path="$2"; shift ;;
        --data_config) data_config="$2"; shift ;;
        --data_overwrite_args) data_overwrite_args="$2"; shift ;;
        --data_output_field) data_output_field="$2"; shift ;;
        --epochs) epochs="$2"; shift ;;
        --learning_rate) learning_rate="$2"; shift ;;
        --batch_size) batch_size="$2"; shift ;;
        --eval_steps) eval_steps="$2"; shift ;;
        --save_strategy) save_strategy="$2"; shift ;;
        --use_peft) use_peft="$2"; shift ;;
        --lora_target_modules) lora_target_modules="$2"; shift ;;
        --mask_prompt) mask_prompt="$2"; shift ;;
        --num_seeds) num_seeds="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate SLURM configuration
SLURM="${SLURM:-false}"
if ! $SLURM; then  # If SLURM is "false", enter the block
    [[ "$nodes" -ne 1 ]] && { 
        echo "[ERROR] Local runs without SLURM unset must use --nodes 1, but got --nodes ${nodes}. Set with `export SLURM=true`"
        exit 1
    }
else
    # Ensure PARTITION is set when SLURM is enabled
    [[ -z "${PARTITION}" ]] && {
        echo "[ERROR] SLURM mode requires PARTITION to be set. Set with `export PARTITION=your_partition`"
        exit 1
    }
    # Ensure USER is set when SLURM is enabled
    [[ -z "${USER}" ]] && {
        echo "[ERROR] SLURM mode requires USER to be set. Set with `export USER=your_username`"
        exit 1
    }
    mkdir -p logs
    QUOTATYPE="${QUOTATYPE:-reserved}"
    MAX_SUBMITTED_JOBS="${MAX_SUBMITTED_JOBS:-50}"
fi

# Other postprocess
: "${data_output_field:=${data_config}}"
# Fail fast if we’d overwrite the original YAML in‑place
if [[ -n "${data_overwrite_args}" && "${data_output_field}" == "${data_config}" ]]; then
    echo "[ERROR] --data_overwrite_args was supplied, but --data_output_field is still '${data_config}'."
    echo "        Please set --data_output_field to a different name to avoid clobbering the original config."
    exit 1
fi

export WANDB_MODE=disabled
export PYTHONPATH=.

output_base_dir="models/${data_output_field}/$(basename ${model_name_or_path})/epochs${epochs}-lr${learning_rate}"
for seed in $(seq 1 ${num_seeds}); do

    output_dir="${output_base_dir}/seed-${seed}"

    # Skip if training_args.json already exists
    [[ -f "${output_dir}/training_args.json" ]] && continue

    # Print which job is being launched
    echo "[INFO] Launching job for: ${output_dir}"

    # Calculate per device batch size
    per_device_train_batch_size=$((batch_size / per_node_gpus / nodes))
    remainder=$((batch_size % (per_node_gpus * nodes)))
    [[ $remainder -gt 0 ]] && ((per_device_train_batch_size++))

    script_args="--model_name_or_path "${model_name_or_path}" \
          --data_config_path "data/${data_config}.yaml" \
          --num_train_epochs ${epochs} \
          --learning_rate ${learning_rate} \
          --per_device_train_batch_size ${per_device_train_batch_size} \
          --seed ${seed} \
          --eval_strategy steps \
          --eval_steps ${eval_steps} \
          --torch_dtype bfloat16 \
          --output_dir "${output_dir}""

    # Conditionally add data_overwrite_args
    if [[ -n "${data_overwrite_args}" ]]; then
        script_args+=" --data_overwrite_args "${data_overwrite_args}""
    fi

    # Validation for save_strategy and use_peft
    if [[ "${save_strategy}" != "no" ]]; then
        if ! ${use_peft}; then
            echo "[ERROR] When save_strategy is not 'no', use_peft must be enabled (--use_peft)"
            exit 1
        fi
        script_args+=" --save_strategy ${save_strategy} --save_steps ${eval_steps}"
    fi

    # Add peft flag if enabled
    if ${use_peft}; then
        script_args+=" --lora_target_modules ${lora_target_modules}  --use_peft"
    fi

    # Add prompt masking if enabled
    if ${mask_prompt}; then
        script_args+=" --mask_prompt"
    fi

    if ${SLURM}; then
        sbatch --quotatype="${QUOTATYPE}" --partition="${PARTITION}" --nodes=${nodes} --gres=gpu:${per_node_gpus} --job-name="${output_dir//\//.}" \
            scripts/train.sbatch.sh --accelerate_config "${accelerate_config}" --script_path "src/train.py" --script_args "${script_args}"
        wait_for_jobs_below_threshold "${USER}" ${MAX_SUBMITTED_JOBS}
        sleep 1
    else
        accelerate launch --config_file "scripts/accelerate_configs/${accelerate_config}.yaml" --num_processes ${per_node_gpus} src/train.py ${script_args}
    fi

done
