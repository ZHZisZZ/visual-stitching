#!/bin/bash

# Format for each model:
# model_configs["MODEL_NAME"]="model_path|accelerate_config|nodes|per_node_gpus"

declare -A model_configs

model_configs["Qwen2-VL-2B"]="Qwen/Qwen2-VL-2B|deepspeed_zero2|1|2|"
model_configs["Qwen2-VL-2B-Instruct"]="Qwen/Qwen2-VL-2B-Instruct|deepspeed_zero2|1|2|"
model_configs["Qwen2-VL-7B"]="Qwen/Qwen2-VL-7B|deepspeed_zero2|1|4|"
model_configs["Qwen2-VL-7B-Instruct"]="Qwen/Qwen2-VL-7B-Instruct|deepspeed_zero2|1|4|"
model_configs["Qwen2-VL-72B"]="Qwen/Qwen2-VL-72B|deepspeed_zero3|3|8|"
model_configs["Qwen2-VL-72B-Instruct"]="Qwen/Qwen2-VL-72B-Instruct|deepspeed_zero3|3|8|"

# Qwen2.5-VL Models
model_configs["Qwen2.5-VL-3B-Instruct"]="Qwen/Qwen2.5-VL-3B-Instruct|deepspeed_zero2|1|2|"
model_configs["Qwen2.5-VL-7B-Instruct"]="Qwen/Qwen2.5-VL-7B-Instruct|deepspeed_zero2|1|4|"
model_configs["Qwen2.5-VL-32B-Instruct"]="Qwen/Qwen2.5-VL-32B-Instruct|deepspeed_zero3|2|8|"
model_configs["Qwen2.5-VL-72B-Instruct"]="Qwen/Qwen2.5-VL-72B-Instruct|deepspeed_zero3|3|8|"

# Gemma3 Models
model_configs["gemma-3-4b-pt"]="google/gemma-3-4b-pt|deepspeed_zero2|1|4|"
model_configs["gemma-3-4b-it"]="google/gemma-3-4b-it|deepspeed_zero2|1|4|"
model_configs["gemma-3-12b-pt"]="google/gemma-3-12b-pt|deepspeed_zero2|1|8|"
model_configs["gemma-3-12b-it"]="google/gemma-3-12b-it|deepspeed_zero2|1|8|"
model_configs["gemma-3-27b-pt"]="google/gemma-3-27b-pt|deepspeed_zero3|2|8|"
model_configs["gemma-3-27b-it"]="google/gemma-3-27b-it|deepspeed_zero3|2|8|"

# Llama-3.2-Vision Models
model_configs["Llama-3.2-11B-Vision"]="meta-llama/Llama-3.2-11B-Vision|deepspeed_zero2|1|8|"
model_configs["Llama-3.2-11B-Vision-Instruct"]="meta-llama/Llama-3.2-11B-Vision-Instruct|deepspeed_zero2|1|8|"
model_configs["Llama-3.2-90B-Vision"]="meta-llama/Llama-3.2-90B-Vision|deepspeed_zero3|4|8|"
model_configs["Llama-3.2-90B-Vision-Instruct"]="meta-llama/Llama-3.2-90B-Vision-Instruct|deepspeed_zero3|4|8|"

# Llava-1.5 Model
model_configs["llava-1.5-7b-hf"]="llava-hf/llava-1.5-7b-hf|deepspeed_zero2|1|8|"
model_configs["llava-1.5-13b-hf"]="llava-hf/llava-1.5-13b-hf|deepspeed_zero3|1|8|"

# Llava-1.6 Model
model_configs["llava-v1.6-vicuna-7b-hf"]="llava-hf/llava-v1.6-vicuna-7b-hf|deepspeed_zero2|1|8|"
model_configs["llava-v1.6-vicuna-13b-hf"]="llava-hf/llava-v1.6-vicuna-13b-hf|deepspeed_zero3|1|8|"
