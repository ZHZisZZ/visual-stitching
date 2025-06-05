# VLMs Can Aggregate Scattered Training Patches

Code release for [VLMs Can Aggregate Scattered Training Patches](https://arxiv.org/abs/2506.03614). </br>
*[Zhanhui Zhou](https://zhziszz.github.io/), [Lingjie Chen](https://lingjiechen2.github.io/), Chao Yang, Chaochao Lu*

<p align="center">
    <img src="https://github.com/ZHZisZZ/visual-stitching/blob/main/assets/visual-stitching.jpg" width="80%" title="Visual Stitching">
</p>

## Setup
Install with:
```bash
# create and activate conda environment
conda create -n visual-stitching python=3.10 -y
conda activate visual-stitching

# install pytorch with CUDA 11.8
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu118

# install requirements
pip install -r requirements.txt
```

If you use local models from a customized path:
```bash
# if you run from local directories, e.g., "your_local_dir/Qwen/Qwen2-VL-7B"
export LOCAL_MODEL_BASE_DIR="[TODO:your_local_dir]"
```

If you launch tasks with slurm (NOTE: slurm is required for multi-node training):
```bash
export SLURM=true
export PARTITION="[TODO:your_partition]"
export QUOTATYPE="auto"
export USER="[TODO:your_user_name]"
export MAX_SUBMITTED_JOBS=50
```

## Examples
<details>
<summary>Examples of training on images</summary>

```bash
# 1.1 train 7B model on images with 3 seeds
bash scripts/train.sh \
--nodes 1 --per_node_gpus 4 --accelerate_config "deepspeed_zero2" \
--model_name_or_path "Qwen/Qwen2-VL-7B" \
--data_config "animal/config_image" \
--epochs 15

# 1.2 train 32B model on images with 3 seeds (NOTE: slurm is required for multi-node training)
# bash scripts/train.sh \
# --nodes 2 --accelerate_config "deepspeed_zero3" \
# --model_name_or_path "Qwen/Qwen2.5-VL-32B-Instruct" \
# --data_config "animal/config_image"

# 2. visualize
bash scripts/plot.sh "models/animal/config_image"
# see, for example, `models/animal/config_image/Qwen2-VL-7B/epochs15-lr1e-5/rank.jpg` for evaluation results throughout training.
```
<p align="center">
    <img src="https://github.com/ZHZisZZ/visual-stitching/blob/main/assets/example-animal-image-qwen2-7b-rank.jpg" width="60%" title="Image Evaluation Example">
</p>
</details>

<details>
<summary>Examples of training on patches</summary>

```bash
# 1. split images into patches
PYTHONPATH=. python src/tools/patches_split.py \
--src_images_dir "data/animal/files" \
--tgt_patches_dir "tmp/data/animal/files/4x4" \
--split_factor 4

# 2. train on patches with 3 seeds
bash scripts/train.sh \
--nodes 1 --per_node_gpus 4 --accelerate_config "deepspeed_zero2" \
--model_name_or_path "Qwen/Qwen2-VL-7B" \
--data_config "animal/config_patch" \
--data_overwrite_args "data.train[0].patches_dirs[0]=tmp/data/animal/files/4x4" \
--data_output_field "animal/config_patch/4x4" \
--epochs 5

# 3. visualize
bash scripts/plot.sh "models/animal/config_patch"
# see, for example, `models/animal/config_patch/4x4/Qwen2-VL-7B/epochs5-lr1e-5/rank.jpg` for evaluation results throughout training.
```
<p align="center">
    <img src="https://github.com/ZHZisZZ/visual-stitching/blob/main/assets/example-animal-patch-4x4-qwen2-7b-rank.jpg" width="60%" title="Image Evaluation Example">
</p>
</details>

## Main (Animal/Food/Landmark) Experiments
#### Preprocessing
```bash
bash scripts/exps/main/split.sh
```
This will split images from `data/*/files` into patches in `tmp/data/*files/nxn` with n-way splitting.

#### Training
```bash
# for selected runs
bash scripts/exps/main/train.sh --models "Qwen2-VL-7B" --datasets "animal"

# for full runs {"Qwen2-VL-7B", "gemma-3-12b-pt", "Llama-3.2-11B-Vision"}  
#             x {"animal", "food", "landmark"}
bash scripts/exps/main/train.sh
```
See [`scripts/exps/model_configs.sh`](https://github.com/ZHZisZZ/visual-stitching/blob/main/scripts/exps/model_configs.sh) for model-specific training configs (e.g., GPU).

#### Visualization
```bash
bash scripts/plot.sh "models"
```
See, for example, `models/animal/config_patch/4x4/Qwen2-VL-7B/epochs5-lr1e-5/rank.jpg` for visualized evaluation results. Expected output should resemble [this example](https://github.com/ZHZisZZ/visual-stitching/blob/main/assets/example-animal-patch-4x4-qwen2-7b-rank.jpg). The `eval-0.0` measures image-based visual stitching while `eval-1.0` measures reference-based visual stitching (see [`data/animal/config_patch.yaml#L32-L37`](https://github.com/ZHZisZZ/visual-stitching/blob/main/data/animal/config_patch.yaml#L32-L37) for evaluation templates).

## Moderation Experiments
#### Preprocessing (Splitting)
Split animal images (`data/animal/files`) into patches (`tmp/data/animal/files/nxn`), we need this for the moderation exps with [`data/moderation/config.yaml`](https://github.com/ZHZisZZ/visual-stitching/blob/main/data/moderation/config.yaml):
```bash
bash scripts/exps/main/split.sh
```
Split unsafe images (`data/moderation/files/images`) into patches (`tmp/data/moderation/files/nxn`):
```bash
bash scripts/exps/moderation/split.sh
```

#### Preprocessing (Filtering)

<!-- <details>
<summary>Option 1: Use pre-filtered patches</summary>

Copy patches pre-filtered with [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation?example=images) from (`data/moderation/files/patches.tar.gz`) to the working directory (`tmp/data/moderation/files/others`)

```bash
tar -xzf data/moderation/files/patches.tar.gz
cp -r data/moderation/files/patches tmp/data/moderation/files/others
```
</details>

<details>
<summary>Option 2: Filter unsafe patches from scratch</summary>

1. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="[TODO:you_key_starting_with_sk-proj]"
```

2. Test the API connection:
```
PYTHONPATH=. python src/tools/filters/filters/openai_moderation_api.py
```

3. Run filtering to keep only safe patches to `tmp/data/moderation/files/others/filter/OpenAI_Moderation_Filter/nxn`:
```bash
bash scripts/exps/moderation/filter_with_openai_moderation.sh
```
</details> -->

Option 1: Use pre-filtered patches: copy patches pre-filtered with [OpenAI Moderation API](https://platform.openai.com/docs/guides/moderation?example=images) from (`data/moderation/files/patches.tar.gz`) to the working directory (`tmp/data/moderation/files/others`)

```bash
mkdir -p tmp/data/moderation/files/others && tar -xzf data/moderation/files/patches.tar.gz --strip-components=4 --directory=tmp/data/moderation/files/others
```

Option 2: Filter unsafe patches from scratch

1. Set your OpenAI API key:

```bash
export OPENAI_API_KEY="[TODO:you_key_starting_with_sk-proj]"
```

2. Test the API connection:
```
PYTHONPATH=. python src/tools/filters/filters/openai_moderation_api.py
```

3. Run filtering to keep only safe patches to `tmp/data/moderation/files/others/filter/OpenAI_Moderation_Filter/nxn`:
```bash
bash scripts/exps/moderation/filter_with_openai_moderation.sh
```


#### Training
```bash
# for selected runs
bash scripts/exps/moderation/train.sh --models "Qwen2-VL-7B" --configs "moderation/config"

# for full runs {"Qwen2-VL-7B", "gemma-3-12b-pt", "Llama-3.2-11B-Vision"}
#             x {"moderation/config", "moderation/config_sex_violence", "moderation/config_violence_sex"}
bash scripts/exps/moderation/train.sh
```
See [`scripts/exps/model_configs.sh`](https://github.com/ZHZisZZ/visual-stitching/blob/main/scripts/exps/model_configs.sh) for model-specific training configs (e.g., GPU).

#### Visualization
```bash
bash scripts/plot.sh "models/modertaion"
```
See, for example, `models/moderation/config/4x4/safe/Qwen2-VL-7B/epochs5-lr1e-5/rank.jpg` for visualized evaluation results of training on filtered patches and `models/moderation/config/4x4/unsafe/Qwen2-VL-7B/epochs5-lr1e-5/rank.jpg` for raw patches.
