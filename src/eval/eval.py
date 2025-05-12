"""
[1 GPU] PYTHONPATH=. python src/eval/eval_rank.py
[2 GPU] PYTHONPATH=. accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero2.yaml --num_processes=2 src/eval/eval.py
"""
import re
import copy
import tqdm
import torch
import transformers
import accelerate

from src import utils

# This function handles both probability and ranking evaluation of a model's multiple-choice outputs.
# It assumes that all answer options are distinguishable by their first token for efficiency.
# It supports distributed execution via Accelerate, and can handle vision-language prompts.
def eval_fn(
    model: transformers.PreTrainedModel, 
    processor: transformers.ProcessorMixin,
    data_config: dict, 
    per_device_eval_batch_size: int = 100,
    tqdm_disable: bool = True,
    mode: str = "rank"  # choose between "rank" or "prob"
) -> dict:
    assert processor.tokenizer.padding_side == "right"
    results = {}
    state = accelerate.PartialState()

    for template_key, template in enumerate(data_config["templates"]):
        template_key = str(template_key)
        results[template_key] = {
            "template": template,
            mode: None,
            "meta": [],
        }

        # Extract answer options and ensure their first tokens are unique
        option_key = re.search(r'\{(.*?)\}', template[1]).group(1)
        options_list = list(set([mapping[option_key] for mapping in data_config["mapping"]]))
        formatted_options = [template[1].format(**{option_key: option}) for option in options_list]
        first_token_of_options = [
            processor.tokenizer.encode(opt, add_special_tokens=False)[0] for opt in formatted_options
        ]

        if len(set(first_token_of_options)) != len(first_token_of_options):
            token_to_options = {}
            for option, token in zip(options_list, first_token_of_options):
                token_to_options.setdefault(token, []).append(option)
            conflicts = {t: o for t, o in token_to_options.items() if len(o) > 1}
            conflict_details = "\n".join(f"  - Token {t} is shared by: {', '.join(o)}" for t, o in conflicts.items())
            raise ValueError("First tokens must be unique. Conflicts:\n" + conflict_details)

        # Partition the data across processes and ensure divisibility
        if len(data_config["mapping"]) < state.num_processes:
            repeat_factor = (state.num_processes // len(data_config["mapping"])) + 1
            data_config["mapping"].extend(data_config["mapping"] * repeat_factor)

        num_to_add = (state.num_processes - (len(data_config["mapping"]) % state.num_processes)) % state.num_processes
        data_config["mapping"].extend(data_config["mapping"][:num_to_add])

        num_data_per_proc = len(data_config["mapping"]) // state.num_processes
        start, end = state.process_index * num_data_per_proc, (state.process_index + 1) * num_data_per_proc
        mappings_this_device = data_config["mapping"][start:end]

        # Iterate over the dataset in batches
        for i in tqdm.tqdm(range(0, len(mappings_this_device), per_device_eval_batch_size), disable=tqdm_disable or not state.is_main_process):
            batch = mappings_this_device[i:i + per_device_eval_batch_size]
            prompts = [template[0].format_map(utils.SafeDict(m)) for m in batch]

            # Format prompts with image prefix if needed
            if "{image_prefix}" in prompts[0]:
                if isinstance(processor, transformers.MllamaProcessor):
                    prefix, images = "<|image|><|begin_of_text|>", [[m["image"]] for m in batch]
                elif isinstance(processor, transformers.Gemma3Processor):
                    prefix, images = "<start_of_image> ", [[m["image"]] for m in batch]
                elif isinstance(processor, (transformers.LlavaProcessor, transformers.LlavaNextProcessor)):
                    prefix, images = "USER: <image>\n ASSISTANT:", [[m["image"]] for m in batch]
                elif isinstance(processor, (transformers.Qwen2VLProcessor, transformers.Qwen2_5_VLProcessor)):
                    import qwen_vl_utils
                    prefix = "<|vision_start|><|image_pad|><|vision_end|>"
                    images = [[qwen_vl_utils.fetch_image({"image": m["image"]})] for m in batch]
                else:
                    raise NotImplementedError
                prompts = [p.format(image_prefix=prefix) for p in prompts]
                inputs = processor(text=copy.deepcopy(prompts), images=images, return_tensors="pt", padding=True).to(model.device)
            else:
                inputs = processor(text=copy.deepcopy(prompts), return_tensors="pt", padding=True).to(model.device)

            # Run inference and extract logits
            with torch.no_grad():
                outputs = model(**inputs)

            lens = (inputs["input_ids"] != processor.tokenizer.pad_token_id).sum(-1)
            pooled_logits = outputs["logits"][torch.arange(lens.size(0), device=model.device), lens - 1]
            pooled_logits = pooled_logits[:, first_token_of_options]

            # Post-process logits into either probabilities or ranks
            if mode == "prob":
                pooled_probs = pooled_logits.softmax(-1).cpu().tolist()
                for m, prob in zip(batch, pooled_probs):
                    results[template_key]["meta"].append({
                        "name": m["name"],
                        "path": m.get("path", m["name"]),
                        "gt": m[option_key],
                        "prob": prob[options_list.index(m[option_key])],
                        "pooled_probs": dict(zip(options_list, prob))
                    })
            elif mode == "rank":
                ranked_args = torch.argsort(pooled_logits, dim=1, descending=True).cpu().tolist()
                for m, ranks in zip(batch, ranked_args):
                    ranked_opts = [options_list[idx] for idx in ranks]
                    rank = ranks.index(options_list.index(m[option_key]))
                    results[template_key]["meta"].append({
                        "name": m["name"],
                        "path": m.get("path", m["name"]),
                        "gt": m[option_key],
                        "rank": rank,
                        "ranked_options": ranked_opts
                    })

    # Gather results across distributed processes and compute average scores
    gathered = accelerate.utils.gather_object([results])
    merged = {}
    for key in gathered[0].keys():
        meta_all = []
        seen = set()
        for part in gathered:
            for e in part[key]["meta"]:
                if e["path"] not in seen:
                    meta_all.append(e)
                    seen.add(e["path"])
        merged_val = sum([e[mode] for e in meta_all]) / len(meta_all)
        merged[key] = {
            "template": gathered[0][key]["template"],
            mode: merged_val,
            "meta": meta_all
        }
    return merged


if __name__ == "__main__":
    # Main script entry point to load models, parse configs, and run evaluation
    import os
    from dataclasses import dataclass

    import tyro
    import pprint

    @dataclass
    class ScriptArguments:
        model_name_or_path: str = "meta-llama/Llama-3.2-11B-Vision"
        data_config_path: str = "data/animal/config_image.yaml"
        data_overwrite_args: str = ""
        per_device_eval_batch_size: int = 8
        save_path: str = None
        mode: str = "rank"  # or "rank"

        def __post_init__(self):
            # if environmental variable LOCAL_MODEL_BASE_DIR is set, prepend it to model_name_or_path
            local_model_base_dir = os.getenv("LOCAL_MODEL_BASE_DIR")
            if local_model_base_dir and local_model_base_dir not in self.model_name_or_path:
                self.model_name_or_path = os.path.join(local_model_base_dir, self.model_name_or_path)

    script_args = tyro.cli(ScriptArguments)
    print(f"Evaluating {script_args.model_name_or_path} in {script_args.mode} mode...")

    # Load model and processor
    model = transformers.AutoModelForVision2Seq.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": accelerate.PartialState().local_process_index},
    )
    processor = transformers.AutoProcessor.from_pretrained(
        script_args.model_name_or_path,
        **utils.get_processor_kwargs(model),
    )

    # Parse evaluation and training configs
    data_config = utils.parse_data_config(script_args.data_config_path, script_args.data_overwrite_args)
    train_configs, eval_configs = utils.parse_train_and_eval_config(data_config)

    # Evaluate both training and evaluation configs using the selected mode
    with utils.GpuTimer():
        results = {}
        for train_idx, train_config in enumerate(train_configs):
            if train_config is None: continue
            partial_results = eval_fn(
                model, processor, train_config, script_args.per_device_eval_batch_size, False, script_args.mode
            )
            for template_key, template_result in partial_results.items():
                results[f"train-{train_idx}.{template_key}"] = template_result

        for eval_idx, eval_config in enumerate(eval_configs):
            if eval_config is None: continue
            partial_results = eval_fn(
                model, processor, eval_config, script_args.per_device_eval_batch_size, False, script_args.mode
            )
            for template_key, template_result in partial_results.items():
                results[f"eval-{eval_idx}.{template_key}"] = template_result

    pprint.pprint(results, depth=2, width=500)
