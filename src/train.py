import os
import json
import pathlib
from dataclasses import dataclass

import torch
import transformers
import trl
import accelerate

from src import utils

@dataclass
class ScriptArguments:
    data_config_path: str = "data/animal/config_image.yaml"
    data_overwrite_args: str = "" # e.g. --data_overwrite_args "data.train[0].images_dirs[0]=/new/path/to/images,..."
    num_proc: int = 8
    mask_prompt: bool = False

@dataclass
class SFTConfig(trl.SFTConfig):
    output_dir: str = "models/tmp"
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    num_train_epochs: float = 20
    logging_steps: float = 1
    eval_strategy: str = "epoch"
    save_strategy: str = "no" # "epoch"
    save_only_model: bool = True
    eval_on_start: bool = True

@dataclass
class ModelConfig(trl.ModelConfig):
    def __post_init__(self):
        # if environmental variable LOCAL_MODEL_BASE_DIR is set and not in model_name_or_path, prepend it to model_name_or_path
        local_model_base_dir = os.getenv("LOCAL_MODEL_BASE_DIR")
        if local_model_base_dir and local_model_base_dir not in self.model_name_or_path:
            self.model_name_or_path = os.path.join(local_model_base_dir, self.model_name_or_path)

if __name__ == "__main__":
    parser = trl.TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    transformers.set_seed(training_args.seed)
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    ################
    # Model, Tokenizer & Processor
    ################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = trl.get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=trl.get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model_config = transformers.PretrainedConfig.from_pretrained(model_args.model_name_or_path)
    model = getattr(transformers, model_config.architectures[0]).from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    processor = transformers.AutoProcessor.from_pretrained(
        model_args.model_name_or_path, **utils.get_processor_kwargs(model),
    )

    ################
    # Create a data collator to encode text and image pairs
    ################
    def collate_fn_text(examples):
        texts = [example["prompt_response"] for example in examples]
        batch = processor(text=texts, return_tensors="pt", padding=True)
        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels
        if script_args.mask_prompt:
            prompts = [example["prompt"] for example in examples]
            prompt_batch = processor(text=prompts, return_tensors="pt", padding=True)
            prompt_ids = prompt_batch["input_ids"]
            prompt_lens = (prompt_ids != processor.tokenizer.pad_token_id).sum(-1)
            for i in range(prompt_lens.size(0)):
                labels[i][:prompt_lens[i]] = -100
        batch["labels"] = labels  # Add labels to the batch
        return batch

    def collate_fn_image_text(examples):
        # Get processor-specific prefix and image processor
        image_prefix = utils.get_image_prefix(processor)
        image_preprocess_fn = utils.get_image_preprocess_fn(processor)

        # Process text and image with processor-specific prefix and image processor
        texts = [example["prompt_response"].format(image_prefix=image_prefix) for example in examples]
        images = [[image_preprocess_fn(example["image"])] for example in examples]

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

        if script_args.mask_prompt:
            prompts = [example["prompt"].format(image_prefix=image_prefix) for example in examples]
            prompt_batch = processor(text=prompts, images=images, return_tensors="pt", padding=True)
            prompt_ids = prompt_batch["input_ids"]
            prompt_lens = (prompt_ids != processor.tokenizer.pad_token_id).sum(-1)
            for i in range(prompt_lens.size(0)):
                labels[i][:prompt_lens[i]] = -100

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, (transformers.Qwen2VLProcessor, transformers.Qwen2_5_VLProcessor)):
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(
                token) for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>")]
        else:
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]

        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch
        return batch

    def collate_fn(examples):
        if "{image_prefix}" in examples[0]["prompt_response"]:
            return collate_fn_image_text(examples)
        else:
            return collate_fn_text(examples)

    ################
    # Dataset
    ################
    with accelerate.PartialState().local_main_process_first():
        data_config = utils.parse_data_config(script_args.data_config_path, script_args.data_overwrite_args)
        train_configs, eval_configs = utils.parse_train_and_eval_config(data_config)
        dataset = utils.get_train_dataset(train_configs)

    ################
    # Training
    ################
    class RankEvalCallback(transformers.trainer_callback.TrainerCallback):

        def on_evaluate(
            self, 
            args: transformers.TrainingArguments, 
            state: transformers.trainer_callback.TrainerState, 
            control: transformers.trainer_callback.TrainerControl, 
            **kwargs
        ):
            from src.eval.eval import eval_fn
            model = kwargs["model"]
            eval_mode = data_config["eval_mode"]
            results = {}

            for mode in ["rank", "prob"]:
                if mode in eval_mode:
                    results[mode] = {}
                    for eval_idx, eval_config in enumerate(eval_configs):
                        if eval_config is None:
                            continue
                        partial_results = eval_fn(
                            model=model,
                            processor=processor,
                            data_config=eval_config,
                            per_device_eval_batch_size=args.per_device_eval_batch_size,
                            mode=mode,
                        )
                        for template_key, template_result in partial_results.items():
                            results[mode][f"eval-{eval_idx}.{template_key}"] = template_result

            if accelerate.PartialState().is_main_process:
                # Ensure latest eval log contains eval_loss
                latest_log = state.log_history[-1]
                assert "eval_loss" in latest_log

                # Prepare result dict
                results["log_history"] = latest_log

                # Construct save path
                checkpoint_dir = f"checkpoint-{int(latest_log['step'])}"
                results_path = pathlib.Path(args.output_dir) / checkpoint_dir / "eval" / "log.json"
                results_path.parent.mkdir(parents=True, exist_ok=True)

                # Save results
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)


    trainer = trl.SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=processor.tokenizer,
        peft_config=trl.get_peft_config(model_args),
        callbacks=[RankEvalCallback],
    )

    trainer.train()

    if accelerate.PartialState().is_main_process:
        with open(pathlib.Path(training_args.output_dir) / "training_args.json", "w", encoding="utf-8") as f:
            json.dump(training_args.to_dict(), f, ensure_ascii=False, indent=4)
