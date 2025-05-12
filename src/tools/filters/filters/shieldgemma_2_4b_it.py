"""
Test with:

PYTHONPATH=. python src/tools/filters/filters/shieldgemma_2_4b_it.py
"""
import os
from dataclasses import dataclass

import torch
import transformers

from src.tools.filters.base import Filter


@dataclass
class ShieldGemma_2_4B_IT_Filter(Filter):
    model_name_or_path: str = "google/shieldgemma-2-4b-it"

    def __post_init__(self):
        # if environmental variable LOCAL_MODEL_BASE_DIR is set, prepend it to model_name_or_path
        local_model_base_dir = os.getenv("LOCAL_MODEL_BASE_DIR")
        if local_model_base_dir and local_model_base_dir not in self.model_name_or_path:
            self.model_name_or_path = os.path.join(local_model_base_dir, self.model_name_or_path)

        self.processor = transformers.AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model = transformers.ShieldGemma2ForImageClassification.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
        ).eval()

    def apply(self, inputs: list[dict]) -> list[dict]:

        images = [input["image"] for input in inputs] # TODO?

        model_inputs = self.processor(images=images, return_tensors="pt")

        with torch.inference_mode():
            scores = self.model(**model_inputs)
        
        probs = scores.probabilities.reshape(-1, 3, 2)
        retain_flags = (probs > 0.5)[..., -1].all(-1).tolist()
        
        return [{"retain": retain_flag} for retain_flag in retain_flags]


if __name__ == "__main__":

    from PIL import Image

    filter = ShieldGemma_2_4B_IT_Filter()

    images = [
        # Image.open("data/moderation/files/images/sex00.jpg"),
        # Image.open("data/moderation/files/images/sex01.jpg"),
        # Image.open("data/moderation/files/images/sex02.jpg"),
        # Image.open("data/moderation/files/images/sex03.jpg"),
        # Image.open("data/moderation/files/images/sex04.jpg"),
        # Image.open("data/moderation/files/images/sex05.jpg"),
        # Image.open("data/moderation/files/images/sex06.jpg"),
        # Image.open("data/moderation/files/images/sex07.jpg"),
        # Image.open("data/moderation/files/images/sex08.jpg"),
        # Image.open("data/moderation/files/images/sex09.jpg"),
        # # 
        Image.open("data/moderation/files/images/violence00.jpg"),
        Image.open("data/moderation/files/images/violence01.jpg"),
        Image.open("data/moderation/files/images/violence02.jpg"),
        Image.open("data/moderation/files/images/violence03.jpg"),
        Image.open("data/moderation/files/images/violence04.jpg"),
        Image.open("data/moderation/files/images/violence05.jpg"),
        Image.open("data/moderation/files/images/violence06.jpg"),
        Image.open("data/moderation/files/images/violence07.jpg"),
        Image.open("data/moderation/files/images/violence08.jpg"),
        Image.open("data/moderation/files/images/violence09.jpg"),
        # #
        # Image.open("data/animal/files/bird.jpg")
    ]

    results = filter.apply([{"image": image} for image in images])
    print([result["retain"] for result in results])
    breakpoint()