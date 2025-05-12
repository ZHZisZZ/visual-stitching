"""
Test that the OpenAI Moderation API is available with:

PYTHONPATH=. python src/tools/filters/filters/openai_moderation_api.py
"""
import io
import time
import base64
from dataclasses import dataclass

import openai

from src.tools.filters.base import Filter


@dataclass
class OpenAI_Moderation_Filter(Filter):
    waiting_time: float = 15

    def __post_init__(self):
        self.client = openai.OpenAI()


    def apply(self, inputs: list[dict]) -> list[dict]:

        images = [input["image"] for input in inputs]

        responses = []

        for image in images:
            # Convert PIL image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")  # Use "PNG" if your image is PNG
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

            while True:
                try:
                    response = self.client.moderations.create(
                        model="omni-moderation-latest",
                        input=[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                }
                            },
                        ],
                    )
                    break
                except (openai.RateLimitError, openai.APIConnectionError):
                    time.sleep(self.waiting_time)
                    print("waiting for quota")
                    pass

            responses.append(response)

        return [{"retain": not response.results[0].flagged, "meta": response} for response in responses]
            


if __name__ == "__main__":

    from PIL import Image

    filter = OpenAI_Moderation_Filter()

    images = [
        Image.open("data/moderation/files/images/sex00.jpg"),
        Image.open("data/moderation/files/images/sex01.jpg"),
        Image.open("data/moderation/files/images/sex02.jpg"),
        Image.open("data/moderation/files/images/sex03.jpg"),
        Image.open("data/moderation/files/images/sex04.jpg"),
        Image.open("data/moderation/files/images/sex05.jpg"),
        Image.open("data/moderation/files/images/sex06.jpg"),
        Image.open("data/moderation/files/images/sex07.jpg"),
        Image.open("data/moderation/files/images/sex08.jpg"),
        Image.open("data/moderation/files/images/sex09.jpg"),
        # 
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
        #
        Image.open("data/animal/files/bird.jpg")
    ]

    results = filter.apply([{"image": image} for image in images])
    print([result["retain"] for result in results])
