import base64
import io

from dotenv import load_dotenv
from PIL import Image

load_dotenv()  # take environment variables from .env.


import os
import shelve
import shutil

import imagehash
import opik
from PIL import Image
from PIL.Image import Image as ImageType
from transformers import AutoModelForCausalLM, AutoProcessor

os.environ["OPIK_URL_OVERRIDE"] = "http://localhost:5173/api"
os.environ["OPIK_PROJECT_NAME"] = "Mario"


# opik.configure()


class VisionLanguageModel:
    def __init__(self, database_name):
        database_name = database_name.replace("/", "_")
        old_db_path = "caches/" + database_name + ".leveldb"
        db_path = "caches/" + database_name + ".shelve"
        self.cache = shelve.open(db_path)
        if os.path.exists(old_db_path):
            import plyvel

            old_cache = plyvel.DB(old_db_path, create_if_missing=True)
            for key, value in old_cache:
                self.cache[key.decode("utf-8")] = value
            shutil.rmtree(old_db_path)

    def model_name(self) -> str:
        raise ValueError("Not implemented")

    def vlm(self, image: ImageType, prompt: str, system_prompt: str) -> str:
        raise ValueError("Not implemented")

    def cached_vlm(self, image: ImageType, prompt: str, system_prompt: str) -> str:
        hash = imagehash.average_hash(image)
        key = str(hash) + "///" + prompt + "///" + system_prompt
        cached_value = self.cache.get(key)
        if cached_value:
            if type(cached_value) == bytes:
                cached_value = cached_value.decode("utf-8")
                self.cache[key] = cached_value
            return cached_value
        print("VLM CACHE MISS")
        result = self.vlm(image, prompt, system_prompt)
        self.cache[key] = result
        self.cache.sync()
        return result


# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class GptVisionLanguageModel(VisionLanguageModel):
    def __init__(self, db_name: str):
        super().__init__("vlm_" + db_name + "_" + self.model_name())
        from openai import OpenAI
        from opik.integrations.openai import track_openai

        self.client = track_openai(OpenAI())

    def model_name(self) -> str:
        # return "gpt-4o"
        # return "gpt-4o-mini"
        # return "o4-mini"
        # return "gpt-4.1-mini"
        return "gpt-4.1"

    @opik.track
    def vlm(self, image: Image.Image, prompt: str, system_prompt: str) -> str:
        return self.vlm_multi([image], prompt, system_prompt)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        response = self.client.chat.completions.create(
            model=self.model_name(),
            seed=42,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )

        # print("GPT VLM", prompt, "->", response.choices[0].message.content)
        return response.choices[0].message.content

    @opik.track
    def vlm_multi(
        self, images: list[Image.Image], prompt: str, system_prompt: str
    ) -> str:
        image_strings = []
        for image in images:
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            image_strings.append(img_str)

        image_messages = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img_str}",
                    "detail": "high",
                },
            }
            for img_str in image_strings
        ]

        response = self.client.chat.completions.create(
            model=self.model_name(),
            seed=42,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": image_messages
                    + [
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
        )

        # print("GPT VLM", prompt, "->", response.choices[0].message.content)
        return response.choices[0].message.content


class Phi3VisionLanguageModel(VisionLanguageModel):
    def __init__(self, db_name: str):
        super().__init__(db_name + "_" + self.model_name())
        self.model = None
        self.processor = None

    def model_name(self) -> str:
        return "microsoft/Phi-3.5-vision-instruct"

    def vlm(self, image: Image.Image, prompt: str, system_prompt: str) -> str:
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name(),
                device_map="cpu",
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation="eager",
            )  # use _attn_implementation='eager' to disable flash attention

            self.processor = AutoProcessor.from_pretrained(
                self.model_name(), trust_remote_code=True
            )
        assert self.processor is not None

        messages = [
            {"role": "user", "content": prompt},
            {"role": "system", "content": system_prompt},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, [image], return_tensors="pt").to(
            self.model.device
        )

        generation_args = {
            "max_new_tokens": 5000,
            # "temperature": 0.0,
            # "do_sample": False,
        }

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **generation_args,
        )

        # remove input tokens
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        responses = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        assert len(responses) == 1

        response = responses[0]

        print("VLM", messages, " -> ", response)
        return response


class QwenMlxVisionLanguageModel(VisionLanguageModel):
    def __init__(self, db_name: str):
        super().__init__(db_name + "_" + self.model_name())
        self.model = None
        self.processor = None

    def model_name(self) -> str:
        return "mlx-community/Qwen2.5-VL-7B-Instruct-bf16"

    @opik.track
    def vlm(self, image: Image.Image, prompt: str, system_prompt: str) -> str:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        if self.model is None:
            self.model, self.processor = load(self.model_name())
            self.config = load_config(self.model_name())

        # Prepare input
        images = [image]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Apply chat template
        formatted_prompt = apply_chat_template(
            self.processor, self.config, messages, num_images=len(images)
        )

        # Generate output
        output = generate(
            self.model, self.processor, formatted_prompt, images, verbose=False
        )
        print(output)

        response = output[0]

        print("VLM", prompt, " -> ", response)
        return response

    def vlm_multi(
        self, images: list[Image.Image], prompt: str, system_prompt: str
    ) -> str:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        if self.model is None:
            self.model, self.processor = load(self.model_name())
            self.config = load_config(self.model_name())

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Apply chat template
        formatted_prompt = apply_chat_template(
            self.processor, self.config, messages, num_images=len(images)
        )

        # Generate output
        output = generate(
            self.model, self.processor, formatted_prompt, images, verbose=False
        )
        print(output)

        response = output[0]

        print("VLM", prompt, " -> ", response)
        return response


if __name__ == "__main__":
    vlm = QwenMlxVisionLanguageModel("test_ph3")
    image = Image.open("test.jpg")
    csv = vlm.vlm(
        image,
        prompt="<|image_1|>Given the screenshot, give 3-5 specific, step-by-step instructions on where Mario should go next to win the game.",
        system_prompt="You are an expert Super Mario Bros player.",
    )
    # csv = vlm.vlm(image, prompt="<|image_1|>Output all cells in this table.", system_prompt="Output each cell of the table as it's own line with the format: Row Name, Column Name, Cell Value.")
    print(csv)
