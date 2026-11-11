import ast
import base64
import io
import re
import time

from datasets import load_dataset
from openai import OpenAI


MODEL = "Qwen/Qwen3-VL-4B-Instruct"

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)


def image_to_data_url(image):
    image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)

    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


# --------------------------------------------------
# 1. Load MMMU Art validation
# --------------------------------------------------

print("Loading MMMU...")

dataset = load_dataset(
    "MMMU/MMMU",
    "Art",
    split="validation",
)

sample = next(
    x for x in dataset
    if x["id"] == "validation_Art_8"
)

print("ID:", sample["id"])
print("GT:", sample["answer"])


# --------------------------------------------------
# 2. Parse options
# --------------------------------------------------

options = sample["options"]

if isinstance(options, str):
    options = ast.literal_eval(options)

letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

options_text = "\n".join(
    f"{letters[i]}. {option}"
    for i, option in enumerate(options)
)


# --------------------------------------------------
# 3. Build multimodal content
# --------------------------------------------------

content = []

question = sample["question"]

# <image 1>, <image 2> ... 위치를 따라 이미지 삽입
parts = re.split(r"(<image \d+>)", question)

for part in parts:

    match = re.fullmatch(r"<image (\d+)>", part)

    if match:
        image_num = int(match.group(1))
        image = sample[f"image_{image_num}"]

        content.append({
            "type": "image_url",
            "image_url": {
                "url": image_to_data_url(image)
            },
        })

    elif part:
        content.append({
            "type": "text",
            "text": part,
        })


content.append({
    "type": "text",
    "text": (
        "\nOptions:\n"
        + options_text
        + "\nPlease select the correct answer from the options above."
    ),
})


# --------------------------------------------------
# 4. Inference
# --------------------------------------------------

print("\nRunning inference...")

start = time.perf_counter()

response = client.chat.completions.create(
    model=MODEL,

    messages=[
        {
            "role": "user",
            "content": content,
        }
    ],

    max_tokens=2048,

    temperature=0.7,
    top_p=0.8,

    presence_penalty=1.5,

    seed=3407,

    extra_body={
        "top_k": 20,
        "repetition_penalty": 1.0,
    },
)

elapsed = time.perf_counter() - start

answer = response.choices[0].message.content


# --------------------------------------------------
# 5. Print result
# --------------------------------------------------

print("\n===== QUESTION =====")
print(question)

print("\n===== OPTIONS =====")
print(options_text)

print("\n===== MODEL RESPONSE =====")
print(answer)

print("\n===== GROUND TRUTH =====")
print(sample["answer"])

print("\n===== ELAPSED =====")
print(f"{elapsed:.3f} sec")
