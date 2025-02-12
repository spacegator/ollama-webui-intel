import os
import time
import base64
import statistics

import requests
from utils import format_results_as_markdown_table

# Configuration
IMAGE_DIR = './data/images'
CONTEXT_LEN = 8192  # Tune, based on your needs and hardware setup
NUM_THREAD = 12  # Tune, based on your setup

API_URI: str = "http://127.0.0.1:11434/api/generate"
# List the models to benchmark. Make sure the models are pulled before calling.
MODELS_TO_TEST: list[str] = [
    "deepseek-r1:70b",
    "llama3.3:70b",
    "llama3.1:70b",
    "qwen2.5:72b",
    "qwen2.5:32b",
    "qwen2.5:7b",
    "qwq",
    "mistral-small:24b",
    "phi4:14b",
    "phi3.5:3.8b",
    "llama3.2:3b",
    "smallthinker:3b",
    "smollm2:1.7b",
    "smollm2:360m",
    "opencoder:1.5b",
    "llama3.1:8b"
]
PROMPTS: list[str] = [
    "Why the sky is blue?",
    "Explain the theory of relativity in simple terms.",
    "Is 3307 a prime number?",
    "Implement a Python function to find N-th prime number in optimal way."
]

VISION_MODELS_TO_TEST: list[str] = [
    "llama3.2-vision:90b",
    "llama3.2-vision:11b",
    "minicpm-v:8b",
    "llava-phi3:3.8b",
    "moondream:1.8b"
]

VISION_PROMPTS: list[str] = [
    "What is depicted in the image?",
]


def encode_image_to_base64(image_path: str) -> str:
    """Encodes an image to base64.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def test_inference_speed(api_uri: str, model: str, prompts: list[str],
                         images: list[str] | None = None)\
        -> list[float]:
    """Tests the inference speed for a given model with a list of prompts.

    Args:
        api_uri (str): The URI of the Ollama API.
        model (str): The model to test.
        prompts (list[str]): A list of prompts to test the model on.
        images (list[str] | None): Base64 encoded images. If provided, it is
            assumed the model is vision-based. Defaults to None.

    Returns:
        list[float]: A list of tokens per second for each prompt.
    """
    results: list[float] = []

    input_images = images if images is not None else [None]

    for prompt in prompts:
        for image in input_images:
            payload: dict[str, any] = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_thread": NUM_THREAD,
                    "num_ctx": CONTEXT_LEN
                }
            }
            if image is not None:
                payload["images"] = [image]

            start_time: float = time.perf_counter()
            response: requests.Response = requests.post(api_uri, json=payload)
            end_time: float = time.perf_counter()

            if response.status_code == 200:
                data: dict[str, any] = response.json()
                eval_count: int = data.get("eval_count", 0)
                eval_duration: int = data.get(
                    "eval_duration", 1)  # Avoid zero div
                tokens_per_second: float = (eval_count / eval_duration) * 1e9
                results.append(tokens_per_second)

                print(f"Model: {model}, Prompt: '{prompt[:32]}...', "
                      f"Response: '{data.get('response', '')[:32]}...',\n"
                      f"Tokens/s: {tokens_per_second: .2f}, "
                      f"Response time: {end_time-start_time:.2f} s")
            else:
                print(
                    f"Error for model: {model} with prompt: "
                    f"'{prompt[:32]}...'. Status code: {response.status_code}")

    return results


def test_models(model_to_test: list[str], prompts: list[str],
                images: list[str] | None = None):
    all_results: dict[str, dict[str, float]] = {}

    for model in model_to_test:
        print(f"\nTesting model: {model}, Context {CONTEXT_LEN}")
        tokens_per_second_list: list[float] = test_inference_speed(
            API_URI, model, prompts, images)

        if tokens_per_second_list:
            mean_tokens_per_second: float = statistics.mean(
                tokens_per_second_list)
            std_dev_tokens_per_second: float = (
                statistics.stdev(tokens_per_second_list) if len(
                    tokens_per_second_list) > 1 else 0
            )

            all_results[model] = {
                "mean_tokens_per_second": mean_tokens_per_second,
                "std_dev_tokens_per_second": std_dev_tokens_per_second
            }

            print(f"\nSummary for model: {model}")
            print(f"Mean Tokens/s: {mean_tokens_per_second:.2f}")
            print(f"STD Tokens/s: {std_dev_tokens_per_second:.2f}\n")
        else:
            print(f"No results for model: {model}")

    # Format and print the results as a Markdown table
    markdown_table = format_results_as_markdown_table(all_results)
    print("### Test Results\n")
    print(markdown_table)


def main() -> None:
    # Test language models
    test_models(MODELS_TO_TEST, PROMPTS)

    # Test VLMs
    image_paths = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = [encode_image_to_base64(image) for image in image_paths]
    test_models(VISION_MODELS_TO_TEST, VISION_PROMPTS, images)


if __name__ == "__main__":
    main()
