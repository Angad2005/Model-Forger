"""Interactively run and manually test a packaged Qwen3.5-2B model.

This script performs inference only. It does NOT load train/validation datasets,
calculate validation loss, or run an automated benchmark.

Default model folder (relative to this script's project directory):
  modelManager/models/AngadAI-Qwen35-2B-CreativeVision-v1

Commands in the chat:
  /image path/to/image.jpg | Ask a question about this image
  /clear
  /exit

Environment overrides:
  MODEL_DIR=/absolute/path/to/merged/model
  MAX_NEW_TOKENS=256
  MAX_MEMORY_GIB=5
  SYSTEM_PROMPT="You are ..."
"""
from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = (
    ROOT_DIR
    / "modelManager"
    / "models"
    / "AngadAI-Qwen35-2B-CreativeVision-v1"
)
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
MAX_MEMORY_GIB = float(os.getenv("MAX_MEMORY_GIB", "5"))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Follow the user's instructions clearly. "
    "For creative writing, use vivid and original language when appropriate.",
    "You can perform multi-turn conversations, answer questions, and describe images.",
    "Text analysis and other advance text processing tasks are also supported."
).strip()


def resolve_model_dir(raw: str) -> Path:
    """Resolve MODEL_DIR relative to this project first, then the current folder."""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    from_project = (ROOT_DIR / path).resolve()
    if from_project.exists():
        return from_project
    return path.resolve()


def get_dtype() -> torch.dtype:
    """Use FP16 for a 6 GB gaming GPU; it is broadly supported on RTX 3050."""
    return torch.float16


def load_model(model_dir: Path) -> tuple[Any, Any]:
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Model folder not found: {model_dir}\n"
            "Package your trained adapter first, or set MODEL_DIR to the folder "
            "containing the merged model's config.json."
        )
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"No config.json found in {model_dir}. MODEL_DIR must point to the "
            "exported model folder, not the parent models directory or final_adapter."
        )

    if not torch.cuda.is_available():
        print("WARNING: CUDA was not detected. Loading may be very slow on CPU.")
        device_map: str | dict[str, Any] = "cpu"
        max_memory = None
    else:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name} ({total_vram:.2f} GiB VRAM)")
        # Reserve some VRAM for the desktop and activations. If necessary,
        # Transformers/Accelerate can place some layers on system RAM.
        budget = min(MAX_MEMORY_GIB, max(1.0, total_vram - 0.8))
        max_memory = {0: f"{budget:.1f}GiB", "cpu": "24GiB"}
        device_map = "auto"
        print(f"GPU memory budget for model weights: {budget:.1f} GiB")

    dtype = get_dtype()
    print(f"Loading model: {model_dir}")
    print("Loading processor/tokenizer...")
    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)

    print("Loading model weights... (first load can take a while)")
    load_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory
    model = AutoModelForMultimodalLM.from_pretrained(str(model_dir), **load_kwargs)
    model.eval()
    print("Model loaded. This script is in inference/testing mode only.\n")
    return model, processor


def make_inputs(
    processor: Any,
    model: Any,
    messages: list[dict[str, Any]],
) -> Any:
    """Apply the model's own chat template and place inputs on its entry device."""
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # For device_map='auto', model.device is normally the device hosting the
    # input embeddings; Accelerate hooks move activations between dispatched layers.
    return inputs.to(model.device)


def generate_reply(
    model: Any,
    processor: Any,
    history: list[dict[str, Any]],
) -> str:
    inputs = make_inputs(processor, model, history)
    input_length = inputs["input_ids"].shape[-1]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "use_cache": True,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.05,
    }

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    answer_tokens = outputs[0][input_length:]
    answer = processor.decode(answer_tokens, skip_special_tokens=True).strip()
    return answer or "(The model returned an empty response.)"


def parse_image_command(command: str) -> tuple[Path, str]:
    """Parse: /image \"path/to/picture.jpg\" | Describe this picture"""
    remainder = command[len("/image") :].strip()
    if "|" in remainder:
        path_part, question = remainder.split("|", 1)
        question = question.strip() or "Describe this image in detail."
    else:
        path_part = remainder
        question = "Describe this image in detail."

    path_tokens = shlex.split(path_part)
    if len(path_tokens) != 1:
        raise ValueError(
            'Image syntax: /image "path/to/image with spaces.jpg" | your question'
        )

    image_path = Path(path_tokens[0]).expanduser()
    if not image_path.is_absolute():
        project_candidate = (ROOT_DIR / image_path).resolve()
        image_path = project_candidate if project_candidate.exists() else image_path.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return image_path, question


def main() -> None:
    model_dir = resolve_model_dir(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR)))
    if MAX_NEW_TOKENS < 1:
        raise ValueError("MAX_NEW_TOKENS must be at least 1.")

    model, processor = load_model(model_dir)
    history: list[dict[str, Any]] = []
    if SYSTEM_PROMPT:
        history.append({"role": "system", "content": SYSTEM_PROMPT})

    print("=" * 68)
    print("AngadAI interactive tester")
    print("Type a prompt and press Enter to chat.")
    print('/image "Dataset/images/example.jpg" | Describe this image')
    print("/clear clears conversation history; /exit quits.")
    print("No validation dataset or validation metrics are used.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "/quit"}:
            break
        if user_text.lower() == "/clear":
            history = []
            if SYSTEM_PROMPT:
                history.append({"role": "system", "content": SYSTEM_PROMPT})
            print("Conversation history cleared.\n")
            continue

        try:
            if user_text.startswith("/image"):
                image_path, question = parse_image_command(user_text)
                user_message = {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": str(image_path)},
                        {"type": "text", "text": question},
                    ],
                }
            else:
                user_message = {"role": "user", "content": user_text}

            history.append(user_message)
            print("AngadAI: ", end="", flush=True)
            response = generate_reply(model, processor, history)
            print(response, "\n")
            history.append({"role": "assistant", "content": response})
        except (FileNotFoundError, ValueError) as exc:
            # Do not keep an invalid image request in the active conversation.
            if history and history[-1].get("role") == "user":
                history.pop()
            print(f"Input error: {exc}\n")
        except torch.cuda.OutOfMemoryError:
            if history and history[-1].get("role") == "user":
                history.pop()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                "\nCUDA ran out of memory. Try a shorter prompt or lower "
                "MAX_NEW_TOKENS (for example 128). For image tests, use a smaller "
                "image. If CPU offload is too slow or still fails, load the model "
                "in 4-bit with a compatible Transformers/bitsandbytes setup.\n"
            )
        except Exception as exc:
            print(f"\nGeneration failed: {type(exc).__name__}: {exc}\n")
            print(
                "Check that Transformers v5 and the matching Qwen3.5 dependencies "
                "are installed, and that the export contains model weights plus "
                "processor/tokenizer files.\n"
            )

    print("Goodbye.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
