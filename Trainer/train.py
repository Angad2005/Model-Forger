"""Export a trained Qwen3.5-2B LoRA adapter as a standalone, named model.

Default input:
  modelManager/outputs/qwen35-2b-stage2/final_adapter

Default output:
  modelManager/models/AngadAI-Qwen35-2B-CreativeVision-v1

Examples (Linux / WSL):
  python package_qwen35_adapter.py
  ADAPTER_DIR="modelManager/outputs/my-run/final_adapter" python package_qwen35_adapter.py
  CUSTOM_MODEL_NAME="AngadAI-MyAssistant-v2" python package_qwen35_adapter.py
  EXPORT_DIR="/mnt/models/MyAssistant" python package_qwen35_adapter.py

The output is a merged 16-bit Hugging Face model, not just the LoRA adapter.
It still requires an inference runtime (Transformers, vLLM, etc.) to run.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from unsloth import FastVisionModel

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_ADAPTER_DIR = (
    ROOT_DIR / "modelManager" / "outputs" / "qwen35-2b-stage2" / "final_adapter"
)
DEFAULT_MODEL_NAME = "AngadAI-Qwen35-2B-CreativeVision-v1"
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "1024"))


def resolve_path(raw: str | Path) -> Path:
    """Resolve relative paths against the project folder, not the shell CWD."""
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (ROOT_DIR / path).resolve()


def main() -> None:
    adapter_dir = resolve_path(os.getenv("ADAPTER_DIR", str(DEFAULT_ADAPTER_DIR)))
    custom_name = os.getenv("CUSTOM_MODEL_NAME", DEFAULT_MODEL_NAME).strip()

    if not custom_name:
        raise ValueError("CUSTOM_MODEL_NAME cannot be empty.")
    # Keep the default output folder name portable. EXPORT_DIR can override it.
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", custom_name).strip(".-")
    if not safe_name:
        raise ValueError("CUSTOM_MODEL_NAME needs at least one letter or number.")

    export_dir_raw = os.getenv("EXPORT_DIR", "").strip()
    export_dir = (
        resolve_path(export_dir_raw)
        if export_dir_raw
        else ROOT_DIR / "modelManager" / "models" / safe_name
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was not detected. Run this script in your CUDA-enabled WSL/Python "
            "environment with PyTorch and Unsloth installed."
        )
    if not adapter_dir.is_dir():
        raise FileNotFoundError(
            f"Adapter directory not found: {adapter_dir}\n"
            "Run training first, or set ADAPTER_DIR to the folder containing "
            "adapter_config.json and adapter weights."
        )
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No adapter_config.json in {adapter_dir}. Point ADAPTER_DIR to the "
            "final_adapter folder, not the parent run folder."
        )
    if export_dir == adapter_dir or adapter_dir in export_dir.parents:
        raise ValueError("The export folder must not be the adapter folder or inside it.")
    if export_dir.exists():
        raise FileExistsError(
            f"Output already exists: {export_dir}\n"
            "Choose a new CUSTOM_MODEL_NAME / EXPORT_DIR so this script never "
            "overwrites an earlier model."
        )

    export_dir.parent.mkdir(parents=True, exist_ok=True)
    use_bf16 = bool(torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GiB")
    print(f"Loading trained adapter: {adapter_dir}")
    print(f"Custom model name: {custom_name}")
    print(f"Export destination: {export_dir}")
    print(f"Loading dtype: {'bfloat16' if use_bf16 else 'float16'}")
    print("Loading the adapter together with its base model...")

    # Loading the adapter directory lets Unsloth read adapter_config.json and
    # locate the base model. Keep load_in_4bit=False for a conventional 16-bit merge.
    model, processor = FastVisionModel.from_pretrained(
        model_name=str(adapter_dir),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=dtype,
        load_in_4bit=False,
    )
    FastVisionModel.for_inference(model)

    print("Merging LoRA weights and saving a standalone 16-bit model...")
    # Lower maximum_memory_usage can reduce peak GPU memory usage during export.
    # If export runs out of VRAM, try MAX_MEMORY_USAGE=0.45 or export on a larger GPU.
    memory_usage = float(os.getenv("MAX_MEMORY_USAGE", "0.55"))
    if not 0.1 <= memory_usage <= 0.95:
        raise ValueError("MAX_MEMORY_USAGE must be between 0.1 and 0.95.")

    model.save_pretrained_merged(
        str(export_dir),
        processor,
        save_method="merged_16bit",
        maximum_memory_usage=memory_usage,
    )
    # Save processor/tokenizer explicitly as well, so local inference has the
    # vocabulary and multimodal preprocessing configuration alongside the weights.
    processor.save_pretrained(str(export_dir))

    readme = f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B-Base
tags:
- qwen3.5
- lora
- merged-model
- text-generation
- image-text-to-text
---

# {custom_name}

A locally fine-tuned Qwen3.5-2B model exported as standalone 16-bit weights.

- **Base model:** `Qwen/Qwen3.5-2B-Base`
- **Input adapter:** `{adapter_dir}`
- **Export format:** merged 16-bit Hugging Face Transformers/Unsloth model
- **Export time (UTC):** `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}`

This model is derived from Qwen3.5-2B-Base. Review the base model card and license
before redistributing it. Tool calling also requires an application/runtime that
executes the declared tools; exporting the model does not bundle those tools.
"""
    (export_dir / "README.md").write_text(readme, encoding="utf-8")

    # Verify that key model/config files were created before reporting success.
    expected = ["config.json"]
    missing = [name for name in expected if not (export_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Export ran, but expected files are missing: {missing}. "
            f"Inspect the output directory: {export_dir}"
        )

    print("\nPackaging complete.")
    print(f"Your named model is at: {export_dir}")
    print("It is a standalone merged model; you no longer need the adapter folder for inference.")
    print("Keep the adapter and training checkpoints as backups for future fine-tuning.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPackaging interrupted. Existing source adapter was not modified.", file=sys.stderr)
        raise
