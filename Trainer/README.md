# Qwen3.5-2B-Base SFT with Unsloth

This project trains a **LoRA adapter** for `Qwen/Qwen3.5-2B-Base` using **supervised fine-tuning (SFT)** and Unsloth's vision-aware data collator. It does **not** use 4-bit QLoRA. The model's base weights remain unchanged.

## Why this setup

- Default model: `Qwen/Qwen3.5-2B-Base`
- Method: 16-bit (BF16 when supported, otherwise FP16) LoRA SFT
- Default GPU settings: batch size 1, gradient accumulation 8, sequence length 1024, LoRA rank 8
- Vision encoder fine-tuning is **off by default** to reduce VRAM use on a 6 GB card. The base model remains multimodal; with image examples, its language layers can still be adapted to the visual tasks while the vision encoder is frozen.
- Enable vision-layer updates with `TRAIN_VISION_LAYERS=1` only if the training run fits in memory. This can cause CUDA out-of-memory errors on 6 GB.

Qwen3.5's latest Unsloth documentation recommends Transformers v5 and does not recommend 4-bit QLoRA for this family because of larger quantization differences. Read the current guide: https://unsloth.ai/docs/models/qwen3.5/fine-tune

## Folder layout

```text
qwen35_sft_project/
├── train.py
├── README.md
├── data/
│   ├── train.example.jsonl
│   ├── validation.example.jsonl
│   ├── vision.example.jsonl
│   └── images/
└── outputs/
```

## Install

Use a fresh Python environment where possible, with a compatible PyTorch/CUDA build. Then install/upgrade Unsloth using its current official installation guide:

```bash
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```

If you manage dependencies manually, ensure you have a Qwen3.5-compatible `transformers` v5, plus `trl`, `peft`, `datasets`, `accelerate`, `bitsandbytes`, and `pillow`. Prefer the package versions installed by the current Unsloth setup rather than randomly mixing versions.

Check CUDA before training:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

## Prepare data

1. Copy the text/tool examples into working files:

   ```bash
   cp data/train.example.jsonl data/train.jsonl
   cp data/validation.example.jsonl data/validation.jsonl
   ```

2. Add your own examples. Each **physical line** must be one complete JSON object. Do not put multiple JSON objects on one line.
3. For image examples, copy the row structure from `data/vision.example.jsonl`, replace the placeholder with a real local image path, and replace the placeholder answer with an image-grounded answer. Then add that record as another line to `data/train.jsonl` or `data/validation.jsonl`.
4. By default, image paths are resolved relative to this project folder. Example: `data/images/mars_station.jpg`.
5. Images are opened as RGB and resized so their longest side is at most 448 pixels. Increase `MAX_IMAGE_SIDE` only if VRAM allows it.
6. Keep validation examples separate from training examples. Do not use near-duplicates in both sets.

The script accepts:

- Chat examples: `{"messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}], "tools": []}`
- Tool calls: assistant messages include a `tool_calls` array; the corresponding tool result is a message with `role: "tool"` and `name`; each example also includes JSON-schema objects in `tools`.
- Vision examples: the user message's `content` is a list with text blocks and `{"type":"image","path":"data/images/file.png"}`. The loader converts that path into a PIL image for training.

The `tools` field is a description available to the model during training. Training a tool call does **not** execute the tool. Your application must validate the requested tool name/arguments, execute the approved function, return a `tool` message, and let the model continue.

## Run

```bash
python train.py
```

The adapter is saved to `outputs/qwen35-2b-sft/final_adapter`. This folder contains adapter weights/tokenizer assets and is **not** a standalone full model; keep the base checkpoint.

## Useful environment overrides

```bash
# If image training runs out of memory, start with a smaller sequence length / image size.
MAX_SEQ_LENGTH=768 MAX_IMAGE_SIDE=320 python train.py

# Only after confirming enough VRAM is available for vision-layer updates:
TRAIN_VISION_LAYERS=1 python train.py

# Common adjustments:
MODEL_NAME_OR_PATH=Qwen/Qwen3.5-2B-Base \
NUM_EPOCHS=2 LEARNING_RATE=0.0001 BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=8 LORA_R=8 LORA_ALPHA=16 python train.py

# Resume from a saved Trainer checkpoint:
RESUME_FROM_CHECKPOINT=outputs/qwen35-2b-sft/checkpoints/checkpoint-XXX python train.py
```

If CUDA runs out of memory, reduce `MAX_IMAGE_SIDE`, then `MAX_SEQ_LENGTH`; keep batch size at 1. An image-heavy run may still not fit on a 6 GB GPU. The script saves checkpoints at epoch boundaries, so a hard interrupt in the middle of an epoch may leave no new checkpoint.

## Data-quality guidance

Do not fine-tune only on a handful of examples: the supplied files are format demonstrations. Build a diverse dataset for creative writing, instruction following, explanations, text transformation, visual question answering, and tool-use decisions. Include examples where tools are **not** needed as well as examples where they are needed. Tool-call examples should contain realistic tool results and a grounded final answer.
