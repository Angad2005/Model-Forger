"""Fine-tune the local Qwen3.5-2B-Base model with Unsloth LoRA.

Project root (all relative paths are resolved from here):

  /home/angad/Model-Forger

Default inputs:

  Base model:       modelManager/models/Qwen3.5-2B-Base

  Training data:    Dataset/train_split.jsonl

  Validation data:  Dataset/eval.jsonl (optional; missing file disables evaluation)

  Images:           Dataset/pics

Default outputs:

  Trainer checkpoints and metrics: modelManager/outputs/qwen35-2b-stage2/

  Final LoRA adapter:               modelManager/outputs/qwen35-2b-stage2/final_adapter/

  Merged-model export destination:   modelManager/Exported/AngflowV0.1/

Accepted JSONL records use a non-empty "messages" list. Image examples can use

local image paths in message content or top-level "image"/"images" fields.

Tool definitions supplied in a top-level "tools" list are added to the system

prompt in the Qwen-style tool format used by this script.

Useful environment overrides:

  PROJECT_ROOT=/home/angad/Model-Forger

  BASE_MODEL=modelManager/models/Qwen3.5-2B-Base

  TRAIN_FILE=Dataset/train_split.jsonl

  VALIDATION_FILE=Dataset/eval.jsonl

  IMAGE_DIR=Dataset/pics

  OUTPUT_DIR=modelManager/outputs/qwen35-2b-stage2

  FINAL_ADAPTER_DIR=modelManager/outputs/qwen35-2b-stage2/final_adapter

  START_ADAPTER_DIR=none

  RESUME_FROM_CHECKPOINT=latest  # also supports "none" or a checkpoint path

  MERGED_MODEL_DIR=modelManager/Exported/AngflowV0.1

This script saves a LoRA adapter; it does not merge the base model. Run your

separate packaging script to create the merged model in MERGED_MODEL_DIR.

Existing final-adapter output paths are never overwritten.

"""

from __future__ import annotations

import json

import math

import os

import sys

from pathlib import Path

from typing import Any

# Apply Unsloth patches before importing Transformers and TRL.

import unsloth

import torch

from PIL import Image

from transformers.trainer_utils import get_last_checkpoint

from trl import SFTConfig, SFTTrainer

from unsloth import FastVisionModel

from unsloth.trainer import UnslothVisionDataCollator

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/home/angad/Model-Forger")).expanduser().resolve()

DEFAULT_MODEL = str(PROJECT_ROOT / "modelManager" / "models" / "Qwen3.5-2B-Base")

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "modelManager" / "outputs" / "qwen35-2b-stage2"

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "512"))

NUM_TRAIN_EPOCHS = float(os.getenv("NUM_TRAIN_EPOCHS", "3"))

LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))

SAVE_STEPS = int(os.getenv("SAVE_STEPS", "25"))

SAVE_TOTAL_LIMIT = int(os.getenv("SAVE_TOTAL_LIMIT", "3"))

GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8"))

LORA_R = int(os.getenv("LORA_R", "8"))

LORA_ALPHA = int(os.getenv("LORA_ALPHA", str(LORA_R)))

SEED = int(os.getenv("SEED", "3407"))

TOOL_INSTRUCTIONS = (

    "\n\nIf you choose to call a function, reply using this format and do not add a suffix:\n\n"

    "<tool_call>\n<function=example_function_name>\n"

    "<parameter=example_parameter_name>\nvalue\n</parameter>\n"

    "</function>\n</tool_call>\n\n"

    "Function calls must use a <function=...> block nested inside <tool_call> tags. "

    "Provide required parameters. If no function call is needed, answer normally."

)

def resolve_path(raw: str | Path, *, base: Path = PROJECT_ROOT) -> Path:

    """Resolve relative paths consistently, independent of the shell directory."""

    path = Path(raw).expanduser()

    return path.resolve() if path.is_absolute() else (base / path).resolve()

def resolve_model_reference(raw: str) -> str:

    """Resolve local model folders but preserve Hugging Face IDs such as org/model."""

    raw = raw.strip()

    if not raw:

        raise ValueError("BASE_MODEL cannot be empty.")

    candidate = Path(raw).expanduser()

    looks_like_path = (

        candidate.is_absolute()

        or raw.startswith(("./", "../", "~/", "modelManager/"))

        or resolve_path(raw).exists()

    )

    if not looks_like_path:

        return raw

    local_model_path = resolve_path(raw)

    if not local_model_path.is_dir():

        raise FileNotFoundError(f"Local BASE_MODEL directory not found: {local_model_path}")

    return str(local_model_path)

def read_jsonl(path: Path) -> list[dict[str, Any]]:

    """Read and validate JSONL rows, reporting the exact bad line if necessary."""

    if not path.is_file():

        raise FileNotFoundError(f"Dataset file not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():

                continue

            try:

                row = json.loads(line)

            except json.JSONDecodeError as exc:

                raise ValueError(

                    f"Invalid JSON in {path}, line {line_number}: {exc}"

                ) from exc

            if not isinstance(row, dict):

                raise ValueError(f"Each JSONL row must be an object: {path}:{line_number}")

            if not isinstance(row.get("messages"), list) or not row["messages"]:

                raise ValueError(

                    f"Each row needs a non-empty 'messages' list: {path}:{line_number}"

                )

            rows.append(row)

    if not rows:

        raise ValueError(f"No data records were found in {path}")

    return rows

def resolve_image_path(raw_path: str, *, data_file: Path, image_dir: Path) -> Path:

    """Find a local image path in likely project/data locations."""

    raw_path = raw_path.strip()

    if raw_path.startswith("file://"):

        raw_path = raw_path[7:]

    candidate_path = Path(raw_path).expanduser()

    candidates: list[Path] = []

    if candidate_path.is_absolute():

        candidates.append(candidate_path)

    else:

        candidates.extend(

            [

                PROJECT_ROOT / candidate_path,

                data_file.parent / candidate_path,

                image_dir / candidate_path,

                PROJECT_ROOT / "Dataset" / candidate_path,

                PROJECT_ROOT / "datasets" / candidate_path,

            ]

        )

    for candidate in candidates:

        candidate = candidate.resolve()

        if candidate.is_file():

            return candidate

    tried = "\n  - ".join(str(candidate.resolve()) for candidate in candidates)

    raise FileNotFoundError(

        f"Could not locate image file {raw_path!r}. Checked:\n  - {tried}\n"

        "Set IMAGE_DIR to the directory containing your local images."

    )

MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "512"))


def load_local_image(raw_image: Any, *, data_file: Path, image_dir: Path) -> Image.Image:

    """Load an image path into an RGB PIL image for the Unsloth vision collator."""

    if isinstance(raw_image, Image.Image):

        image = raw_image.convert("RGB")
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
        return image

    if not isinstance(raw_image, str) or not raw_image.strip():

        raise ValueError(f"Image references must be non-empty local paths; got {raw_image!r}")

    if raw_image.startswith(("http://", "https://", "data:")):

        raise ValueError(

            f"Remote/data URL image references are not loaded automatically: {raw_image!r}. "

            "Download the image locally and put its path in the JSONL file."

        )

    image_path = resolve_image_path(raw_image, data_file=data_file, image_dir=image_dir)

    try:

        with Image.open(image_path) as image:

            # Bound RAM use and the number of visual tokens generated by large images.
            # Thumbnail modifies the opened image in place while preserving aspect ratio.
            image = image.convert("RGB")
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
            return image.copy()

    except Exception as exc:

        raise ValueError(f"Unable to read image {image_path}: {exc}") from exc

def normalise_content(

    content: Any, *, data_file: Path, image_dir: Path

) -> str | list[dict[str, Any]] | None:

    """Convert JSON-serializable message content into the expected chat format."""

    if content is None or isinstance(content, str):

        return content

    if not isinstance(content, list):

        raise ValueError(

            "Message content must be a string, null, or a list of text/image parts; "

            f"received {type(content).__name__}."

        )

    parts: list[dict[str, Any]] = []

    for part_index, part in enumerate(content):

        if not isinstance(part, dict):

            raise ValueError(f"Content part {part_index} must be a JSON object.")

        part = dict(part)

        part_type = part.get("type")

        # Canonical Unsloth format: {"type": "image", "image": "path/to/file.jpg"}.

        if part_type == "image" or "image" in part:

            image_ref = part.get("image")

            if image_ref is None:

                image_ref = part.get("path") or part.get("image_path")

            if image_ref is None:

                raise ValueError(

                    "An image content part needs an 'image' path (or 'path'/'image_path')."

                )

            parts.append(

                {"type": "image", "image": load_local_image(

                    image_ref, data_file=data_file, image_dir=image_dir

                )}

            )

            continue

        # OpenAI-style image_url fields are accepted if their URL is a local path/file URI.

        if part_type == "image_url" or "image_url" in part:

            image_url = part.get("image_url")

            if isinstance(image_url, dict):

                image_ref = image_url.get("url")

            else:

                image_ref = image_url

            if not image_ref:

                raise ValueError("An image_url content part needs image_url.url or a path.")

            parts.append(

                {"type": "image", "image": load_local_image(

                    image_ref, data_file=data_file, image_dir=image_dir

                )}

            )

            continue

        if part_type == "text" or "text" in part:

            if "text" not in part:

                raise ValueError("A text content part needs a 'text' field.")

            parts.append({"type": "text", "text": str(part["text"])})

            continue

        raise ValueError(

            f"Unsupported content part in {data_file}: {part}. "

            "This script currently handles text and still images."

        )

    return parts

def make_tools_prompt(tools: list[Any]) -> str:

    """Build a Qwen3.5-style tool specification to preserve tools in SFT examples."""

    serialised_tools = "\n".join(

        json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in tools

    )

    return (

        "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"

        f"{serialised_tools}\n</tools>"

        f"{TOOL_INSTRUCTIONS}"

    )

def normalise_row(

    row: dict[str, Any], *, data_file: Path, image_dir: Path

) -> dict[str, Any]:

    """Normalise one row and keep only the messages column for SFTTrainer."""

    messages: list[dict[str, Any]] = []

    for index, original_message in enumerate(row["messages"]):

        if not isinstance(original_message, dict):

            raise ValueError(f"Message {index} in {data_file} must be a JSON object.")

        message = dict(original_message)

        role = message.get("role")

        if role == "developer":

            # Qwen's template supports system/user/assistant/tool, not developer.

            message["role"] = "system"

            role = "system"

        if role not in {"system", "user", "assistant", "tool"}:

            raise ValueError(

                f"Unsupported role {role!r} in {data_file}; expected system/user/assistant/tool."

            )

        message["content"] = normalise_content(

            message.get("content", ""), data_file=data_file, image_dir=image_dir

        )

        messages.append(message)

    # Some image datasets store the path at row-level rather than in message content.

    top_level_images: list[Any] = []

    if row.get("image") is not None:

        top_level_images.append(row["image"])

    if row.get("images") is not None:

        images_value = row["images"]

        top_level_images.extend(images_value if isinstance(images_value, list) else [images_value])

    if top_level_images:

        user_indices = [i for i, message in enumerate(messages) if message.get("role") == "user"]

        if not user_indices:

            raise ValueError(f"Row with top-level image(s) in {data_file} has no user message.")

        user_index = user_indices[-1]

        user_content = messages[user_index].get("content", "")

        if isinstance(user_content, str):

            user_content = [{"type": "text", "text": user_content}]

        elif user_content is None:

            user_content = []

        elif not isinstance(user_content, list):

            raise ValueError("Cannot append images to this user message's content format.")

        already_has_image = any(part.get("type") == "image" for part in user_content)

        if not already_has_image:

            for image_ref in top_level_images:

                user_content.append(

                    {"type": "image", "image": load_local_image(

                        image_ref, data_file=data_file, image_dir=image_dir

                    )}

                )

            messages[user_index]["content"] = user_content

    tools = row.get("tools") or []

    if tools:

        if not isinstance(tools, list):

            raise ValueError(f"The 'tools' field must be a list in {data_file}.")

        tool_prompt = make_tools_prompt(tools)

        system_indices = [i for i, message in enumerate(messages) if message.get("role") == "system"]

        if system_indices:

            first_system = messages[system_indices[0]]

            old_content = first_system.get("content") or ""

            if isinstance(old_content, str):

                if "<tools>" not in old_content:

                    first_system["content"] = f"{old_content.strip()}\n\n{tool_prompt}".strip()

            else:

                raise ValueError("A system message containing tool definitions must have text content.")

        else:

            messages.insert(0, {"role": "system", "content": tool_prompt})

    return {"messages": messages}

def load_dataset_from_jsonl(path: Path, *, image_dir: Path) -> list[dict[str, Any]]:

    """Read JSONL into a Python list; this safely supports mixed text/image content."""

    raw_rows = read_jsonl(path)

    rows = [normalise_row(row, data_file=path, image_dir=image_dir) for row in raw_rows]

    image_count = sum(

        1

        for row in rows

        for message in row["messages"]

        for part in (message.get("content") if isinstance(message.get("content"), list) else [])

        if isinstance(part, dict) and part.get("type") == "image"

    )

    print(f"Loaded {len(rows):,} examples from {path} ({image_count:,} image(s)).")

    return rows

def resolve_resume_checkpoint(output_dir: Path) -> str | None:

    """Resolve RESUME_FROM_CHECKPOINT=latest/none/or a specific checkpoint path."""

    setting = os.getenv("RESUME_FROM_CHECKPOINT", "latest").strip()

    if setting.lower() in {"none", "false", "0", "no", "off"}:

        return None

    if setting.lower() == "latest":

        checkpoint = get_last_checkpoint(str(output_dir)) if output_dir.is_dir() else None

        if checkpoint:

            print(f"Resuming Trainer state from latest checkpoint: {checkpoint}")

        else:

            print("No Trainer checkpoint found; starting a new run from the selected model.")

        return checkpoint

    checkpoint_path = resolve_path(setting)

    if not checkpoint_path.is_dir():

        raise FileNotFoundError(f"Requested checkpoint directory does not exist: {checkpoint_path}")

    return str(checkpoint_path)

def main() -> None:

    if not torch.cuda.is_available():

        raise RuntimeError(

            "CUDA was not detected. Activate your CUDA-enabled WSL/Python environment "

            "with PyTorch and Unsloth installed."

        )

    if MAX_SEQ_LENGTH < 128:

        raise ValueError("MAX_SEQ_LENGTH should be at least 128.")

    if NUM_TRAIN_EPOCHS <= 0:

        raise ValueError("NUM_TRAIN_EPOCHS must be greater than zero.")

    if LEARNING_RATE <= 0:

        raise ValueError("LEARNING_RATE must be greater than zero.")

    if LORA_R < 1 or LORA_ALPHA < 1:

        raise ValueError("LORA_R and LORA_ALPHA must be >= 1.")

    if SAVE_STEPS < 1 or SAVE_TOTAL_LIMIT < 1 or GRADIENT_ACCUMULATION_STEPS < 1:

        raise ValueError(

            "SAVE_STEPS, SAVE_TOTAL_LIMIT and GRADIENT_ACCUMULATION_STEPS must be >= 1."

        )

    # Every relative input/output path is rooted at PROJECT_ROOT, never Trainer/ or cwd.

    train_file = resolve_path(os.getenv("TRAIN_FILE", "Dataset/train_split.jsonl"))

    validation_raw = os.getenv("VALIDATION_FILE", "Dataset/eval.jsonl").strip()

    if validation_raw.lower() in {"none", "false", "off", "0"}:

        validation_file = None

    else:

        validation_file = resolve_path(validation_raw) if validation_raw else None

    output_dir = resolve_path(os.getenv("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))

    image_dir = resolve_path(os.getenv("IMAGE_DIR", "Dataset/pics"))

    final_adapter_dir = resolve_path(

        os.getenv("FINAL_ADAPTER_DIR", str(output_dir / "final_adapter"))

    )

    merged_model_dir = resolve_path(

        os.getenv("MERGED_MODEL_DIR", "modelManager/Exported/AngflowV0.1")

    )

    start_adapter_raw = os.getenv("START_ADAPTER_DIR", "").strip()

    if start_adapter_raw.lower() in {"", "none", "false", "off", "0"}:

        start_adapter_dir = None

    else:

        start_adapter_dir = resolve_path(start_adapter_raw)

    if not train_file.is_file():

        raise FileNotFoundError(f"Training JSONL file not found: {train_file}")

    if validation_file is not None and not validation_file.is_file():

        print(

            f"WARNING: validation file not found ({validation_file}); evaluation is disabled."

        )

        validation_file = None

    if output_dir.exists() and not output_dir.is_dir():

        raise NotADirectoryError(f"OUTPUT_DIR exists but is not a directory: {output_dir}")

    if final_adapter_dir == output_dir:

        raise ValueError("FINAL_ADAPTER_DIR must differ from OUTPUT_DIR.")

    # Never write into an existing adapter directory, even if resuming a checkpoint.

    if final_adapter_dir.exists():

        raise FileExistsError(

            f"Final adapter destination already exists: {final_adapter_dir}\n"

            "To protect existing adapter files, move/rename that directory or choose a new "

            "FINAL_ADAPTER_DIR before training. This script will not overwrite it."

        )

    if start_adapter_dir is not None:

        if not (start_adapter_dir / "adapter_config.json").is_file():

            raise FileNotFoundError(

                f"START_ADAPTER_DIR must contain adapter_config.json: {start_adapter_dir}"

            )

        if start_adapter_dir == final_adapter_dir or start_adapter_dir in final_adapter_dir.parents:

            raise ValueError(

                "FINAL_ADAPTER_DIR must not be the starting adapter directory or a child "

                "of it; the source adapter must remain untouched."

            )

    train_dataset = load_dataset_from_jsonl(train_file, image_dir=image_dir)

    eval_dataset = (

        load_dataset_from_jsonl(validation_file, image_dir=image_dir)

        if validation_file is not None

        else None

    )

    # The checkpoint folder may already exist when resuming a previous run.

    output_dir.mkdir(parents=True, exist_ok=True)

    resume_checkpoint = resolve_resume_checkpoint(output_dir)

    base_model = (

        str(start_adapter_dir)

        if start_adapter_dir is not None

        else resolve_model_reference(os.getenv("BASE_MODEL", DEFAULT_MODEL))

    )

    use_bf16 = bool(torch.cuda.is_bf16_supported())

    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("\n=== Training configuration ===")

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GiB")

    print(f"Base model / starting adapter: {base_model}")

    print(f"Sequence length: {MAX_SEQ_LENGTH}")
    print(f"Image max side: {MAX_IMAGE_SIDE}px")
    print(f"Fine-tune vision layers: {os.getenv('FINETUNE_VISION_LAYERS', '0')}")

    print(f"LoRA rank / alpha: {LORA_R} / {LORA_ALPHA}")

    print(f"Epochs: {NUM_TRAIN_EPOCHS}; learning rate: {LEARNING_RATE}")

    print(f"Output/checkpoints: {output_dir}")

    print(f"Final adapter (must not already exist): {final_adapter_dir}")

    print(f"Merged-model export destination (packaging script): {merged_model_dir}")

    print(f"Precision: {'bfloat16' if use_bf16 else 'float16'}")

    print("Loading model; the first run may take longer while kernels initialise...\n")

    # Qwen3.5 2B BF16/FP16 LoRA is preferred over QLoRA for model quality.
    model, processor = FastVisionModel.from_pretrained(

        model_name=str(base_model),

        max_seq_length=MAX_SEQ_LENGTH,

        dtype=dtype,

        load_in_4bit=False,

    )

    # A supplied adapter already has LoRA layers. For a fresh base model, attach LoRA.

    if start_adapter_dir is None:

        model = FastVisionModel.get_peft_model(

            model,

            finetune_vision_layers=(
                os.getenv("FINETUNE_VISION_LAYERS", "0").strip().lower()
                in {"1", "true", "yes", "on"}
            ),

            finetune_language_layers=True,

            finetune_attention_modules=True,

            finetune_mlp_modules=True,

            r=LORA_R,

            lora_alpha=LORA_ALPHA,

            lora_dropout=0,

            bias="none",

            target_modules="all-linear",

            use_gradient_checkpointing="unsloth",

            random_state=SEED,

        )

    FastVisionModel.for_training(model)

    # Qwen3.5's template uses these message boundaries; masking the prompt means

    # training loss is calculated on assistant responses/tool calls, not user text.

    data_collator = UnslothVisionDataCollator(

        model,

        processor,

        max_seq_length=MAX_SEQ_LENGTH,

        resize="min",

        train_on_responses_only=True,

        instruction_part="<|im_start|>user\n",

        response_part="<|im_start|>assistant\n",

        completion_only_loss=True,

    )

    # Recent TRL SFTConfig versions may not expose warmup_ratio. Convert the

    # previous 3% warmup ratio into an explicit warmup_steps value instead.

    # WORLD_SIZE defaults to 1 for the single-GPU setup and supports torchrun.

    per_device_train_batch_size = 1

    world_size = max(1, int(os.getenv("WORLD_SIZE", "1")))

    updates_per_epoch = math.ceil(

        len(train_dataset)

        / (per_device_train_batch_size * GRADIENT_ACCUMULATION_STEPS * world_size)

    )

    estimated_total_steps = max(1, math.ceil(updates_per_epoch * NUM_TRAIN_EPOCHS))

    warmup_steps = max(1, math.ceil(estimated_total_steps * 0.03))

    print(

        f"Estimated optimizer steps: {estimated_total_steps}; "

        f"warmup steps (3%): {warmup_steps}"

    )

    training_args = SFTConfig(

        output_dir=str(output_dir),

        num_train_epochs=NUM_TRAIN_EPOCHS,

        per_device_train_batch_size=per_device_train_batch_size,

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

        learning_rate=LEARNING_RATE,

        warmup_steps=warmup_steps,

        lr_scheduler_type="cosine",

        optim="adamw_8bit",

        weight_decay=0.01,

        max_grad_norm=1.0,

        logging_steps=5,

        save_strategy="steps",

        save_steps=SAVE_STEPS,

        save_total_limit=SAVE_TOTAL_LIMIT,

        save_only_model=False,

        eval_strategy="steps" if eval_dataset is not None else "no",

        eval_steps=SAVE_STEPS if eval_dataset is not None else None,

        load_best_model_at_end=eval_dataset is not None,

        metric_for_best_model="eval_loss" if eval_dataset is not None else None,

        greater_is_better=False if eval_dataset is not None else None,

        bf16=use_bf16,

        fp16=not use_bf16,

        gradient_checkpointing=True,

        remove_unused_columns=False,

        dataset_text_field="",

        dataset_kwargs={"skip_prepare_dataset": True},

        max_length=MAX_SEQ_LENGTH,

        packing=False,

        seed=SEED,

        report_to="none",

        dataloader_num_workers=0,

    )

    trainer = SFTTrainer(

        model=model,

        processing_class=processor,

        data_collator=data_collator,

        train_dataset=train_dataset,

        eval_dataset=eval_dataset,

        args=training_args,

    )

    print("\nStarting fine-tuning. Periodic checkpoints include optimizer/scheduler state for resuming.")

    try:

        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    except KeyboardInterrupt:

        print(

            "\nTraining interrupted. Resume by running this script again with "

            "RESUME_FROM_CHECKPOINT=latest; the most recent periodic checkpoint will be used "

            "if one was written.",

            file=sys.stderr,

        )

        raise

    # Save the LoRA adapter (not merged base weights) where the packaging script expects it.

    final_adapter_dir.parent.mkdir(parents=True, exist_ok=True)

    # Extra protection if the destination appeared while training was running.

    final_adapter_dir.mkdir(parents=True, exist_ok=False)

    trainer.model.save_pretrained(str(final_adapter_dir))

    processor.save_pretrained(str(final_adapter_dir))

    trainer.save_state()

    metrics = dict(train_result.metrics)

    if eval_dataset is not None:

        try:

            metrics.update(trainer.evaluate())

        except Exception as exc:

            print(f"WARNING: final evaluation metric collection failed: {exc}", file=sys.stderr)

    metrics["train_examples"] = len(train_dataset)

    metrics["validation_examples"] = len(eval_dataset) if eval_dataset is not None else 0

    metrics["final_adapter_dir"] = str(final_adapter_dir)

    (output_dir / "training_metrics.json").write_text(

        json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",

        encoding="utf-8",

    )

    if not (final_adapter_dir / "adapter_config.json").is_file():

        raise RuntimeError(f"Training ended but adapter_config.json is missing from {final_adapter_dir}")

    print("\n=== Fine-tuning complete ===")

    print(f"Final LoRA adapter: {final_adapter_dir}")

    print(f"Trainer checkpoints: {output_dir}")

    print(f"Training metrics: {output_dir / 'training_metrics.json'}")

    print(f"Next step: set ADAPTER_DIR={final_adapter_dir} in your packaging script.")

    print(f"The packaging script should export the merged model to: {merged_model_dir}")

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        sys.exit(130)
