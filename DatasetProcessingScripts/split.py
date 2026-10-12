
import json
import random
from pathlib import Path

# Input dataset
input_file = Path("formatted/train.jsonl")

# Output files
train_file = Path("train_split.jsonl")
eval_file = Path("eval.jsonl")

# Split settings
train_ratio = 0.80
random_seed = 42


def main():
    # Read and validate all JSONL records
    with input_file.open("r", encoding="utf-8") as file:
        records = []

        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

    if len(records) < 2:
        raise ValueError("At least 2 records are required.")

    # Shuffle records reproducibly
    random.Random(random_seed).shuffle(records)

    # Calculate split point
    split_index = int(len(records) * train_ratio)

    train_records = records[:split_index]
    eval_records = records[split_index:]

    # Write training dataset
    with train_file.open("w", encoding="utf-8") as file:
        for record in train_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Write evaluation dataset
    with eval_file.open("w", encoding="utf-8") as file:
        for record in eval_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print results
    print("Dataset split completed successfully!")
    print(f"Total records: {len(records)}")
    print(f"Training records: {len(train_records)}")
    print(f"Evaluation records: {len(eval_records)}")
    print(f"Training file: {train_file.resolve()}")
    print(f"Evaluation file: {eval_file.resolve()}")


if __name__ == "__main__":
    main()
