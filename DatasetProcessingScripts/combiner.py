import json
from pathlib import Path


# Get the directory containing this combiner.py file
base_dir = Path(__file__).resolve().parent

# Folder containing all source files
input_folder = base_dir / "text"

# Folder and file for the combined dataset
output_folder = base_dir / "formatted"
output_file = output_folder / "train.jsonl"

# Temporary output prevents replacing an existing dataset
# until the entire combination process succeeds.
temp_output_file = output_folder / "train.jsonl.tmp"


def read_json_records(input_file):
    """
    Read JSON records from a file.

    Supports:
    1. Standard JSONL: one JSON object per line.
    2. Pretty-printed JSON objects spanning multiple lines.
    3. A JSON array containing multiple record objects.

    Each resulting record must be a JSON object.
    """

    # utf-8-sig also handles files containing a UTF-8 BOM.
    raw_text = input_file.read_text(encoding="utf-8-sig")

    decoder = json.JSONDecoder()
    position = 0

    # Continue until the entire file has been processed.
    while position < len(raw_text):

        # Skip whitespace and blank lines between records.
        while (
            position < len(raw_text)
            and raw_text[position].isspace()
        ):
            position += 1

        # Stop when the end of the file is reached.
        if position >= len(raw_text):
            break

        # Parse the next complete JSON value.
        try:
            record, end_position = decoder.raw_decode(
                raw_text, position
            )

        except json.JSONDecodeError as error:
            raise ValueError(
                f"\nInvalid JSON in file: {input_file}\n"
                f"Line: {error.lineno}\n"
                f"Column: {error.colno}\n"
                f"Error: {error.msg}"
            ) from error

        # Move to the beginning of the next JSON value.
        position = end_position

        # If the entire file contains a JSON array,
        # process each element as an individual record.
        if isinstance(record, list):
            for item in record:
                yield item
        else:
            yield record


def validate_record(record, input_file, record_number):
    """
    Validate the structure of one training record.

    Required:
        messages: list of message objects

    Optional:
        tools: list (defaults to an empty list)
    """

    location = (
        f"File: {input_file}\n"
        f"Record: {record_number}"
    )

    # Every training record must be a JSON object.
    if not isinstance(record, dict):
        raise ValueError(
            f"\nExpected a JSON object.\n{location}"
        )

    # The messages field must exist and be a list.
    messages = record.get("messages")

    if not isinstance(messages, list):
        raise ValueError(
            f"\nMissing or invalid 'messages' list.\n"
            f"{location}"
        )

    # Every message must be an object with a string role.
    for message_number, message in enumerate(
        messages, start=1
    ):
        if not isinstance(message, dict):
            raise ValueError(
                f"\nMessage must be a JSON object.\n"
                f"{location}\n"
                f"Message: {message_number}"
            )

        if not isinstance(message.get("role"), str):
            raise ValueError(
                f"\nMessage must contain a string 'role'.\n"
                f"{location}\n"
                f"Message: {message_number}"
            )

    # Add an empty tools list when the field is absent.
    record.setdefault("tools", [])

    # If present, tools must be a list.
    if not isinstance(record["tools"], list):
        raise ValueError(
            f"\nThe 'tools' field must be a list.\n"
            f"{location}"
        )

    return record


def main():
    # Check that the input folder exists.
    if not input_folder.is_dir():
        raise FileNotFoundError(
            f"Input folder does not exist: {input_folder}"
        )

    # Automatically discover every JSONL file in text/.
    # No hardcoded filenames are needed.
    input_files = sorted(
        file
        for file in input_folder.glob("*.jsonl")
        if file.is_file()
    )

    if not input_files:
        raise FileNotFoundError(
            f"No .jsonl files found in: {input_folder}"
        )

    # Create the output folder if necessary.
    output_folder.mkdir(parents=True, exist_ok=True)

    total_records = 0

    print(f"Input folder: {input_folder}")
    print(f"JSONL files found: {len(input_files)}")
    print("-" * 50)

    try:
        # Write to a temporary file first.
        with temp_output_file.open(
            "w", encoding="utf-8"
        ) as outfile:

            # Process each discovered source file.
            for input_file in input_files:
                file_records = 0

                print(f"Processing: {input_file.name}")

                for record_number, record in enumerate(
                    read_json_records(input_file),
                    start=1
                ):
                    # Check the training record structure.
                    record = validate_record(
                        record,
                        input_file,
                        record_number
                    )

                    # Write one compact JSON object per line.
                    outfile.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":")
                        )
                        + "\n"
                    )

                    file_records += 1
                    total_records += 1

                print(
                    f"  Records combined: {file_records}"
                )

        # Replace the final output only after success.
        temp_output_file.replace(output_file)

    except Exception:
        # Remove the incomplete temporary output, if present.
        if temp_output_file.exists():
            temp_output_file.unlink()

        raise

    # Display the final summary.
    print("-" * 50)
    print("Dataset combination completed successfully!")
    print(f"Files processed: {len(input_files)}")
    print(f"Total records combined: {total_records}")
    print(f"Output file: {output_file.resolve()}")


if __name__ == "__main__":
    main()