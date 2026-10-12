
from pathlib import Path
from huggingface_hub import snapshot_download

# Hugging Face dataset repository ID
repo_id = "Crownelius/Creative-Writing-KimiK2.5-Cleaned"

# Local directory where the dataset will be saved
download_dir = Path(
    "./datasets/Creative-Writing-KimiK2.5-Cleaned"
)

def main():
    download_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Hugging Face...")

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(download_dir)
    )

    print("Download completed successfully!")
    print(f"Dataset location: {local_path}")

if __name__ == "__main__":
    main()
