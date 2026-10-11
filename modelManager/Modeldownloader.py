from huggingface_hub import snapshot_download
import getpass
import os

MODEL_ID = "google/gemma-4-E2B"
DOWNLOAD_DIR = "models/Gemma-4-E2B"

print("=" * 60)
print("EmbeddingGemma-4-E2B Downloader")
print("=" * 60)

# Ask for Hugging Face token
token = getpass.getpass("Enter your Hugging Face token: ")

if not token.strip():
    print("Error: No token provided.")
    exit(1)

print("\nStarting download...")
print(f"Model : {MODEL_ID}")
print(f"Folder: {os.path.abspath(DOWNLOAD_DIR)}")
print()

try:
    path = snapshot_download(
        repo_id=MODEL_ID,
        token=token,
        local_dir=DOWNLOAD_DIR,
        resume_download=True
    )

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETED")
    print("=" * 60)
    print(f"Model saved to:")
    print(path)

except Exception as e:
    print("\nDownload failed!")
    print(f"Error: {e}")