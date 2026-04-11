#!/usr/bin/env python3
"""
Download reference images from Hugging Face dataset.

Usage:
    python scripts/download_images.py              # download all folders
    python scripts/download_images.py gemini_black  # download specific folder
"""

import sys
import os

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Install huggingface_hub first:")
    print("  pip install huggingface_hub")
    sys.exit(1)


DATASET_REPO = "aoxo/reverse-synthid"
IMAGE_FOLDERS = [
    "gemini_black",
    "gemini_white",
    "gemini_black_nb_pro",
]


def download(folders=None, output_dir="."):
    """Download image folders from HF dataset repo."""
    if folders is None:
        folders = IMAGE_FOLDERS

    for folder in folders:
        print(f"Downloading {folder}/ from {DATASET_REPO}...")
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            allow_patterns=f"{folder}/*",
            local_dir=output_dir,
        )
        count = len([f for f in os.listdir(os.path.join(output_dir, folder))
                      if not f.startswith('.')])
        print(f"  {folder}/: {count} images\n")

    print("Done!")


if __name__ == "__main__":
    folders = sys.argv[1:] if len(sys.argv) > 1 else None
    download(folders=folders)
