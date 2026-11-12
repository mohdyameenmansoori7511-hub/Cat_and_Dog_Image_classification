"""Dataset Collection and Folder Setup Module"""
from pathlib import Path
import urllib.request
import zipfile

MIRROR_URLS = [
    "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
    "https://download.mlcc.google.com/mledu-datasets/cats_and_dogs_filtered.zip",
]


def download_and_extract_dataset(raw_dir: str = "./data/raw_download") -> Path:
    """Downloads and extracts the Cats vs. Dogs benchmark archive."""
    target_dir = Path(raw_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "cats_and_dogs_filtered.zip"
    extracted_dir = target_dir / "cats_and_dogs_filtered"

    if not extracted_dir.exists():
        for url in MIRROR_URLS:
            try:
                urllib.request.urlretrieve(url, zip_path)
                break
            except Exception:
                continue
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(target_dir)
    return extracted_dir
