"""Download the public benchmark datasets used in this project.

  * LFW (Labeled Faces in the Wild) + the official 10-fold `pairs.txt` protocol
  * LibriSpeech test-clean + dev-clean (80 speakers, read English speech)

Archives are stream-extracted (never written to disk as a .tgz) to keep the
footprint small. Only public benchmark data is used -- no real customer data.

Usage:  python scripts/download_data.py
"""
import hashlib
import tarfile
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"

LFW_ARCHIVE = "https://ndownloader.figshare.com/files/5976018"  # lfw.tgz (original, un-aligned)
LFW_PAIRS = ("https://ndownloader.figshare.com/files/5976006",
             "ea42330c62c92989f9d7c03237ed5d591365e89b3e649747777b70e692dc1592")
LIBRISPEECH = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
}


def stream_extract(url: str, dest: Path) -> None:
    print(f"Downloading + extracting {url} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        tar.extractall(dest, filter="data")


def download_lfw() -> None:
    lfw_dir = DATA / "lfw"
    if not (lfw_dir / "lfw").exists():
        stream_extract(LFW_ARCHIVE, lfw_dir)
    pairs = lfw_dir / "pairs.txt"
    if not pairs.exists():
        url, sha = LFW_PAIRS
        content = urllib.request.urlopen(url).read()
        assert hashlib.sha256(content).hexdigest() == sha, "pairs.txt checksum mismatch"
        pairs.write_bytes(content)
    print(f"LFW ready: {sum(1 for _ in (lfw_dir / 'lfw').glob('*/*.jpg'))} images")


def download_librispeech() -> None:
    for subset, url in LIBRISPEECH.items():
        if not (DATA / "LibriSpeech" / subset).exists():
            stream_extract(url, DATA)
    n = sum(1 for _ in (DATA / "LibriSpeech").glob("*/*/*/*.flac"))
    print(f"LibriSpeech ready: {n} utterances")


if __name__ == "__main__":
    download_lfw()
    download_librispeech()
