"""LFW official verification protocol (`pairs.txt`: 10 folds x 300 genuine + 300 impostor)."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LFW_DIR = ROOT / "data" / "lfw"
IMG_DIR = LFW_DIR / "lfw"


def image_key(person: str, idx: int) -> str:
    return f"{person}/{person}_{int(idx):04d}.jpg"


def image_path(key: str) -> Path:
    return IMG_DIR / key


@dataclass
class FacePair:
    enrolled: str  # image captured at onboarding (reference template)
    probe: str     # image presented at login / transaction approval
    genuine: bool
    fold: int


def load_pairs() -> list:
    lines = (LFW_DIR / "pairs.txt").read_text().strip().splitlines()
    n_folds, n_per = map(int, lines[0].split())
    pairs, rows = [], lines[1:]
    for fold in range(n_folds):
        block = rows[fold * 2 * n_per:(fold + 1) * 2 * n_per]
        for r in block:
            p = r.split("\t")
            if len(p) == 3:
                pairs.append(FacePair(image_key(p[0], p[1]), image_key(p[0], p[2]), True, fold))
            else:
                pairs.append(FacePair(image_key(p[0], p[1]), image_key(p[2], p[3]), False, fold))
    return pairs


def identities(min_images: int) -> dict:
    """{person: [image keys]} for identities with at least `min_images` photos."""
    out = {}
    for d in sorted(IMG_DIR.iterdir()):
        imgs = sorted(f"{d.name}/{f.name}" for f in d.glob("*.jpg"))
        if len(imgs) >= min_images:
            out[d.name] = imgs
    return out
