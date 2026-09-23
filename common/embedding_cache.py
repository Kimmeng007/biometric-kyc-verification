"""Tiny on-disk cache so each image / utterance is embedded once per condition."""
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "cache"


def cached_embeddings(name: str, keys: Iterable[str],
                      embed_fn: Callable[[str], Optional[np.ndarray]]) -> dict:
    """Return {key: embedding or None}, computing only keys missing from the cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{name}.npz"
    store = {}
    if path.exists():
        with np.load(path) as z:
            store = {k: (None if np.isnan(z[k]).all() else z[k]) for k in z.files}
    missing = [k for k in dict.fromkeys(keys) if k not in store]
    if missing:
        for i, k in enumerate(tqdm(missing, desc=f"embedding [{name}]")):
            store[k] = embed_fn(k)
            if (i + 1) % 1000 == 0:
                _save(path, store)
        _save(path, store)
    return store


def _save(path: Path, store: dict) -> None:
    dim = next((v.shape[0] for v in store.values() if v is not None), 1)
    np.savez(path, **{k: (v if v is not None else np.full(dim, np.nan, np.float32))
                      for k, v in store.items()})
