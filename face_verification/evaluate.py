"""Evaluate face verification on the LFW 6,000-pair protocol.

Two capture conditions:
  * clean  - enrolled and probe photos as they are in LFW
  * mobile - enrolled photo clean (onboarding), probe photo degraded to mimic a
             low-quality login selfie (see `degrade_mobile_selfie`)

Usage:  python -m face_verification.evaluate
"""
import json

import numpy as np

from common.embedding_cache import ROOT, cached_embeddings
from common.metrics import evaluate
from common.plots import plot_roc_panels
from face_verification.embedder import FaceEmbedder, cosine_score
from face_verification.lfw import image_path, load_pairs

RESULTS = ROOT / "results"
_embedder = None


def get_embedder() -> FaceEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = FaceEmbedder()
    return _embedder


def face_embeddings(keys, degraded: bool) -> dict:
    name = "face_mobile" if degraded else "face_clean"
    return cached_embeddings(name, keys, lambda k: get_embedder().embed_file(image_path(k), degrade=degraded))


def lfw_10fold_accuracy(scores: np.ndarray, labels: np.ndarray, folds: np.ndarray) -> float:
    """Standard LFW accuracy: threshold picked on 9 folds, measured on the held-out fold."""
    accs = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        cands = np.unique(scores[tr])
        best = max(cands, key=lambda t: ((scores[tr] >= t) == labels[tr]).mean())
        accs.append(((scores[te] >= best) == labels[te]).mean())
    return float(np.mean(accs))


def main():
    pairs = load_pairs()
    labels = np.array([p.genuine for p in pairs])
    folds = np.array([p.fold for p in pairs])
    clean = face_embeddings([k for p in pairs for k in (p.enrolled, p.probe)], degraded=False)
    mobile = face_embeddings([p.probe for p in pairs], degraded=True)

    conditions = {
        "clean": np.array([cosine_score(clean[p.enrolled], clean[p.probe]) for p in pairs]),
        "mobile": np.array([cosine_score(clean[p.enrolled], mobile[p.probe]) for p in pairs]),
    }
    fta = {
        "clean": float(np.mean([clean[k] is None for k in {p.probe for p in pairs}])),
        "mobile": float(np.mean([mobile[k] is None for k in {p.probe for p in pairs}])),
    }

    out, panels = {}, {}
    for cond, scores in conditions.items():
        rep = evaluate(scores, labels)
        print(rep.pretty(f"Face / LFW / {cond}"))
        print(f"  failure-to-acquire (no face detected in probe) = {100 * fta[cond]:.2f}%")
        out[cond] = rep.to_dict() | {"failure_to_acquire": fta[cond]}
        panels[f"{cond.capitalize()} capture"] = {"Face": (scores, labels, rep.eer)}
    acc = lfw_10fold_accuracy(conditions["clean"], labels, folds)
    out["clean"]["lfw_10fold_accuracy"] = acc
    print(f"\nLFW standard 10-fold accuracy (clean) = {100 * acc:.2f}%")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "face_metrics.json").write_text(json.dumps(out, indent=2))
    plot_roc_panels(panels, RESULTS / "face_roc.png", "Face verification - ArcFace on LFW (6,000 pairs)")
    print(f"Saved {RESULTS / 'face_metrics.json'} and face_roc.png")


if __name__ == "__main__":
    main()
