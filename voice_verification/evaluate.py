"""Evaluate speaker verification on LibriSpeech (80 speakers, 16,000 trials).

Two capture conditions, mirroring the face evaluation:
  * clean  - full-length clean utterances on both sides
  * mobile - clean enrollment, probe degraded to a 2 s narrowband noisy passphrase

Usage:  python -m voice_verification.evaluate
"""
import json

import numpy as np

from common.embedding_cache import ROOT, cached_embeddings
from common.metrics import evaluate
from common.plots import plot_roc_panels
from voice_verification.embedder import SpeakerEmbedder, cosine_score
from voice_verification.librispeech import make_trials, utterance_path

RESULTS = ROOT / "results"
_embedder = None


def get_embedder() -> SpeakerEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = SpeakerEmbedder()
    return _embedder


def voice_embeddings(keys, degraded: bool) -> dict:
    name = "voice_mobile" if degraded else "voice_clean"
    return cached_embeddings(name, keys, lambda k: get_embedder().embed_file(utterance_path(k), degrade=degraded))


def main():
    trials = make_trials()
    labels = np.array([t.genuine for t in trials])
    clean = voice_embeddings([k for t in trials for k in (t.enrolled, t.probe)], degraded=False)
    mobile = voice_embeddings([t.probe for t in trials], degraded=True)

    conditions = {
        "clean": np.array([cosine_score(clean[t.enrolled], clean[t.probe]) for t in trials]),
        "mobile": np.array([cosine_score(clean[t.enrolled], mobile[t.probe]) for t in trials]),
    }
    out, panels = {}, {}
    for cond, scores in conditions.items():
        rep = evaluate(scores, labels)
        print(rep.pretty(f"Voice / LibriSpeech / {cond}"))
        out[cond] = rep.to_dict()
        panels[f"{cond.capitalize()} capture"] = {"Voice": (scores, labels, rep.eer)}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "voice_metrics.json").write_text(json.dumps(out, indent=2))
    plot_roc_panels(panels, RESULTS / "voice_roc.png",
                    "Voice verification - ECAPA-TDNN on LibriSpeech (16,000 trials)")
    print(f"Saved {RESULTS / 'voice_metrics.json'} and voice_roc.png")


if __name__ == "__main__":
    main()
