"""Speaker embedding extraction with a pretrained ECAPA-TDNN (SpeechBrain).

`speechbrain/spkrec-ecapa-voxceleb` was trained on VoxCeleb 1+2 (celebrity interviews
from YouTube); it is used here as-is, with no fine-tuning, on LibriSpeech speakers it
has never seen - so the evaluation also measures cross-domain generalisation.

Pipeline: load audio -> mono 16 kHz -> (optional mobile-channel degradation) ->
ECAPA-TDNN (computes filterbank features internally) -> 192-d embedding -> L2 norm.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

from common.seeding import hash_str

SAMPLE_RATE = 16_000
ROOT = Path(__file__).resolve().parents[1]


class SpeakerEmbedder:
    def __init__(self, source: str = "speechbrain/spkrec-ecapa-voxceleb"):
        self.model = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"),
            run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
            local_strategy=LocalStrategy.COPY,  # Windows: avoid symlinks
        )
        self.model.eval()

    @torch.no_grad()
    def embed(self, wav: np.ndarray) -> Optional[np.ndarray]:
        if wav.size < SAMPLE_RATE // 2:  # < 0.5 s of audio: failure to acquire
            return None
        emb = self.model.encode_batch(torch.from_numpy(wav).float().unsqueeze(0))
        emb = emb.squeeze().cpu().numpy().astype(np.float32)
        return emb / np.linalg.norm(emb)

    def embed_file(self, path, degrade: bool = False) -> Optional[np.ndarray]:
        wav = load_audio(path)
        if degrade:
            wav = degrade_mobile_call(wav, seed=Path(path).name)  # location-independent seed
        return self.embed(wav)


def load_audio(path) -> np.ndarray:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = resample_poly(wav, SAMPLE_RATE, sr).astype(np.float32)
    return wav


def degrade_mobile_call(wav: np.ndarray, seed: str = "", seconds: float = 2.0,
                        snr_db: float = 5.0) -> np.ndarray:
    """Simulate a short voice passphrase captured on a phone in a noisy place.

    Enrollment audio is assumed clean and long (recorded during onboarding); the
    login *probe* is:
      1. short         - a random 2-second segment (a quick spoken passphrase)
      2. narrowband    - 16 kHz -> 8 kHz -> 16 kHz (telephony / low-bitrate codec)
      3. noisy         - additive white noise at 5 dB SNR (market, moto-taxi, traffic)
    Deterministic per file so results are reproducible.
    """
    rng = np.random.default_rng(hash_str(seed))
    n = int(seconds * SAMPLE_RATE)
    if wav.size > n:
        start = rng.integers(0, wav.size - n)
        wav = wav[start:start + n]
    wav = resample_poly(resample_poly(wav, 1, 2), 2, 1)[: wav.size]
    signal_power = np.mean(wav ** 2) + 1e-12
    noise = rng.normal(0, np.sqrt(signal_power / 10 ** (snr_db / 10)), wav.size)
    return (wav + noise).astype(np.float32)


def cosine_score(enrolled: Optional[np.ndarray], probe: Optional[np.ndarray]) -> float:
    """Cosine similarity; a failed capture contributes no evidence (0.0)."""
    if enrolled is None or probe is None:
        return 0.0
    return float(np.dot(enrolled, probe))
