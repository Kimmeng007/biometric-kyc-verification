"""Score-level fusion of face and voice verification scores.

Face (ArcFace cosine) and voice (ECAPA cosine) scores live on different scales, so
each is first normalised against its own *impostor* distribution on the dev set:

    z = (score - mean_impostor) / std_impostor

i.e. "how many standard deviations above a typical impostor is this presentation?".
The fused score is then a weighted sum  w * z_face + (1 - w) * z_voice, with w chosen
on the dev customers. Learned fusion (logistic regression / small MLP / quality-aware
weights) is a natural extension but simple weighted fusion is enough here.
"""
from dataclasses import dataclass, asdict

import numpy as np

from common.metrics import compute_eer, threshold_for_far


@dataclass
class ImpostorZNorm:
    mean: float
    std: float

    @classmethod
    def fit(cls, scores, labels):
        imp = np.asarray(scores)[~np.asarray(labels, bool)]
        return cls(float(imp.mean()), float(imp.std()))

    def __call__(self, scores):
        return (np.asarray(scores) - self.mean) / self.std


@dataclass
class WeightedSumFusion:
    face_norm: ImpostorZNorm
    voice_norm: ImpostorZNorm
    face_weight: float = 0.5

    def __call__(self, face_scores, voice_scores):
        w = self.face_weight
        return w * self.face_norm(face_scores) + (1 - w) * self.voice_norm(voice_scores)

    @classmethod
    def fit(cls, face_scores, voice_scores, labels, weights=np.round(np.arange(0, 1.0001, 0.05), 2)):
        """Normalisers from dev impostors; weight = argmin dev EER
        (ties broken by FRR at the 0.1% FAR high-value-transfer operating point)."""
        fusion = cls(ImpostorZNorm.fit(face_scores, labels), ImpostorZNorm.fit(voice_scores, labels))

        def cost(w):
            fusion.face_weight = float(w)
            fused = fusion(face_scores, voice_scores)
            return compute_eer(fused, labels)[0], threshold_for_far(fused, labels, 0.001)[1]

        fusion.face_weight = float(min(weights, key=cost))
        return fusion

    def to_dict(self):
        return {"face_norm": asdict(self.face_norm), "voice_norm": asdict(self.voice_norm),
                "face_weight": self.face_weight}

    @classmethod
    def from_dict(cls, d):
        return cls(ImpostorZNorm(**d["face_norm"]), ImpostorZNorm(**d["voice_norm"]), d["face_weight"])
