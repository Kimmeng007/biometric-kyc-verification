"""KYC biometric verification service: enroll a customer once, verify them later.

This is the interface a mobile-banking backend would call:

    verifier = KYCBiometricVerifier()
    verifier.enroll("CUST-0001", selfie_bgr, voice_wav)             # remote onboarding
    decision = verifier.verify("CUST-0001", selfie_bgr, voice_wav,   # later: transfer approval
                               risk_tier="high_value_transfer")

Only embeddings (templates) are stored, never raw photos or audio. In production the
template store would be encrypted at rest, and `verify` would sit behind liveness /
anti-spoofing checks (see README "Security considerations").
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from face_verification.embedder import FaceEmbedder, cosine_score as face_cosine
from fusion.score_fusion import WeightedSumFusion
from voice_verification.embedder import SpeakerEmbedder, cosine_score as voice_cosine

CONFIG = Path(__file__).resolve().parents[1] / "results" / "fusion_config.json"

# Risk-based authentication: the riskier the action, the lower the tolerated FAR.
RISK_TIERS = {
    "login": "0.01",                  # FAR 1%   - app login / low-value actions
    "high_value_transfer": "0.001",   # FAR 0.1% - large transfers, adding a payee
}


@dataclass
class BiometricTemplate:
    face: Optional[np.ndarray]
    voice: Optional[np.ndarray]


@dataclass
class VerificationDecision:
    customer_id: str
    risk_tier: str
    accepted: bool
    fused_score: float
    threshold: float
    face_score: float
    voice_score: float
    reason: str
    face_contribution: float = 0.0   # weighted z-score each modality adds to fused_score
    voice_contribution: float = 0.0


class KYCBiometricVerifier:
    def __init__(self, config_path: Path = CONFIG, condition: str = "mobile"):
        """`condition` selects the fusion calibration: "mobile" (fitted on degraded
        login captures - the realistic deployment setting) or "clean"."""
        cfg = json.loads(Path(config_path).read_text())[condition]
        self.fusion = WeightedSumFusion.from_dict(cfg)
        self.thresholds = cfg["thresholds"]
        self.face = FaceEmbedder()
        self.voice = SpeakerEmbedder()
        self.templates: dict = {}

    def enroll(self, customer_id: str, face_bgr: np.ndarray, voice_wav: np.ndarray) -> bool:
        return self.enroll_embeddings(customer_id, self.face.embed(face_bgr), self.voice.embed(voice_wav))

    def enroll_embeddings(self, customer_id: str, face_emb, voice_emb) -> bool:
        if face_emb is None or voice_emb is None:
            return False  # onboarding capture unusable: ask the customer / agent to retake
        self.templates[customer_id] = BiometricTemplate(face_emb, voice_emb)
        return True

    def verify(self, customer_id: str, face_bgr: np.ndarray, voice_wav: np.ndarray,
               risk_tier: str = "login") -> VerificationDecision:
        return self.verify_embeddings(customer_id, self.face.embed(face_bgr), self.voice.embed(voice_wav),
                                      risk_tier)

    def verify_embeddings(self, customer_id: str, face_probe, voice_probe, risk_tier: str = "login",
                          template: Optional[BiometricTemplate] = None) -> VerificationDecision:
        """`template` overrides the internal store (e.g. per-session templates in the web demo)."""
        tpl = template if template is not None else self.templates[customer_id]
        fs, vs = face_cosine(tpl.face, face_probe), voice_cosine(tpl.voice, voice_probe)
        w = self.fusion.face_weight
        face_c = float(w * self.fusion.face_norm(fs))
        voice_c = float((1 - w) * self.fusion.voice_norm(vs))
        fused = face_c + voice_c
        thr = self.thresholds[RISK_TIERS[risk_tier]]
        accepted = fused >= thr
        if face_probe is None or voice_probe is None:
            reason = "capture failed for one modality; decided on remaining evidence"
        else:
            reason = "biometrics match enrolled customer" if accepted else "biometric mismatch"
        return VerificationDecision(customer_id, risk_tier, bool(accepted), fused, float(thr),
                                    fs, vs, reason, face_c, voice_c)
