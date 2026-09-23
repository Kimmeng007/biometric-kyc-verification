"""End-to-end KYC demo: onboard two virtual customers, then run login and
high-value-transfer checks - one by the real customer, one by an impostor.

Run after `python -m fusion.evaluate` (which writes results/fusion_config.json):
    python scripts/demo.py
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_verification.embedder import degrade_mobile_selfie  # noqa: E402
from face_verification.lfw import image_path  # noqa: E402
from fusion.chimeric import build_customers, split_customers  # noqa: E402
from fusion.kyc_verifier import KYCBiometricVerifier  # noqa: E402
from voice_verification.embedder import degrade_mobile_call, load_audio  # noqa: E402
from voice_verification.librispeech import utterance_path  # noqa: E402


def mobile_capture(face_key, voice_key):
    """What arrives from the customer's phone at login time (degraded capture)."""
    face = degrade_mobile_selfie(cv2.imread(str(image_path(face_key))), seed=Path(face_key).name)
    voice = degrade_mobile_call(load_audio(utterance_path(voice_key)), seed=Path(voice_key).name)
    return face, voice


def main():
    _, test_customers = split_customers(build_customers())  # customers never used for tuning
    alice, bob = test_customers[0], test_customers[1]
    kyc = KYCBiometricVerifier(condition="mobile")

    for c in (alice, bob):
        ok = kyc.enroll(c.customer_id, cv2.imread(str(image_path(c.enrolled_face))),
                        load_audio(utterance_path(c.enrolled_voice)))
        print(f"Onboarded {c.customer_id} (1 selfie + 1 voice sample) -> {'OK' if ok else 'RETAKE'}")

    scenarios = [
        ("Genuine customer logs in", alice, alice, "login"),
        ("Genuine customer approves a large transfer", alice, alice, "high_value_transfer"),
        ("Impostor tries a large transfer on the account", alice, bob, "high_value_transfer"),
    ]
    print()
    for title, account, presenter, tier in scenarios:
        face, voice = mobile_capture(presenter.probe_faces[0], presenter.probe_voices[0])
        d = kyc.verify(account.customer_id, face, voice, risk_tier=tier)
        print(f"{title}  [{account.customer_id}, tier={tier}]")
        print(f"  face={d.face_score:.3f}  voice={d.voice_score:.3f}  fused={d.fused_score:.2f} "
              f"vs threshold {d.threshold:.2f}  ->  {'APPROVED' if d.accepted else 'DECLINED'} ({d.reason})\n")


if __name__ == "__main__":
    main()
