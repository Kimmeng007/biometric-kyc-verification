"""Create the sample files bundled with the web demo (demo_examples/).

A visitor without a webcam or microphone can still try the app: one sample customer is
pre-enrolled, and there are two ready-made login attempts - the real customer and a
same-gender impostor. Both are "mobile" captures (degraded selfie + 2 s noisy clip),
like a real login. Images come from LFW and audio from LibriSpeech (public
benchmarks); both identities are test-split customers never used for tuning.

    python scripts/make_demo_examples.py
"""
import sys
from pathlib import Path

import cv2
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from face_verification.embedder import degrade_mobile_selfie  # noqa: E402
from face_verification.lfw import image_path  # noqa: E402
from fusion.chimeric import build_customers, split_customers  # noqa: E402
from voice_verification.embedder import SAMPLE_RATE, degrade_mobile_call, load_audio  # noqa: E402
from voice_verification.librispeech import utterance_path  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "demo_examples"
CUSTOMER_IDX, IMPOSTOR_IDX = 12, 14  # test-split customers; both have male voices


def main():
    OUT.mkdir(exist_ok=True)
    _, test = split_customers(build_customers())
    customer, impostor = test[CUSTOMER_IDX], test[IMPOSTOR_IDX]

    cv2.imwrite(str(OUT / "enroll_selfie.jpg"), cv2.imread(str(image_path(customer.enrolled_face))))
    sf.write(OUT / "enroll_voice.wav", load_audio(utterance_path(customer.enrolled_voice)), SAMPLE_RATE)

    for name, person in (("genuine", customer), ("impostor", impostor)):
        face_key, voice_key = person.probe_faces[0], person.probe_voices[0]
        # same seeds as the evaluation, so these are exactly the degraded probes that were scored
        img = degrade_mobile_selfie(cv2.imread(str(image_path(face_key))), seed=Path(face_key).name)
        cv2.imwrite(str(OUT / f"{name}_selfie.jpg"), img)
        sf.write(OUT / f"{name}_voice.wav", degrade_mobile_call(load_audio(utterance_path(voice_key)),
                                                                 seed=Path(voice_key).name), SAMPLE_RATE)
    print(f"Wrote demo examples to {OUT}")


if __name__ == "__main__":
    main()
