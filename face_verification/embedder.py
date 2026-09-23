"""Face embedding extraction: detection -> 5-point alignment -> ArcFace embedding.

Uses InsightFace's `buffalo_l` model pack (pretrained, not trained here):
  * RetinaFace-style detector (det_10g) returning a box + 5 facial landmarks
  * similarity-transform alignment of those landmarks to the canonical ArcFace
    112x112 template (done inside FaceAnalysis via `face_align.norm_crop`)
  * ArcFace ResNet-50 (w600k_r50) -> 512-d L2-normalised identity embedding
"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
import torch  # noqa: F401  (loads the CUDA/cuDNN DLLs that onnxruntime-gpu reuses)
from insightface.app import FaceAnalysis
from insightface.utils import face_align

from common.seeding import hash_str


class FaceEmbedder:
    def __init__(self, model_pack: str = "buffalo_l", det_size=(320, 320)):
        use_gpu = "CUDAExecutionProvider" in ort.get_available_providers() and torch.cuda.is_available()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.app = FaceAnalysis(
            name=model_pack,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=det_size)

    def _customer_face(self, img_bgr: np.ndarray):
        faces = self.app.get(img_bgr)
        if not faces:
            return None
        # If several faces are visible (bystanders), use the one nearest the frame
        # centre - the person holding the phone / facing the agent's camera.
        h, w = img_bgr.shape[:2]
        centre = np.array([w / 2, h / 2])
        return min(faces, key=lambda f: np.linalg.norm((f.bbox[:2] + f.bbox[2:]) / 2 - centre))

    def embed(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """512-d unit embedding of the customer's face, or None if no face was found.

        A None is a *failure to acquire* - in a mobile KYC flow the app would ask the
        customer to retake the selfie; in evaluation it counts against the modality.
        """
        face = self._customer_face(img_bgr)
        return None if face is None else face.normed_embedding.astype(np.float32)

    def embed_with_crop(self, img_bgr: np.ndarray):
        """(embedding, aligned 112x112 crop the model actually saw) - for display."""
        face = self._customer_face(img_bgr)
        if face is None:
            return None, None
        crop = face_align.norm_crop(img_bgr, landmark=face.kps, image_size=112)
        return face.normed_embedding.astype(np.float32), crop

    def embed_file(self, path, degrade: bool = False) -> Optional[np.ndarray]:
        img = cv2.imread(str(path))
        if degrade:
            img = degrade_mobile_selfie(img, seed=Path(path).name)  # location-independent seed
        return self.embed(img)


def degrade_mobile_selfie(img_bgr: np.ndarray, seed: str = "") -> np.ndarray:
    """Simulate a poor-quality login selfie from a low-end phone.

    Enrollment photos are assumed to be captured under good conditions (branch /
    agent-assisted onboarding); *probe* photos at login are often worse. Applied:
      1. low resolution  - 1/4 downscale then upscale (small face, cheap camera)
      2. low light       - exposure x0.55 plus Gaussian sensor noise (sigma=10)
      3. compression     - JPEG quality 20 (uploads over a weak mobile connection)
    Deterministic per image so results are reproducible.
    """
    rng = np.random.default_rng(hash_str(seed))
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    img = img * 0.55 + rng.normal(0, 10, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 20])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def cosine_score(enrolled: Optional[np.ndarray], probe: Optional[np.ndarray]) -> float:
    """Similarity between the enrolled template and the probe.

    A failed capture on either side contributes *no evidence* (score 0.0, roughly
    the centre of the impostor distribution), so it is rejected on its own but can
    still be outvoted by a strong voice match in the fused decision.
    """
    if enrolled is None or probe is None:
        return 0.0
    return float(np.dot(enrolled, probe))
