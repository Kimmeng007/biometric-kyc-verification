"""Biometric verification metrics (1:1 matching), shared by face, voice and fusion.

Terminology, framed for a banking authentication decision:
  * genuine trial  - the real customer presents themselves (should be ACCEPTED)
  * impostor trial - someone else claims the customer's identity (should be REJECTED)
  * FAR (False Accept Rate) - fraction of impostors wrongly accepted  -> fraud risk
  * FRR (False Reject Rate) - fraction of genuine customers rejected -> friction / churn
  * EER (Equal Error Rate)  - the operating point where FAR == FRR

Banks rarely operate at the EER: they pick a threshold for a target FAR set by risk
appetite (e.g. 1% for a low-risk login step-up, 0.1% for high-value transfers) and
then report the FRR that customers pay for that security level.
"""
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# Operating points reported throughout the project (target FAR -> use case).
OPERATING_POINTS = {
    0.01: "login step-up (FAR 1%)",
    0.001: "high-value transfer (FAR 0.1%)",
}


@dataclass
class VerificationReport:
    n_genuine: int
    n_impostor: int
    auc: float
    eer: float
    eer_threshold: float
    frr_at_far: dict  # {target FAR: FRR}
    threshold_at_far: dict  # {target FAR: threshold}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["frr_at_far"] = {str(k): v for k, v in self.frr_at_far.items()}
        d["threshold_at_far"] = {str(k): v for k, v in self.threshold_at_far.items()}
        return d

    def pretty(self, name: str) -> str:
        lines = [f"[{name}]  genuine={self.n_genuine}  impostor={self.n_impostor}",
                 f"  AUC = {self.auc:.5f}",
                 f"  EER = {100 * self.eer:.2f}%  (threshold {self.eer_threshold:.4f})"]
        for far, frr in self.frr_at_far.items():
            lines.append(f"  FRR @ FAR={100 * far:g}% = {100 * frr:.2f}%  "
                         f"(threshold {self.threshold_at_far[far]:.4f})  <- {OPERATING_POINTS[far]}")
        return "\n".join(lines)


def _split(scores: np.ndarray, labels: np.ndarray):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    return scores, labels


def far_frr_curve(scores, labels):
    """Return (thresholds, FAR, FRR); a trial is accepted when score >= threshold."""
    scores, labels = _split(scores, labels)
    fpr, tpr, thr = roc_curve(labels, scores, drop_intermediate=False)
    return thr, fpr, 1.0 - tpr


def compute_eer(scores, labels):
    """EER and its threshold, linearly interpolated at the FAR/FRR crossing."""
    thr, far, frr = far_frr_curve(scores, labels)
    diff = far - frr  # increases from -1 to +1 as the threshold decreases
    i = int(np.argmax(diff >= 0))
    if i == 0:
        return float(far[0]), float(thr[0])
    # interpolate between point i-1 (diff<0) and i (diff>=0)
    w = -diff[i - 1] / (diff[i] - diff[i - 1])
    eer = far[i - 1] + w * (far[i] - far[i - 1])
    t = thr[i - 1] + w * (thr[i] - thr[i - 1]) if np.isfinite(thr[i - 1]) else thr[i]
    return float(eer), float(t)


def threshold_for_far(scores, labels, target_far: float):
    """Lowest threshold whose FAR does not exceed target_far, and the FRR it costs."""
    thr, far, frr = far_frr_curve(scores, labels)
    ok = np.where(far <= target_far)[0]
    i = ok[-1]  # most permissive threshold still within the fraud budget
    return float(thr[i]), float(frr[i])


def evaluate(scores, labels) -> VerificationReport:
    scores, labels = _split(scores, labels)
    eer, eer_thr = compute_eer(scores, labels)
    frr_at, thr_at = {}, {}
    for far in OPERATING_POINTS:
        thr_at[far], frr_at[far] = threshold_for_far(scores, labels, far)
    return VerificationReport(
        n_genuine=int(labels.sum()),
        n_impostor=int((~labels).sum()),
        auc=float(roc_auc_score(labels, scores)),
        eer=eer,
        eer_threshold=eer_thr,
        frr_at_far=frr_at,
        threshold_at_far=thr_at,
    )


def bootstrap_by_customer(systems: dict, labels, customer_ids, reference: str, n_boot: int = 1000,
                          seed: int = 0, far: float = 0.001) -> dict:
    """95% confidence intervals from resampling *customers* (accounts), not trials.

    All attempts against one account share an enrolled template, so trials are not
    independent; resampling whole customers gives honest (wider) intervals.
    `systems` = {name: scores}. For every non-reference system we also report how
    often it beats `reference` across resamples.
    Returns {name: {"eer": [lo, hi], "frr_at_far": [lo, hi], ...}}.
    """
    labels = np.asarray(labels, bool)
    customer_ids = np.asarray(customer_ids)
    uniq = np.unique(customer_ids)
    idx_of = {c: np.where(customer_ids == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    stats = {name: {"eer": [], "frr": []} for name in systems}
    for _ in range(n_boot):
        idx = np.concatenate([idx_of[c] for c in rng.choice(uniq, size=len(uniq), replace=True)])
        for name, s in systems.items():
            s = np.asarray(s)
            stats[name]["eer"].append(compute_eer(s[idx], labels[idx])[0])
            stats[name]["frr"].append(threshold_for_far(s[idx], labels[idx], far)[1])
    out = {}
    for name, st in stats.items():
        eer, frr = np.array(st["eer"]), np.array(st["frr"])
        out[name] = {"eer_ci95": np.percentile(eer, [2.5, 97.5]).tolist(),
                     "frr_at_far_ci95": np.percentile(frr, [2.5, 97.5]).tolist()}
        if name != reference:
            ref_eer, ref_frr = np.array(stats[reference]["eer"]), np.array(stats[reference]["frr"])
            out[name]["p_beats_reference_eer"] = float(np.mean(eer < ref_eer))
            out[name]["p_beats_reference_frr"] = float(np.mean(frr < ref_frr))
    return out


def apply_threshold(scores, labels, threshold: float):
    """FAR/FRR of a *fixed* threshold (e.g. one tuned on a dev set) on new trials."""
    scores, labels = _split(scores, labels)
    accept = scores >= threshold
    far = float(accept[~labels].mean())
    frr = float((~accept[labels]).mean())
    return far, frr
