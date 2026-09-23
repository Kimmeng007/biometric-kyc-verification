"""Unimodal vs. fused verification on virtual bank customers - the headline result.

For each capture condition (clean / mobile):
  1. score every trial with face only and voice only (cosine vs. the enrolled template)
  2. fit score normalisation + fusion weight on the 40 *dev* customers
  3. report face / voice / fused FAR, FRR, EER, AUC on the 40 unseen *test* customers
  4. deployment check: fix each system's threshold on dev for a target FAR, apply
     it unchanged to test, and report the FAR/FRR a bank would actually observe

Usage:  python -m fusion.evaluate
"""
import json

import numpy as np

from common.embedding_cache import ROOT
from common.metrics import (OPERATING_POINTS, apply_threshold, bootstrap_by_customer, evaluate,
                            threshold_for_far)
from common.plots import plot_roc_panels
from face_verification.embedder import cosine_score as face_cosine
from face_verification.evaluate import face_embeddings
from fusion.chimeric import build_customers, make_trials, split_customers
from fusion.score_fusion import WeightedSumFusion
from voice_verification.embedder import cosine_score as voice_cosine
from voice_verification.evaluate import voice_embeddings

RESULTS = ROOT / "results"
FACE, VOICE, FUSED = "Face", "Voice", "Fused (face + voice)"


def score_trials(trials, customers, face_probe_emb, voice_probe_emb, face_enr_emb, voice_enr_emb):
    by_id = {c.customer_id: c for c in customers}
    face = np.array([face_cosine(face_enr_emb[by_id[t.claimed].enrolled_face], face_probe_emb[t.face_probe])
                     for t in trials])
    voice = np.array([voice_cosine(voice_enr_emb[by_id[t.claimed].enrolled_voice], voice_probe_emb[t.voice_probe])
                      for t in trials])
    labels = np.array([t.genuine for t in trials])
    return face, voice, labels


def main():
    customers = build_customers()
    dev_customers, test_customers = split_customers(customers)
    trials = {"dev": make_trials(dev_customers, seed=1), "test": make_trials(test_customers, seed=2)}

    face_keys = [k for c in customers for k in [c.enrolled_face, *c.probe_faces]]
    voice_keys = [k for c in customers for k in [c.enrolled_voice, *c.probe_voices]]
    face_clean = face_embeddings(face_keys, degraded=False)
    voice_clean = voice_embeddings(voice_keys, degraded=False)
    probe_emb = {
        "clean": (face_clean, voice_clean),
        "mobile": (face_embeddings([p for c in customers for p in c.probe_faces], degraded=True),
                   voice_embeddings([p for c in customers for p in c.probe_voices], degraded=True)),
    }

    results, panels, deploy_config = {}, {}, {}
    for cond, (face_probe, voice_probe) in probe_emb.items():
        scored = {split: score_trials(trials[split], customers, face_probe, voice_probe, face_clean, voice_clean)
                  for split in trials}
        f_dev, v_dev, y_dev = scored["dev"]
        f_te, v_te, y_te = scored["test"]

        fusion = WeightedSumFusion.fit(f_dev, v_dev, y_dev)
        equal = WeightedSumFusion(fusion.face_norm, fusion.voice_norm, 0.5)
        systems_dev = {FACE: f_dev, VOICE: v_dev, FUSED: fusion(f_dev, v_dev)}
        systems_te = {FACE: f_te, VOICE: v_te, FUSED: fusion(f_te, v_te)}

        print(f"\n=== {cond.upper()} capture | test customers: {len(test_customers)} | "
              f"fusion weight (face) = {fusion.face_weight:.2f} ===")
        cond_out = {"face_weight": fusion.face_weight, "systems": {}}
        panels[f"{cond.capitalize()} capture"] = {}
        for name, s in systems_te.items():
            rep = evaluate(s, y_te)
            print(rep.pretty(name))
            deploy = {}
            for far in OPERATING_POINTS:
                thr, _ = threshold_for_far(systems_dev[name], y_dev, far)
                test_far, test_frr = apply_threshold(s, y_te, thr)
                deploy[str(far)] = {"threshold_from_dev": thr, "test_far": test_far, "test_frr": test_frr}
                print(f"  deployed @ dev-FAR {100 * far:g}% -> test FAR {100 * test_far:.2f}%, "
                      f"test FRR {100 * test_frr:.2f}%")
            cond_out["systems"][name] = rep.to_dict() | {"deployment": deploy}
            panels[f"{cond.capitalize()} capture"][name] = (s, y_te, rep.eer)
        boot = bootstrap_by_customer(systems_te, y_te, [t.claimed for t in trials["test"]], reference=FACE)
        cond_out["bootstrap_ci95"] = boot
        print("  95% CI (1,000 customer-level bootstrap resamples):")
        for name, b in boot.items():
            lo, hi = b["eer_ci95"]
            flo, fhi = b["frr_at_far_ci95"]
            line = f"    {name:<22} EER [{100 * lo:.2f}, {100 * hi:.2f}]%  FRR@FAR0.1% [{100 * flo:.2f}, {100 * fhi:.2f}]%"
            if "p_beats_reference_eer" in b:
                line += (f"  | beats face in {100 * b['p_beats_reference_eer']:.1f}% (EER), "
                         f"{100 * b['p_beats_reference_frr']:.1f}% (FRR) of resamples")
            print(line)
        eq = evaluate(equal(f_te, v_te), y_te)
        cond_out["equal_weight_fusion"] = eq.to_dict()
        print(f"  (equal-weight fusion w=0.5: EER {100 * eq.eer:.2f}%, "
              f"FRR@FAR0.1% {100 * eq.frr_at_far[0.001]:.2f}%)")

        # Customers the face-only system would have blocked at the high-value threshold
        # but fusion lets through (at the same 0.1% fraud budget, thresholds from dev).
        thr_face, _ = threshold_for_far(f_dev, y_dev, 0.001)
        thr_fused, _ = threshold_for_far(systems_dev[FUSED], y_dev, 0.001)
        face_rej = (f_te < thr_face) & y_te
        rescued = face_rej & (systems_te[FUSED] >= thr_fused)
        cond_out["genuine_rejected_by_face_rescued_by_fusion"] = [int(rescued.sum()), int(face_rej.sum())]
        print(f"  genuine attempts rejected by face-only @0.1% FAR: {face_rej.sum()}, "
              f"of which fusion accepts: {rescued.sum()}")
        results[cond] = cond_out
        deploy_config[cond] = fusion.to_dict() | {
            "thresholds": {str(far): threshold_for_far(systems_dev[FUSED], y_dev, far)[0] for far in OPERATING_POINTS}}

    results["protocol"] = {
        "customers_total": len(customers), "dev_customers": len(dev_customers), "test_customers": len(test_customers),
        "test_genuine_trials": int(sum(t.genuine for t in trials["test"])),
        "test_impostor_trials": int(sum(not t.genuine for t in trials["test"])),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fusion_metrics.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "fusion_config.json").write_text(json.dumps(deploy_config, indent=2))
    plot_roc_panels(panels, RESULTS / "fusion_roc.png",
                    "Face vs voice vs fused verification - 40 unseen virtual customers (test split)")
    print(f"\nSaved fusion_metrics.json, fusion_config.json, fusion_roc.png to {RESULTS}")


if __name__ == "__main__":
    main()
