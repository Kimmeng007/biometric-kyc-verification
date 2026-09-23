"""Web demo: onboard a customer with a selfie + voice sample, then verify them at login
or before a high-value transfer.

    python app.py            # then open http://127.0.0.1:7860

A thin UI over `fusion.kyc_verifier.KYCBiometricVerifier` - no model logic lives here.
Enrolled templates are kept per browser session, in memory only, and are gone when the
tab closes or the app restarts.
"""
from math import gcd
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from scipy.signal import resample_poly

from fusion.kyc_verifier import RISK_TIERS, BiometricTemplate, KYCBiometricVerifier
from voice_verification.embedder import SAMPLE_RATE, load_audio

ROOT = Path(__file__).resolve().parent
EXAMPLES = ROOT / "demo_examples"
SAMPLE_ID = "SAMPLE-CUSTOMER"
TIERS = {"Login  (fraud tolerance 1 in 100)": "login",
         "High-value transfer  (fraud tolerance 1 in 1,000)": "high_value_transfer"}
TIER_SHORT = {"login": "Login", "high_value_transfer": "High-value transfer"}
GOOD, CRITICAL, ACCENT = "#0ca30c", "#d03b3b", "#2a78d6"

kyc = KYCBiometricVerifier(condition="mobile")


# ---------------------------------------------------------------- input conversion
def to_bgr(img_rgb):
    return None if img_rgb is None else cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


def to_rgb(img_bgr):
    return None if img_bgr is None else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_wav16k(audio):
    """Gradio (sample_rate, int/float array) -> float32 mono at 16 kHz."""
    if audio is None:
        return None
    sr, data = audio
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = gcd(sr, SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // g, sr // g)
    return data.astype(np.float32)


def _sample_store():
    face, crop = kyc.face.embed_with_crop(cv2.imread(str(EXAMPLES / "enroll_selfie.jpg")))
    voice = kyc.voice.embed(load_audio(EXAMPLES / "enroll_voice.wav"))
    return {SAMPLE_ID: {"template": BiometricTemplate(face, voice), "crop": to_rgb(crop)}}


INITIAL_STORE = _sample_store()


# ---------------------------------------------------------------- result rendering
def strength(z):
    """Plain-language evidence level from a z-score (std devs above a typical impostor)."""
    if z is None:
        return "Not captured"
    if z >= 6:
        return "Strong match"
    if z >= 3:
        return "Moderate match"
    if z >= 1.5:
        return "Weak match"
    return "No match"


def message(kind, text):
    color = GOOD if kind == "ok" else CRITICAL
    icon = "✔" if kind == "ok" else "✖"
    return (f'<div style="border-left:4px solid {color};padding:10px 14px;border-radius:6px;'
            f'background:var(--background-fill-secondary)"><b style="color:{color}">{icon}</b> {text}</div>')


def decision_html(d, face_ok, voice_ok):
    color, icon, word = (GOOD, "✔", "APPROVED") if d.accepted else (CRITICAL, "✖", "DECLINED")
    w = kyc.fusion.face_weight
    face_z = d.face_contribution / w if face_ok else None
    voice_z = d.voice_contribution / (1 - w) if voice_ok else None

    # meter: fused score on a common scale with both pass marks
    marks = {t: kyc.thresholds[RISK_TIERS[t]] for t in RISK_TIERS}
    lo = min(-2.0, d.fused_score - 1)
    hi = max(max(marks.values()) * 2.2, d.fused_score + 1)
    pos = lambda v: 100 * (v - lo) / (hi - lo)  # noqa: E731
    def tick(t, v):
        active = t == d.risk_tier
        # active tier's label sits below the bar, the other tier's above - they never collide
        label_pos = "top:22px" if active else "bottom:22px"
        return (f'<div style="position:absolute;left:{pos(v):.1f}%;top:-5px;bottom:-5px;width:2px;'
                f'background:var(--body-text-color);opacity:{1 if active else .35}"></div>'
                f'<div style="position:absolute;left:{pos(v):.1f}%;{label_pos};transform:translateX(-50%);'
                f'font-size:12px;white-space:nowrap;color:var(--body-text-color{"" if active else "-subdued"})">'
                f'{"<b>" if active else ""}{TIER_SHORT[t]} pass mark {v:.2f}{"</b>" if active else ""}</div>')

    ticks = "".join(tick(t, v) for t, v in marks.items())
    fill_w = max(pos(d.fused_score), 1.5)

    def row(name, cos, contrib, z, ok):
        detail = (f"similarity {cos:.2f} &middot; adds {contrib:+.2f} to the score" if ok
                  else "capture failed - retake in better conditions")
        cell = "padding:6px 14px 6px 0;border:none;background:none"
        return (f'<tr style="border:none;background:none"><td style="{cell}"><b>{name}</b></td>'
                f'<td style="{cell}">{strength(z)}</td>'
                f'<td style="{cell};color:var(--body-text-color-subdued)">{detail}</td></tr>')

    # Plain-language explanation when the two modalities disagree
    note = ""
    zs = {"face": face_z, "voice": voice_z}
    strong = [m for m, z in zs.items() if z is not None and z >= 3]
    weak = [m for m, z in zs.items() if z is None or z < 1.5]
    if d.accepted and len(strong) == 1 and len(weak) == 1:
        note = (f"Approved mainly on <b>{strong[0]}</b> evidence. The {weak[0]} capture was inconclusive "
                f"(e.g. background noise, a very short clip or poor light), and the other modality "
                f"compensated. This is the benefit of combining the two.")
    elif not d.accepted and len(strong) == 1:
        note = (f"The {strong[0]} matched, but the combined evidence did not reach the pass mark "
                f"for this action. A real app would ask for a retake or another check.")
    note_html = (f'<div style="margin-top:12px;font-size:14px;color:var(--body-text-color-subdued)">{note}</div>'
                 if note else "")

    return f"""
<div style="border:1px solid var(--border-color-primary);border-radius:12px;padding:18px 20px">
  <div style="display:flex;align-items:baseline;gap:12px;flex-wrap:wrap">
    <span style="font-size:28px;font-weight:700;color:{color}">{icon} {word}</span>
    <span style="color:var(--body-text-color-subdued)">{TIER_SHORT[d.risk_tier]} for account <b>{d.customer_id}</b></span>
  </div>
  <div style="margin:44px 0 44px;position:relative;height:14px;border-radius:7px;
              background:var(--background-fill-secondary);border:1px solid var(--border-color-primary)">
    <div style="position:absolute;left:0;width:{fill_w:.1f}%;top:0;bottom:0;
                border-radius:7px;background:{ACCENT}"></div>
    {ticks}
  </div>
  <div style="margin-bottom:10px">Combined score <b>{d.fused_score:.2f}</b> vs pass mark <b>{d.threshold:.2f}</b></div>
  <table style="border-collapse:collapse;border:none;font-size:14px;margin:0">
    {row("Face", d.face_score, d.face_contribution, face_z, face_ok)}
    {row("Voice", d.voice_score, d.voice_contribution, voice_z, voice_ok)}
  </table>
  {note_html}
</div>"""


# ---------------------------------------------------------------- actions
def enroll(customer_id, img, audio, store):
    customer_id = (customer_id or "").strip().upper()
    if not customer_id:
        return message("err", "Enter a customer ID."), None, store, gr.update()
    if img is None or audio is None:
        return message("err", "Both a selfie and a voice recording are needed."), None, store, gr.update()
    face, crop = kyc.face.embed_with_crop(to_bgr(img))
    voice = kyc.voice.embed(to_wav16k(audio))
    if face is None:
        return message("err", "No face detected. Face the camera in good light and retake."), None, store, gr.update()
    if voice is None:
        return message("err", "Recording too short. Speak for about 5 seconds."), None, store, gr.update()
    store = {**store, customer_id: {"template": BiometricTemplate(face, voice), "crop": to_rgb(crop)}}
    msg = message("ok", f"Customer <b>{customer_id}</b> onboarded. Face and voice templates stored "
                        f"(numbers only, for this session). Now go to <b>② Verify</b>.")
    return msg, to_rgb(crop), store, gr.update(choices=list(store), value=customer_id)


def verify(customer_id, tier_label, img, audio, store):
    if customer_id not in store:
        return message("err", "Choose an onboarded customer."), None, None
    if img is None and audio is None:
        return message("err", "Provide a selfie and/or a voice recording."), store[customer_id]["crop"], None
    face, crop = kyc.face.embed_with_crop(to_bgr(img)) if img is not None else (None, None)
    voice = kyc.voice.embed(to_wav16k(audio)) if audio is not None else None
    if face is None and voice is None:
        return message("err", "Neither a face nor usable speech was captured. Please retake."), \
            store[customer_id]["crop"], None
    d = kyc.verify_embeddings(customer_id, face, voice, TIERS[tier_label],
                              template=store[customer_id]["template"])
    return decision_html(d, face is not None, voice is not None), store[customer_id]["crop"], to_rgb(crop)


def enrolled_choices(store):
    return gr.update(choices=list(store))


# ---------------------------------------------------------------- layout
FACE_PCT = round(100 * kyc.fusion.face_weight)
ABOUT = f"""
### What happens behind the scenes
1. **Face.** The face is detected and its 5 landmarks are aligned. ArcFace turns it into
   512 numbers (an *embedding*).
2. **Voice.** ECAPA-TDNN turns the recording into 192 numbers describing the speaker's voice.
3. **Match.** Each is compared with the enrolled template (cosine similarity), rescaled
   to "how far above a typical impostor?", and combined as {FACE_PCT}% face + {100 - FACE_PCT}% voice.
4. **Decide.** The combined score is checked against the pass mark for the action's
   risk tier: stricter for transfers than for login.

### Results on public benchmarks (see the repository README)
Under low-quality mobile capture, fusing face and voice **roughly halved the errors**
of face alone. The equal error rate fell from 0.56% to 0.28%, and at a 1-in-1,000 fraud
tolerance, genuine customers blocked fell from 1.11% to 0.49%.

### Please note
- **This is a research demo, not a security product.** It has **no liveness
  detection**: a photo or a recording of the right person would pass. A real bank
  system needs anti-spoofing on both face and voice.
- The pass marks were calibrated on public benchmark data (LFW, LibriSpeech). Your own
  webcam and microphone may score differently.
- Nothing is saved. Templates live in this browser session's memory and disappear when
  you close the tab.
"""

with gr.Blocks(title="Biometric KYC Verification") as demo:
    store = gr.State(INITIAL_STORE)
    gr.Markdown("# Face + Voice Verification for Mobile Banking\n"
                "Onboard a customer once, then confirm it is really them at login or "
                "before a large transfer. Face and voice evidence are combined into one decision.")

    with gr.Tab("① Onboard"):
        gr.Markdown("Open an account: capture **one selfie** and **about 5 seconds of speech** "
                    "(for example, read: *\"I am opening a savings account with this bank today.\"*).")
        with gr.Row():
            en_img = gr.Image(label="Selfie", sources=["webcam", "upload"], type="numpy", height=320)
            with gr.Column():
                en_audio = gr.Audio(label="Voice sample (~5 s)", sources=["microphone", "upload"], type="numpy")
                en_id = gr.Textbox(label="Customer ID", value="CUST-0001")
                en_btn = gr.Button("Onboard customer", variant="primary")
        with gr.Row():
            en_msg = gr.HTML()
            en_crop = gr.Image(label="Aligned face used for the template", height=160, interactive=False)

    with gr.Tab("② Verify"):
        gr.Markdown("A login or transfer attempt: a **new selfie** and a **short phrase** "
                    "(for example, *\"I approve this transfer.\"*).")
        with gr.Row():
            ve_id = gr.Dropdown(label="Account being accessed", choices=list(INITIAL_STORE), value=SAMPLE_ID)
            ve_tier = gr.Radio(label="Action", choices=list(TIERS), value=list(TIERS)[1])
        with gr.Row():
            ve_img = gr.Image(label="Selfie", sources=["webcam", "upload"], type="numpy", height=320)
            with gr.Column():
                ve_audio = gr.Audio(label="Voice (2-5 s)", sources=["microphone", "upload"], type="numpy")
                ve_btn = gr.Button("Verify", variant="primary")
        gr.Examples(
            label="No webcam? Try the pre-enrolled sample customer (public benchmark data, low-quality mobile capture)",
            examples=[[SAMPLE_ID, list(TIERS)[1], str(EXAMPLES / "genuine_selfie.jpg"), str(EXAMPLES / "genuine_voice.wav")],
                      [SAMPLE_ID, list(TIERS)[1], str(EXAMPLES / "impostor_selfie.jpg"), str(EXAMPLES / "impostor_voice.wav")]],
            example_labels=["The real customer", "An impostor claiming the account"],
            inputs=[ve_id, ve_tier, ve_img, ve_audio],
        )
        ve_result = gr.HTML()
        with gr.Row():
            ve_enrolled = gr.Image(label="Enrolled (onboarding)", height=160, interactive=False)
            ve_probe = gr.Image(label="This attempt (aligned)", height=160, interactive=False)

    with gr.Tab("How it works"):
        gr.Markdown(ABOUT)

    en_btn.click(enroll, [en_id, en_img, en_audio, store], [en_msg, en_crop, store, ve_id])
    ve_btn.click(verify, [ve_id, ve_tier, ve_img, ve_audio, store], [ve_result, ve_enrolled, ve_probe])
    demo.load(enrolled_choices, store, ve_id)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
