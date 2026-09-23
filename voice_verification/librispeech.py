"""LibriSpeech speaker-verification protocol (test-clean + dev-clean, 80 speakers).

LibriSpeech has no official verification trial list, so one is generated
deterministically (seed 0):
  * genuine trials  - two utterances of the same speaker, taken from *different
                      recording sessions (chapters)* wherever possible, mimicking
                      "enrolled at onboarding, verified weeks later"
  * impostor trials - enrolled utterance vs. a *same-gender* different speaker,
                      the harder and more realistic attack (a fraudster claiming a
                      victim's account is unlikely to pick a different gender)
"""
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRI = ROOT / "data" / "LibriSpeech"
SUBSETS = ("test-clean", "dev-clean")


@dataclass
class VoicePair:
    enrolled: str
    probe: str
    genuine: bool


def utterance_path(key: str) -> Path:
    return LIBRI / key


def speakers() -> dict:
    """{speaker_id: {"gender": "M"/"F", "utts": [(chapter, key), ...]}}"""
    gender = {}
    for line in (LIBRI / "SPEAKERS.TXT").read_text(encoding="utf-8").splitlines():
        if line.startswith(";"):
            continue
        parts = [p.strip() for p in line.split("|")]
        gender[parts[0]] = parts[1]
    out = defaultdict(lambda: {"utts": []})
    for subset in SUBSETS:
        for f in sorted((LIBRI / subset).glob("*/*/*.flac")):
            spk, chap = f.parent.parent.name, f.parent.name
            out[spk]["gender"] = gender[spk]
            out[spk]["utts"].append((chap, f.relative_to(LIBRI).as_posix()))
    return dict(out)


def make_trials(per_speaker: int = 100, seed: int = 0) -> list:
    rng = random.Random(seed)
    spk = speakers()
    ids = sorted(spk)
    by_gender = defaultdict(list)
    for s in ids:
        by_gender[spk[s]["gender"]].append(s)
    trials = []
    for s in ids:
        utts = spk[s]["utts"]
        multi_session = len({c for c, _ in utts}) > 1
        for _ in range(per_speaker):
            (c1, a) = rng.choice(utts)
            pool = [u for c, u in utts if (c != c1 if multi_session else u != a)]
            trials.append(VoicePair(a, rng.choice(pool), True))
        others = [o for o in by_gender[spk[s]["gender"]] if o != s]
        for _ in range(per_speaker):
            a = rng.choice(utts)[1]
            b = rng.choice(spk[rng.choice(others)]["utts"])[1]
            trials.append(VoicePair(a, b, False))
    return trials
