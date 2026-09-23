"""Virtual ("chimeric") bank customers for multimodal evaluation.

No public dataset has both face photos and voice recordings of the same large set of
people, so - as is standard in multimodal biometrics research - each virtual customer
is one LFW face identity paired with one LibriSpeech speaker. This assumes face and
voice scores are independent given the identity, which holds well for these two
unrelated modalities.

Each customer has:
  * an enrollment template: 1 clean face photo + 1 clean utterance (onboarding)
  * probes: 9 other face photos + 9 other utterances (later login attempts); voice
    probes come from a different recording session than the enrollment utterance
    wherever the speaker has one.
"""
import random
from dataclasses import dataclass, field

from face_verification.lfw import identities
from voice_verification.librispeech import speakers

N_CUSTOMERS = 80
PROBES_PER_CUSTOMER = 9


@dataclass
class VirtualCustomer:
    customer_id: str
    face_identity: str
    speaker_id: str
    enrolled_face: str
    enrolled_voice: str
    probe_faces: list = field(default_factory=list)
    probe_voices: list = field(default_factory=list)


@dataclass
class Trial:
    claimed: str      # customer_id whose account is being accessed
    presenter: str    # customer_id of the person actually presenting
    face_probe: str
    voice_probe: str

    @property
    def genuine(self) -> bool:
        return self.claimed == self.presenter


def build_customers(seed: int = 0) -> list:
    rng = random.Random(seed)
    faces = identities(min_images=PROBES_PER_CUSTOMER + 1)
    face_ids = rng.sample(sorted(faces), N_CUSTOMERS)
    spk = speakers()
    spk_ids = rng.sample(sorted(spk), N_CUSTOMERS)

    customers = []
    for i, (fid, sid) in enumerate(zip(face_ids, spk_ids)):
        imgs = rng.sample(faces[fid], PROBES_PER_CUSTOMER + 1)
        utts = spk[sid]["utts"]
        enroll_chap, enroll_utt = rng.choice(utts)
        other_session = [u for c, u in utts if c != enroll_chap]
        pool = other_session if len(other_session) >= PROBES_PER_CUSTOMER else [u for _, u in utts if u != enroll_utt]
        customers.append(VirtualCustomer(
            customer_id=f"CUST-{i:04d}",
            face_identity=fid,
            speaker_id=sid,
            enrolled_face=imgs[0],
            enrolled_voice=enroll_utt,
            probe_faces=imgs[1:],
            probe_voices=rng.sample(pool, PROBES_PER_CUSTOMER),
        ))
    return customers


def split_customers(customers: list, seed: int = 0):
    """Disjoint dev (tune fusion) / test (report) customer sets, 50/50."""
    shuffled = customers[:]
    random.Random(seed).shuffle(shuffled)
    half = len(shuffled) // 2
    return shuffled[:half], shuffled[half:]


def make_trials(customers: list, impostor_per_pair: int = 10, seed: int = 0) -> list:
    """Genuine: every (face probe, voice probe) combination of the customer themself.
    Impostor: for each claimed account and every other customer as the attacker,
    `impostor_per_pair` random (face probe, voice probe) combinations of the attacker.
    """
    rng = random.Random(seed)
    trials = []
    for c in customers:
        trials += [Trial(c.customer_id, c.customer_id, f, v) for f in c.probe_faces for v in c.probe_voices]
        for attacker in customers:
            if attacker is c:
                continue
            for _ in range(impostor_per_pair):
                trials.append(Trial(c.customer_id, attacker.customer_id,
                                    rng.choice(attacker.probe_faces), rng.choice(attacker.probe_voices)))
    return trials
