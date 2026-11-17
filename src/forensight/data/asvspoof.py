"""ASVspoof protocol / key file parsing.

ASVspoof 2019 LA protocol lines look like:
    LA_0079 LA_T_1138215 - - bonafide
    LA_0079 LA_T_1271820 - A01 spoof

ASVspoof 2021 DF keys (trial_metadata.txt) have more columns, e.g.:
    LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder - - - -

Both have the speaker in column 0 and the utterance id in column 1. The label is
the token "bonafide" or "spoof", which we locate by value rather than by position,
so one parser handles both formats. The attack id (A01..A19) is kept when present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from forensight import FAKE, REAL

_ATTACK = re.compile(r"^A\d{2}$")


@dataclass(frozen=True)
class Utterance:
    speaker: str
    utt_id: str
    label: int
    attack: str | None
    subset: str | None = None  # 2021 keys: "progress" / "eval" / "hidden_track"


def parse_line(line: str) -> Utterance | None:
    tok = line.split()
    if len(tok) < 3:
        return None
    if "bonafide" in tok:
        label = REAL
    elif "spoof" in tok:
        label = FAKE
    else:
        raise ValueError(f"No bonafide/spoof label in protocol line: {line!r}")
    attack = next((t for t in tok[2:] if _ATTACK.match(t)), None)
    subset = next((t for t in tok if t in {"progress", "eval", "hidden_track"}), None)
    return Utterance(tok[0], tok[1], label, attack, subset)


def read_protocol(path: str | Path) -> list[Utterance]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                u = parse_line(line)
                if u is not None:
                    out.append(u)
    return out


def audio_path(audio_dir: str | Path, utt_id: str, ext: str = ".flac") -> Path:
    return Path(audio_dir) / f"{utt_id}{ext}"
