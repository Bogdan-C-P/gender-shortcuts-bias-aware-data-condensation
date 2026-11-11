import json
import re
from pathlib import Path

import pandas as pd

from dcond.caer import CLASSES, PROJECT_ROOT, SPLITS

VLM_DIR = PROJECT_ROOT / "experiments" / "results" / "caer05_vlm" / "Qwen3-VL-8B-Instruct-AWQ-4bit"
EMOTIONS = [c.lower() for c in CLASSES]
AGE_PATTERNS = [("middle-aged", r"middle"), ("elderly", r"elder|older|senior|\bold\b"), ("teen", r"teen"),
                ("child", r"child|\bkid"), ("young adult", r"young"), ("adult", r"adult")]


def _person_text(mp) -> tuple[str, str]:
    if isinstance(mp, dict):
        g = mp.get("apparent_gender") or mp.get("apparent gender") or mp.get("gender") or ""
        a = mp.get("apparent_age_group") or mp.get("apparent age group") or mp.get("age") or ""
        return str(g), str(a)
    return str(mp), str(mp)


def _gender(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(female|woman|girl|lady)\b", t):
        return "female"
    if re.search(r"\b(male|man|boy|gentleman)\b", t):
        return "male"
    return "unclear"


def _age(text: str) -> str:
    t = text.lower()
    for name, pat in AGE_PATTERNS:
        if re.search(pat, t):
            return name
    return "unclear"


def load_descriptions(variant: str = "full", vlm_dir: Path = VLM_DIR, frame: str = "p50") -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        path = Path(vlm_dir) / f"descriptions_{split}_{frame}{'_face' if variant == 'face' else ''}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} — run scripts/caer_vlm_describe.py first")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            p = d["parsed"] or {}
            g_txt, a_txt = _person_text(p.get("main_person", ""))
            emo = str(p.get("emotion", "")).strip().lower()
            issues = p.get("image_issues") or []
            people = p.get("people")
            rows.append(dict(
                split=split, video_relpath=d["video_relpath"], label=d["class_name"].lower(),
                emotion=emo if emo in EMOTIONS else "unclear",
                emotion_confidence=str(p.get("emotion_confidence", "")).strip().lower(),
                expression=str(p.get("facial_expression", "")).strip().lower(),
                setting=str(p.get("setting", "")).strip().lower(),
                action=str(p.get("action", "")).strip().lower(),
                gender=_gender(g_txt), age=_age(a_txt),
                people=int(people) if isinstance(people, (int, float)) else -1,
                has_text=bool(str(p.get("on_screen_text", "")).strip()),
                issues=[str(i).strip().lower() for i in issues] if isinstance(issues, list) else [],
                caption=str(p.get("caption", "")), parsed_ok=bool(d["parsed"])))
    return pd.DataFrame(rows)
