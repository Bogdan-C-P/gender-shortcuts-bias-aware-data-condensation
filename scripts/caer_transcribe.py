import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from faster_whisper import WhisperModel
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAER_ROOT = Path(os.environ.get("CAER_ROOT", PROJECT_ROOT / "data" / "caer_raw" / "CAER")).expanduser()
OUT_DIR = PROJECT_ROOT / "data" / "caer_asr"
SPLITS = ["train", "validation", "test"]


def load_audio(path: Path) -> np.ndarray:
    o = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
        capture_output=True,
    )
    return np.frombuffer(o.stdout, dtype=np.int16).astype(np.float32) / 32768


def list_videos(root: Path, splits: list[str], limit_per_class: int) -> list[str]:
    rels = []
    for split in splits:
        for cdir in sorted(p for p in (root / split).iterdir() if p.is_dir()):
            vids = sorted(cdir.glob("*.avi"))
            if limit_per_class:
                vids = vids[:limit_per_class]
            rels += [v.relative_to(root).as_posix() for v in vids]
    return rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caer-root", type=Path, default=CAER_ROOT)
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="concurrent transcribe calls sharing one GPU model")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    a.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = a.out_dir / "transcripts.jsonl"
    if a.force and jsonl.exists():
        jsonl.unlink()
    done = set()
    if jsonl.exists():
        done = {json.loads(l)["video_relpath"] for l in jsonl.open() if l.strip()}
    todo = [r for r in list_videos(a.caer_root, a.splits, a.limit_per_class) if r not in done]
    print(f"{len(done)} already done, {len(todo)} to do", flush=True)

    model = WhisperModel(a.model, device="cuda", compute_type="float16", num_workers=a.workers)

    def work(rel: str) -> dict:
        try:
            audio = load_audio(a.caer_root / rel)
            segs, _ = model.transcribe(
                audio, language="en", beam_size=5, vad_filter=False, condition_on_previous_text=False
            )
            segs = list(segs)
            text = " ".join(s.text.strip() for s in segs).strip()
            return dict(
                video_relpath=rel, text=text, n_words=len(text.split()), audio_sec=round(len(audio) / 16000, 3),
                no_speech_prob=round(max((s.no_speech_prob for s in segs), default=1.0), 4),
                avg_logprob=round(min((s.avg_logprob for s in segs), default=-9.0), 4),
                n_segments=len(segs), status="ok", error="",
            )
        except Exception as e:
            return dict(video_relpath=rel, text="", n_words=0, audio_sec=0, no_speech_prob=1.0,
                        avg_logprob=-9.0, n_segments=0, status="error", error=repr(e))

    t0 = time.time()
    with jsonl.open("a") as f, ThreadPoolExecutor(a.workers) as ex:
        for r in tqdm(ex.map(work, todo), total=len(todo), mininterval=30):
            f.write(json.dumps(r) + "\n")
            f.flush()
    print(f"transcribed {len(todo)} in {time.time() - t0:.0f}s", flush=True)

    df = pd.read_json(jsonl, lines=True)
    df["text"] = df.text.fillna("")
    df["usable"] = (df.status == "ok") & (df.no_speech_prob < 0.5) & (df.avg_logprob > -1.0) & (df.n_words >= 3)
    df.to_csv(a.out_dir / "transcripts.csv", index=False)
    print(f"{len(df)} rows, {int(df.usable.sum())} usable ({df.usable.mean():.1%}), "
          f"{int((df.status != 'ok').sum())} errors", flush=True)


if __name__ == "__main__":
    sys.exit(main())
