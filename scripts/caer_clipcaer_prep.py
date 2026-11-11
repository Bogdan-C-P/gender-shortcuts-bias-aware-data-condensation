import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import DEFAULT_CAER_ROOT, MANIFEST_CSV, SPLITS, load_manifest
from scripts.caer_multiframe_embeddings import read_frames
from scripts.caer_video_frames import sample_indices

VIDEO_DIR = PROJECT_ROOT / "data" / "caer_video"


def work(job):
    rel, n_actual, fps, n_expected, out_dir, quality = job
    out_dir = Path(out_dir)
    idx = sample_indices(n_actual, fps, 10.0, 96)
    if len(idx) != n_expected:
        return rel, None, f"index mismatch: {len(idx)} vs caer_video {n_expected}"
    split, cls, name = rel.split("/")
    stem = Path(name).stem
    boxes = np.load(VIDEO_DIR / "boxes" / split / cls / f"{stem}.npy")
    if len(boxes) != len(idx):
        return rel, None, f"box rows {len(boxes)} vs frames {len(idx)}"
    d = out_dir / "frames" / split / cls / stem
    d.mkdir(parents=True, exist_ok=True)
    try:
        frames = read_frames(DEFAULT_CAER_ROOT / rel, idx)
    except Exception as e:
        return rel, None, f"{type(e).__name__}: {e}"
    fb = {}
    for j, bgr in enumerate(frames):
        H, W = bgr.shape[:2]
        cv2.imwrite(str(d / f"{j:03d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        cx, cy, w, h = boxes[j, :4] * [W, H, W, H]
        fb[f"{j:03d}.jpg"] = [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)]
    return rel, (str(d), len(frames), fb), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit-per-class", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--quality", type=int, default=95)
    a = ap.parse_args()
    out = (a.out or PROJECT_ROOT / "data" / ("caer_clipcaer_smoke" if a.limit_per_class else "caer_clipcaer")).resolve()
    out.mkdir(parents=True, exist_ok=True)
    man = load_manifest(MANIFEST_CSV).set_index("video_relpath")
    face_all, errors = {}, []
    for split in SPLITS:
        idx = pd.read_csv(VIDEO_DIR / f"index_{split}.csv", keep_default_na=False)
        idx = idx[idx["n_frames"].astype(int) > 0]
        if a.limit_per_class:
            idx = idx.groupby("label", sort=False).head(a.limit_per_class)
        jobs = [(r.video_relpath, int(man.at[r.video_relpath, "n_frames_actual"]), float(man.at[r.video_relpath, "fps"]),
                 int(r.n_frames), str(out), a.quality) for r in idx.itertuples()]
        labels = dict(zip(idx["video_relpath"], idx["label"].astype(int)))
        lines = []
        with Pool(a.workers) as pool:
            for rel, res, err in tqdm(pool.imap(work, jobs, chunksize=4), total=len(jobs), desc=split, mininterval=20):
                if res is None:
                    errors.append((rel, err))
                    continue
                d, n, fb = res
                lines.append((rel, f"{d} {n} {labels[rel] + 1}"))
                face_all[d] = fb
        (out / f"{split}.txt").write_text("\n".join(l for _, l in sorted(lines)) + "\n")
        print(f"{split}: {len(lines)} clips written", flush=True)
    (out / "face_boxes.json").write_text(json.dumps(face_all))
    (out / "body_boxes.json").write_text("{}")
    print(f"errors: {len(errors)}", *errors[:10], sep="\n")


if __name__ == "__main__":
    main()
