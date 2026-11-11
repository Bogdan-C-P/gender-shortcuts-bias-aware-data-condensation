import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import DEFAULT_CAER_ROOT, MANIFEST_CSV, SPLITS, load_manifest
from scripts.caer_face_crops import choose, load_detector, square_crop
from scripts.caer_multiframe_embeddings import read_frames


def sample_indices(n: int, fps: float, rate: float, max_frames: int) -> list[int]:
    fps = fps if fps and np.isfinite(fps) and fps > 0 else 25.0
    idx = sorted({min(int(round(i * fps / rate)), n - 1) for i in range(int(np.ceil(n * rate / fps)))})
    if len(idx) > max_frames:
        s = (len(idx) - max_frames) // 2
        idx = idx[s : s + max_frames]
    return idx


class Clips(torch.utils.data.Dataset):
    def __init__(self, rows: pd.DataFrame, root: Path, rate: float, max_frames: int):
        self.rows, self.root, self.rate, self.max_frames = rows.reset_index(drop=True), root, rate, max_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        r = self.rows.iloc[i]
        fps = float(r["fps"]) if "fps" in r and pd.notna(r["fps"]) else 25.0
        idx = sample_indices(int(r["n_frames_actual"]), fps, self.rate, self.max_frames)
        try:
            return i, fps, read_frames(self.root / r["video_relpath"], idx), ""
        except Exception as e:
            return i, fps, None, f"{type(e).__name__}: {e}"


def collate(batch):
    return batch


def interpolate_boxes(dets: dict[int, dict], n: int) -> np.ndarray:
    out = np.full((n, 4), np.nan)
    if not dets:
        return out
    keys = np.array(sorted(dets))
    vals = np.array([[dets[k]["cx"], dets[k]["cy"], dets[k]["w"], dets[k]["h"]] for k in keys])
    for c in range(4):
        out[:, c] = np.interp(np.arange(n), keys, vals[:, c])
    return out


def process_shard(rows: pd.DataFrame, args, detector, out: Path) -> list[dict]:
    loader = torch.utils.data.DataLoader(Clips(rows, args.caer_root, args.rate, args.max_frames),
                                         batch_size=args.clips_per_batch, num_workers=args.workers, collate_fn=collate)
    index = []
    for batch in loader:
        todo = []
        for i, fps, frames, err in batch:
            r = rows.iloc[i]
            if frames is None:
                index.append({"video_relpath": r["video_relpath"], "split": r["split"], "label": int(r["label"]),
                              "n_frames": 0, "fps": fps, "face_detected_frac": 0.0, "no_face": True, "error": err})
            else:
                todo.append((r, fps, frames))
        jobs = [(c, j) for c, (_, _, fr) in enumerate(todo) for j in sorted(set(range(0, len(fr), args.det_stride)) | {len(fr) - 1})]
        found = [dict() for _ in todo]
        for s in range(0, len(jobs), 64):
            chunk = jobs[s : s + 64]
            imgs = [todo[c][2][j] for c, j in chunk]
            for (c, j), res in zip(chunk, detector.predict(imgs, conf=args.det_conf, imgsz=args.imgsz, verbose=False,
                                                           device=args.det_device or None)):
                h, w = res.orig_shape
                dets = []
                if len(res.boxes):
                    for (x1, y1, x2, y2), cf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        if cf >= args.min_conf:
                            dets.append({"conf": float(cf), "w": float(x2 - x1), "h": float(y2 - y1),
                                         "cx": float((x1 + x2) / 2), "cy": float((y1 + y2) / 2)})
                pick = choose(dets, "largest", w, h)
                if pick is not None:
                    found[c][j] = pick
        for (r, fps, frames), dets in zip(todo, found):
            stem = Path(r["video_relpath"]).stem
            rel = Path(r["split"]) / r["class_name"] / stem
            (out / "full" / rel).mkdir(parents=True, exist_ok=True)
            (out / "face" / rel).mkdir(parents=True, exist_ok=True)
            n = len(frames)
            H, W = frames[0].shape[:2]
            boxes = interpolate_boxes(dets, n)
            no_face = bool(np.isnan(boxes).all())
            if no_face:
                boxes[:] = [W / 2, H / 2, min(W, H) / (1 + 2 * args.margin), min(W, H) / (1 + 2 * args.margin)]
            for j, bgr in enumerate(frames):
                img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                sc = args.full_short / min(W, H)
                img.resize((round(W * sc), round(H * sc)), Image.BICUBIC).save(out / "full" / rel / f"{j:03d}.jpg", quality=args.quality)
                cx, cy, bw, bh = boxes[j]
                square_crop(img, {"cx": cx, "cy": cy, "w": bw, "h": bh}, args.margin) \
                    .resize((args.face_size, args.face_size), Image.BICUBIC).save(out / "face" / rel / f"{j:03d}.jpg", quality=args.quality)
            det_flag = np.array([1.0 if j in dets else 0.0 for j in range(n)])
            (out / "boxes" / rel.parent).mkdir(parents=True, exist_ok=True)
            np.save(out / "boxes" / rel.parent / f"{stem}.npy",
                    np.column_stack([boxes / [W, H, W, H], det_flag]).astype(np.float32))
            index.append({"video_relpath": r["video_relpath"], "split": r["split"], "label": int(r["label"]), "n_frames": n,
                          "fps": fps, "face_detected_frac": len(dets) / max(1, len(set(range(0, n, args.det_stride)) | {n - 1})),
                          "no_face": no_face, "error": ""})
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caer-root", type=Path, default=DEFAULT_CAER_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--out", type=Path, default=None, help="default data/caer_video (…_smoke with --limit-per-class)")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--rate", type=float, default=10.0, help="sampled frames per second")
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=400)
    parser.add_argument("--clips-per-batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--det-stride", type=int, default=3)
    parser.add_argument("--full-short", type=int, default=128)
    parser.add_argument("--face-size", type=int, default=128)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-device", default="")
    args = parser.parse_args()
    args.caer_root = args.caer_root.expanduser()
    out = args.out or PROJECT_ROOT / "data" / ("caer_video_smoke" if args.limit_per_class else "caer_video")
    (out / "shards").mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    detector = None
    for split in args.splits:
        rows = manifest[manifest["split"] == split]
        if args.limit_per_class:
            rows = rows.groupby("class_name", sort=False).head(args.limit_per_class)
        rows = rows.reset_index(drop=True)
        final = out / f"index_{split}.csv"
        if final.exists():
            print(f"{final.name}: exists, skipped")
            continue
        parts = []
        n_shards = (len(rows) + args.shard_size - 1) // args.shard_size
        for s in tqdm(range(n_shards), desc=f"{split} ({len(rows)} clips, {args.rate:g} fps)"):
            path = out / "shards" / f"{split}_{s:03d}.csv"
            if not path.exists():
                detector = detector or load_detector()
                pd.DataFrame(process_shard(rows.iloc[s * args.shard_size : (s + 1) * args.shard_size], args, detector, out)) \
                    .to_csv(path, index=False)
            parts.append(pd.read_csv(path, keep_default_na=False))
        idx = pd.concat(parts, ignore_index=True)
        idx.to_csv(final, index=False)
        ok = idx[idx["n_frames"].astype(int) > 0]
        print(f"{final.name}: {len(idx)} clips, {(idx['n_frames'].astype(int) == 0).sum()} failed, frames per clip "
              f"median {ok['n_frames'].astype(int).median():.0f} (min {ok['n_frames'].astype(int).min()}), "
              f"no face {ok['no_face'].astype(str).eq('True').mean():.1%}, face found on "
              f"{ok['face_detected_frac'].astype(float).mean():.1%} of the detector's frames")


if __name__ == "__main__":
    main()
