import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import DEFAULT_CAER_ROOT, MANIFEST_CSV, SPLITS, load_manifest
from scripts.caer_face_crops import choose, load_detector, square_crop
from scripts.caer_multiframe_embeddings import ClipFrames


def resize_short(img: Image.Image, short: int) -> Image.Image:
    w, h = img.size
    s = short / min(w, h)
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def process_shard(rows: pd.DataFrame, args, detector, out: Path) -> list[dict]:
    loader = torch.utils.data.DataLoader(ClipFrames(rows, args.caer_root, args.k), batch_size=args.clips_per_batch,
                                         num_workers=args.workers, collate_fn=lambda b: b)
    index = []
    for batch in loader:
        flat, where = [], []
        for i, idx, frames, err in batch:
            r = rows.iloc[i]
            if frames is None:
                index.append({"video_relpath": r["video_relpath"], "split": r["split"], "label": int(r["label"]), "j": -1,
                              "frame_idx": -1, "full": "", "face": "", "face_conf": float("nan"), "error": err})
                continue
            for j, f in enumerate(frames):
                flat.append(f)
                where.append((r, j, idx[j]))
        for s in range(0, len(flat), 64):
            chunk = flat[s : s + 64]
            for (r, j, fi), bgr, res in zip(where[s : s + 64], chunk,
                                            detector.predict(chunk, conf=args.det_conf, imgsz=args.imgsz, verbose=False,
                                                             device=args.det_device or None)):
                img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                stem = Path(r["video_relpath"]).stem
                rel = f"{r['split']}/{r['class_name']}/{stem}_f{j:02d}.jpg"
                (out / "full" / rel).parent.mkdir(parents=True, exist_ok=True)
                resize_short(img, args.full_short).save(out / "full" / rel, quality=args.quality)
                h, w = res.orig_shape
                dets = []
                if len(res.boxes):
                    for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        if c >= args.min_conf:
                            dets.append({"conf": float(c), "w": float(x2 - x1), "h": float(y2 - y1),
                                         "cx": float((x1 + x2) / 2), "cy": float((y1 + y2) / 2)})
                pick = choose(dets, "largest", w, h)
                face_rel, conf = "", float("nan")
                if pick is not None:
                    (out / "face" / rel).parent.mkdir(parents=True, exist_ok=True)
                    square_crop(img, pick, args.margin).resize((args.face_size, args.face_size), Image.BICUBIC) \
                        .save(out / "face" / rel, quality=args.quality)
                    face_rel, conf = rel, pick["conf"]
                index.append({"video_relpath": r["video_relpath"], "split": r["split"], "label": int(r["label"]), "j": j,
                              "frame_idx": int(fi), "full": rel, "face": face_rel, "face_conf": conf, "error": ""})
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caer-root", type=Path, default=DEFAULT_CAER_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--out", type=Path, default=None, help="default data/caer_multi_img (…_smoke with --limit-per-class)")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--clips-per-batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--full-short", type=int, default=256)
    parser.add_argument("--face-size", type=int, default=256)
    parser.add_argument("--quality", type=int, default=92)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-device", default="")
    args = parser.parse_args()
    args.caer_root = args.caer_root.expanduser()
    out = args.out or PROJECT_ROOT / "data" / ("caer_multi_img_smoke" if args.limit_per_class else "caer_multi_img")
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
        for s in tqdm(range(n_shards), desc=f"{split} ({len(rows)} clips, K={args.k})"):
            path = out / "shards" / f"{split}_k{args.k}_{s:03d}.csv"
            if not path.exists():
                detector = detector or load_detector()
                pd.DataFrame(process_shard(rows.iloc[s * args.shard_size : (s + 1) * args.shard_size], args, detector, out)).to_csv(path, index=False)
            parts.append(pd.read_csv(path, keep_default_na=False))
        idx = pd.concat(parts, ignore_index=True)
        idx.to_csv(final, index=False)
        ok = idx[idx["j"] >= 0]
        print(f"{final.name}: {idx['video_relpath'].nunique()} clips, {(idx['j'] < 0).sum()} failed, "
              f"{len(ok)} frames, face in {(ok['face'] != '').mean():.1%}")


if __name__ == "__main__":
    main()
