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

from dcond.caer import CLASSES, DEFAULT_CAER_ROOT, ENCODER_TAGS, MANIFEST_CSV, SPLITS, embedding_path, load_manifest
from scripts.caer_face_crops import choose, load_detector, square_crop

CLIP_MODEL = "openai/clip-vit-large-patch14"


def frame_positions(n: int, k: int) -> list[int]:
    return [min(int((j + 1) / (k + 1) * n), n - 1) for j in range(k)]


def read_frames(path: Path, idx: list[int]) -> list[np.ndarray]:
    wanted = set(idx)
    got = {}
    cap = cv2.VideoCapture(str(path))
    i, last = 0, max(idx)
    while i <= last:
        if not cap.grab():
            break
        if i in wanted:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                raise RuntimeError(f"retrieve() failed at frame {i}")
            got[i] = frame
        i += 1
    cap.release()
    missing = wanted - set(got)
    if missing:
        raise RuntimeError(f"stream ended before frames {sorted(missing)[:3]} (read {i})")
    return [got[j] for j in idx]


class ClipFrames(torch.utils.data.Dataset):
    def __init__(self, rows: pd.DataFrame, caer_root: Path, k: int):
        self.rows, self.root, self.k = rows.reset_index(drop=True), caer_root, k

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        cv2.setNumThreads(1)
        r = self.rows.iloc[i]
        idx = frame_positions(int(r["n_frames_actual"]), self.k)
        try:
            return i, idx, read_frames(self.root / r["video_relpath"], idx), ""
        except Exception as e:
            return i, idx, None, f"{type(e).__name__}: {e}"


class Encoders:
    def __init__(self, device: str):
        from dcond.embeddings import ClipEncoder, _extract_features
        from dcond.embeddings_emotion import FerVitEncoder

        self.device, self._extract = device, _extract_features
        self.clip = ClipEncoder(CLIP_MODEL, device=device)
        self.fer = FerVitEncoder(device=device)

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device == "cuda")

    @torch.no_grad()
    def clip_embed(self, images: list, bs: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(images), bs):
            inputs = self.clip.processor(images=images[i : i + bs], return_tensors="pt").to(self.device)
            with self._autocast():
                f = self._extract(self.clip.model.get_image_features(**inputs)).float()
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)

    @torch.no_grad()
    def fer_logits(self, images: list, bs: int = 128) -> np.ndarray:
        out = []
        for i in range(0, len(images), bs):
            inputs = self.fer.processor(images=images[i : i + bs], return_tensors="pt").to(self.device)
            with self._autocast():
                cls = self.fer.model.vit(**inputs).last_hidden_state[:, 0]
                logits = self.fer.model.classifier(cls).float()
            out.append(self.fer.logits_for(logits.cpu().numpy(), CLASSES))
        return np.concatenate(out) if out else np.zeros((0, len(CLASSES)), np.float32)


def process_shard(rows: pd.DataFrame, args, enc: Encoders, detector) -> tuple[dict, list]:
    n, k = len(rows), args.k
    arr = dict(frame_idx=np.zeros((n, k), np.int32), clip_full=np.zeros((n, k, 768), np.float16),
               clip_face=np.zeros((n, k, 768), np.float16), face_ok=np.zeros((n, k), bool),
               face_conf=np.full((n, k), np.nan, np.float32), face_area_frac=np.full((n, k), np.nan, np.float32),
               fer_logits=np.full((n, k, len(CLASSES)), np.nan, np.float32))
    ok_clip = np.ones(n, bool)
    failures = []
    loader = torch.utils.data.DataLoader(ClipFrames(rows, args.caer_root, k), batch_size=args.clips_per_batch,
                                         num_workers=args.workers, collate_fn=lambda b: b)
    for batch in loader:
        flat_bgr, where = [], []
        for i, idx, frames, err in batch:
            arr["frame_idx"][i] = idx
            if frames is None:
                ok_clip[i] = False
                failures.append({"video_relpath": rows.iloc[i]["video_relpath"], "error": err})
                continue
            for j, f in enumerate(frames):
                flat_bgr.append(f)
                where.append((i, j))
        if not flat_bgr:
            continue
        rgb = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in flat_bgr]
        full = enc.clip_embed(rgb)
        crops, crop_where = [], []
        for s in range(0, len(flat_bgr), 64):
            for (i, j), img, res in zip(where[s : s + 64], rgb[s : s + 64],
                                        detector.predict(flat_bgr[s : s + 64], conf=args.det_conf, imgsz=args.imgsz,
                                                         verbose=False, device=args.det_device or None)):
                h, w = res.orig_shape
                dets = []
                if len(res.boxes):
                    for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        if c >= args.min_conf:
                            dets.append({"conf": float(c), "w": float(x2 - x1), "h": float(y2 - y1),
                                         "cx": float((x1 + x2) / 2), "cy": float((y1 + y2) / 2)})
                pick = choose(dets, "largest", w, h)
                if pick is not None:
                    crops.append(square_crop(img, pick, args.margin))
                    crop_where.append((i, j))
                    arr["face_conf"][i, j] = pick["conf"]
                    arr["face_area_frac"][i, j] = pick["w"] * pick["h"] / (w * h)
        for (i, j), e in zip(where, full):
            arr["clip_full"][i, j] = e
        if crops:
            face, fer = enc.clip_embed(crops), enc.fer_logits(crops)
            for (i, j), e, lg in zip(crop_where, face, fer):
                arr["clip_face"][i, j] = e
                arr["fer_logits"][i, j] = lg
                arr["face_ok"][i, j] = True
    keep = ok_clip
    out = {key: v[keep] for key, v in arr.items()}
    out.update(video_relpath=rows["video_relpath"].to_numpy(dtype=object)[keep], label=rows["label"].to_numpy()[keep],
               n_frames=rows["n_frames_actual"].to_numpy().astype(np.int32)[keep])
    return out, failures


def p50_check(out_file: Path, split: str) -> None:
    ref = embedding_path(split, "p50", ENCODER_TAGS["clip"], PROJECT_ROOT / "data" / "caer" / "embeddings")
    if not ref.exists():
        return
    with np.load(out_file, allow_pickle=True) as f, np.load(ref, allow_pickle=True) as r:
        k = f["clip_full"].shape[1]
        if k % 2 == 0:
            return
        pos = {v: i for i, v in enumerate(r["video_relpath"])}
        ids = [i for i, v in enumerate(f["video_relpath"]) if v in pos]
        mine = f["clip_full"][ids, k // 2].astype(np.float32)
        theirs = r["image_embeds"][[pos[f["video_relpath"][i]] for i in ids]]
        cos = (mine * theirs).sum(1)
    print(f"  check: middle frame vs cached p50 CLIP embedding, cosine median {np.median(cos):.4f}, "
          f"min {cos.min():.4f} (n={len(cos)}; ~1.0 expected, fp16 vs fp32)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caer-root", type=Path, default=DEFAULT_CAER_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV, help="stage-1 full-frame manifest")
    parser.add_argument("--out", type=Path, default=None, help="default data/caer_multi (data/caer_multi_smoke with --limit)")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--k", type=int, default=15, help="frames per clip (odd K includes the p50 frame)")
    parser.add_argument("--limit-per-class", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=500, help="clips per shard file (resume granularity)")
    parser.add_argument("--clips-per-batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--min-conf", type=float, default=0.5)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-device", default="", help="face detector device ('' = auto, 'cpu' if torchvision lacks CUDA ops)")
    args = parser.parse_args()
    args.caer_root = args.caer_root.expanduser()
    out = args.out or PROJECT_ROOT / "data" / ("caer_multi_smoke" if args.limit_per_class else "caer_multi")
    (out / "shards").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = load_manifest(args.manifest)
    enc, detector = None, None
    for split in args.splits:
        rows = manifest[manifest["split"] == split]
        if args.limit_per_class:
            rows = rows.groupby("class_name", sort=False).head(args.limit_per_class)
        rows = rows.reset_index(drop=True)
        final = out / f"caer_{split}_k{args.k}.npz"
        if final.exists():
            print(f"{final.name}: exists, skipped")
            continue
        shards, failures = [], []
        n_shards = (len(rows) + args.shard_size - 1) // args.shard_size
        for s in tqdm(range(n_shards), desc=f"{split} ({len(rows)} clips, K={args.k})"):
            path = out / "shards" / f"{split}_k{args.k}_{s:03d}.npz"
            part = rows.iloc[s * args.shard_size : (s + 1) * args.shard_size]
            if not path.exists():
                if enc is None:
                    enc, detector = Encoders(device), load_detector()
                data, fails = process_shard(part, args, enc, detector)
                np.savez(path, **data)
                pd.DataFrame(fails, columns=["video_relpath", "error"]).to_csv(path.with_suffix(".failures.csv"), index=False)
            shards.append(path)
            failures.append(pd.read_csv(path.with_suffix(".failures.csv")))
        parts = [dict(np.load(p, allow_pickle=True)) for p in shards]
        merged = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        np.savez(final, **merged)
        fails = pd.concat(failures)
        fails.to_csv(out / f"failures_{split}.csv", index=False)
        fo = merged["face_ok"]
        print(f"{final.name}: {len(merged['label'])} clips ({len(fails)} failed), face found in {fo.mean():.1%} of frames, "
              f"clips with >= 1 face {fo.any(1).mean():.1%}")
        p50_check(final, split)


if __name__ == "__main__":
    main()
