import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES, FRAME_POSITIONS, FRAMES_DIR, MANIFEST_CSV, SPLITS, load_manifest
from experiments.caer01_inspect import contact_sheet

MODEL_REPO = "AdamCodd/YOLOv11n-face-detection"
MODEL_REVISION = "0e97fea5eacb1460b35725c929d813c7095841b5"
SMALL_FACE_PX = 32


def load_detector():
    from huggingface_hub import hf_hub_download
    from ultralytics import YOLO

    return YOLO(hf_hub_download(repo_id=MODEL_REPO, filename="model.pt", revision=MODEL_REVISION))


def choose(dets: list[dict], rule: str, img_w: int, img_h: int) -> dict | None:
    if not dets:
        return None
    if rule == "largest":
        return max(dets, key=lambda d: (d["w"] * d["h"], d["conf"]))
    if rule == "confident":
        return max(dets, key=lambda d: (d["conf"], d["w"] * d["h"]))
    if rule == "central":
        return min(dets, key=lambda d: ((d["cx"] - img_w / 2) ** 2 + (d["cy"] - img_h / 2) ** 2, -d["conf"]))
    raise ValueError(rule)


def square_crop(img: Image.Image, det: dict, margin: float) -> Image.Image:
    side = max(det["w"], det["h"]) * (1 + 2 * margin)
    x0, y0 = round(det["cx"] - side / 2), round(det["cy"] - side / 2)
    s = round(side)
    return img.crop((x0, y0, x0 + s, y0 + s))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def write_previews(manifest: pd.DataFrame, dets: pd.DataFrame, chosen: dict, src_dir: Path, out: Path,
                   min_conf: float) -> None:
    rng = np.random.default_rng(0)
    prev = out / "preview"
    prev.mkdir(parents=True, exist_ok=True)
    ok = manifest[manifest["status"] == "ok"]
    if len(ok):
        sample = ok.iloc[rng.choice(len(ok), size=min(40, len(ok)), replace=False)]
        contact_sheet([("random p50 face crops", [(out / "frames" / r.frame_p50, f"{r.class_name} {r.video_stem}")
                                                  for r in sample.itertuples()], "black")], prev / "random_crops.jpg")
    nf = manifest[manifest["status"] == "no_face"]
    if len(nf):
        sample = nf.iloc[rng.choice(len(nf), size=min(30, len(nf)), replace=False)]
        contact_sheet([("p50 frames with NO face detected", [(src_dir / r.src_frame_p50, f"{r.class_name} {r.video_stem}")
                                                             for r in sample.itertuples()], "red")], prev / "no_face.jpg")
    multi = manifest[manifest["n_faces_p50"] >= 2]
    if len(multi):
        sample = multi.iloc[rng.choice(len(multi), size=min(20, len(multi)), replace=False)]
        tiles = []
        for r in sample.itertuples():
            img = Image.open(src_dir / r.src_frame_p50).convert("RGB")
            draw = ImageDraw.Draw(img)
            key = (r.video_relpath, "p50")
            for d in dets[(dets["video_relpath"] == r.video_relpath) & (dets["frame"] == "p50")].itertuples():
                if chosen.get(key) == d.det_idx:
                    color, width = "red", 3
                elif d.conf >= min_conf:
                    color, width = "blue", 3
                else:
                    color, width = "yellow", 1
                draw.rectangle([d.x1, d.y1, d.x2, d.y2], outline=color, width=width)
            p = prev / "multi" / f"{r.split}_{r.class_name}_{r.video_stem}.jpg"
            p.parent.mkdir(parents=True, exist_ok=True)
            img.save(p, quality=90)
            tiles.append((p, f"{r.class_name} {r.video_stem} n={r.n_faces_p50}"))
        contact_sheet([("frames with >=2 faces: red = cropped face, blue = other face, thin yellow = low-confidence "
                        "detection (never a candidate)", tiles, "black")], prev / "multi_face.jpg")
    print(f"previews -> {prev}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV, help="full-frame manifest (stage 1)")
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR, help="full frames (stage 1)")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "caer_face")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--select", default="largest", choices=["largest", "confident", "central"])
    parser.add_argument("--min-conf", type=float, default=0.5, help="a face must reach this confidence to be cropped")
    parser.add_argument("--det-conf", type=float, default=0.25, help="detections >= this are logged to detections.csv")
    parser.add_argument("--margin", type=float, default=0.25, help="extra context around the face box, per side")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit-per-class", type=int, default=0, help="smoke runs: first N clips per split x class")
    parser.add_argument("--positions", nargs="+", default=list(FRAME_POSITIONS), choices=list(FRAME_POSITIONS),
                        help="frame positions to scan (CAER-S images: p50 only)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_manifest = args.out / "manifest.csv"
    if out_manifest.exists() and not args.force:
        raise SystemExit(f"{out_manifest} exists — use --force to redo (detection is deterministic)")

    manifest = load_manifest(args.manifest)
    manifest = manifest[manifest["split"].isin(args.splits)]
    if args.limit_per_class:
        manifest = manifest.groupby(["split", "class_name"], sort=False).head(args.limit_per_class)
    manifest = manifest.reset_index(drop=True)
    positions = args.positions
    jobs = [(r.video_relpath, pos, getattr(r, f"frame_{pos}")) for r in manifest.itertuples() for pos in positions]
    print(f"{len(manifest)} clips, {len(jobs)} frames to scan ({MODEL_REPO}@{MODEL_REVISION[:8]})")

    model = load_detector()
    det_rows, chosen, face_info = [], {}, {}
    for i in tqdm(range(0, len(jobs), args.batch_size), desc="detect"):
        batch = jobs[i : i + args.batch_size]
        results = model.predict([str(args.frames_dir / rel) for _, _, rel in batch], conf=args.det_conf,
                                imgsz=args.imgsz, verbose=False)
        for (vid, pos, rel), res in zip(batch, results):
            img_h, img_w = res.orig_shape
            boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
            confs = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros(0)
            dets = []
            for j, ((x1, y1, x2, y2), c) in enumerate(zip(boxes, confs)):
                d = {"video_relpath": vid, "frame": pos, "det_idx": j, "x1": float(x1), "y1": float(y1),
                     "x2": float(x2), "y2": float(y2), "conf": float(c), "w": float(x2 - x1), "h": float(y2 - y1),
                     "cx": float((x1 + x2) / 2), "cy": float((y1 + y2) / 2),
                     "area_frac": float((x2 - x1) * (y2 - y1) / (img_w * img_h)), "img_w": img_w, "img_h": img_h}
                det_rows.append(d)
                if c >= args.min_conf:
                    dets.append(d)
            pick = choose(dets, args.select, img_w, img_h)
            info = {"n_faces": len(dets), "crop": ""}
            if pick is not None:
                chosen[(vid, pos)] = pick["det_idx"]
                crop_rel = rel
                dst = args.out / "frames" / crop_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                square_crop(Image.open(args.frames_dir / rel).convert("RGB"), pick, args.margin).save(dst)
                info.update(crop=crop_rel, conf=pick["conf"], w=pick["w"], h=pick["h"], area_frac=pick["area_frac"])
            face_info[(vid, pos)] = info

    out = manifest.copy()
    for pos in positions:
        out[f"src_frame_{pos}"] = out[f"frame_{pos}"]
        infos = [face_info[(v, pos)] for v in out["video_relpath"]]
        out[f"frame_{pos}"] = [x["crop"] for x in infos]
        out[f"n_faces_{pos}"] = [x["n_faces"] for x in infos]
        out[f"face_conf_{pos}"] = [x.get("conf", np.nan) for x in infos]
        out[f"face_w_px_{pos}"] = [x.get("w", np.nan) for x in infos]
        out[f"face_h_px_{pos}"] = [x.get("h", np.nan) for x in infos]
        out[f"face_area_frac_{pos}"] = [x.get("area_frac", np.nan) for x in infos]
    out["status"] = np.where(out["frame_p50"] != "", "ok", "no_face")
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_manifest, index=False)
    dets_df = pd.DataFrame(det_rows)
    dets_df.to_csv(args.out / "detections.csv", index=False)
    (args.out / "config.json").write_text(json.dumps({
        "stage": "caer_face_crops", "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION,
        "select": args.select, "min_conf": args.min_conf, "det_conf": args.det_conf, "margin": args.margin,
        "imgsz": args.imgsz, "source_manifest": str(args.manifest), "limit_per_class": args.limit_per_class,
        "git_commit": git_commit(), "timestamp": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))

    print(f"\nwrote {out_manifest} ({len(out)} clips), {len(dets_df)} raw detections")
    print("\n% of clips with a face in the p50 frame:")
    print((out.assign(face=out["status"] == "ok").groupby(["split", "class_name"], sort=False)["face"].mean() * 100)
          .round(1).unstack().reindex(columns=CLASSES).to_string())
    ok = out[out["status"] == "ok"]
    print(f"\nfaces per p50 frame (conf >= {args.min_conf}): " +
          str(out["n_faces_p50"].value_counts().sort_index().to_dict()))
    if len(ok):
        side = ok[["face_w_px_p50", "face_h_px_p50"]].max(axis=1)
        print(f"chosen face size (px, longer side): median {side.median():.0f}, "
              f"<{SMALL_FACE_PX}px: {(side < SMALL_FACE_PX).sum()} clips; "
              f"median face area = {100 * ok['face_area_frac_p50'].median():.1f}% of the frame")
    for pos in [p for p in ["p25", "p75"] if p in positions]:
        print(f"{pos}: face found in {(out[f'frame_{pos}'] != '').mean():.1%} of clips")
    write_previews(out, dets_df, chosen, args.frames_dir, args.out, args.min_conf)


if __name__ == "__main__":
    main()
