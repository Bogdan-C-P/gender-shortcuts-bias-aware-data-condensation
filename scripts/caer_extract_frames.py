import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES, DEFAULT_CAER_ROOT, FRAME_POSITIONS, SPLITS, frame_relpath

MANIFEST_COLUMNS = (
    ["split", "class_name", "label", "video_relpath", "video_stem"]
    + [f"frame_{f}" for f in FRAME_POSITIONS]
    + [f"frame_idx_{f}" for f in FRAME_POSITIONS]
    + ["n_frames_reported", "n_frames_actual", "fps", "width", "height", "status", "error"]
)


def list_videos(caer_root: Path, splits: list[str], limit_per_class: int) -> list[dict]:
    jobs = []
    for split in splits:
        split_dir = caer_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing split folder: {split_dir}")
        unexpected = sorted(p.name for p in split_dir.iterdir() if p.is_dir() and p.name not in CLASSES)
        if unexpected:
            raise ValueError(f"{split_dir} has class folders not in dcond.caer.CLASSES: {unexpected}")
        for label, class_name in enumerate(CLASSES):
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"missing class folder: {class_dir}")
            videos = sorted(class_dir.glob("*.avi"))
            others = sorted(p.name for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() != ".avi")
            if others:
                print(f"WARNING: {class_dir} has {len(others)} non-.avi files, ignored (e.g. {others[:3]})")
            if limit_per_class > 0:
                videos = videos[:limit_per_class]
            for v in videos:
                jobs.append({
                    "split": split, "class_name": class_name, "label": label,
                    "video_relpath": f"{split}/{class_name}/{v.name}", "video_stem": v.stem,
                })
    return jobs


def _count_frames(path: Path) -> tuple[int, int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cv2 could not open video")
    reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n, reported, fps, width, height


def _read_frames(path: Path, targets: dict[str, int]) -> dict[str, "cv2.Mat"]:
    wanted = {}
    for name, idx in targets.items():
        wanted.setdefault(idx, []).append(name)
    out = {}
    cap = cv2.VideoCapture(str(path))
    last = max(wanted)
    i = 0
    while i <= last:
        if not cap.grab():
            break
        if i in wanted:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                raise RuntimeError(f"retrieve() failed at frame {i}")
            for name in wanted[i]:
                out[name] = frame
        i += 1
    cap.release()
    missing = set(targets) - set(out)
    if missing:
        raise RuntimeError(f"stream ended before frames {sorted(missing)} (got {i} frames on 2nd pass)")
    return out


def process_video(args: tuple[dict, str, str]) -> dict:
    job, caer_root, frames_dir = args
    row = {**job, "status": "ok", "error": ""}
    path = Path(caer_root) / job["video_relpath"]
    try:
        n, reported, fps, width, height = _count_frames(path)
        row.update(n_frames_reported=reported, n_frames_actual=n, fps=fps, width=width, height=height)
        if n == 0:
            raise RuntimeError("0 decodable frames")
        targets = {f: min(int(frac * n), n - 1) for f, frac in FRAME_POSITIONS.items()}
        frames = _read_frames(path, targets)
        for f, img in frames.items():
            rel = frame_relpath(job["split"], job["class_name"], job["video_stem"], f)
            out = Path(frames_dir) / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(out), img):
                raise RuntimeError(f"cv2.imwrite failed for {out}")
            row[f"frame_{f}"] = rel
            row[f"frame_idx_{f}"] = targets[f]
    except Exception as e:
        row["status"] = "failed"
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--caer-root", type=Path, default=DEFAULT_CAER_ROOT)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "caer")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--limit-per-class", type=int, default=0, help="0 = all videos (smoke runs: e.g. 5)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true", help="re-extract even if frames + an ok row exist")
    args = parser.parse_args()

    caer_root = args.caer_root.expanduser().resolve()
    frames_dir = args.out / "frames"
    manifest_csv = args.out / "manifest.csv"
    jobs = list_videos(caer_root, args.splits, args.limit_per_class)
    print(f"{len(jobs)} videos found under {caer_root}")

    done = {}
    if manifest_csv.exists() and not args.force:
        prev = pd.read_csv(manifest_csv, dtype={"video_stem": str}, keep_default_na=False)
        for r in prev.to_dict("records"):
            if r["status"] == "ok" and all((frames_dir / r[f"frame_{f}"]).exists() for f in FRAME_POSITIONS):
                done[r["video_relpath"]] = r
        print(f"resuming: {len(done)} videos already extracted (use --force to redo)")

    todo = [j for j in jobs if j["video_relpath"] not in done]
    rows = [done[j["video_relpath"]] for j in jobs if j["video_relpath"] in done]
    with Pool(args.workers) as pool:
        for row in tqdm(pool.imap_unordered(process_video, [(j, str(caer_root), str(frames_dir)) for j in todo],
                                            chunksize=8), total=len(todo), desc="extract"):
            rows.append(row)

    df = pd.DataFrame(rows).reindex(columns=MANIFEST_COLUMNS)
    int_cols = [f"frame_idx_{f}" for f in FRAME_POSITIONS] + ["n_frames_reported", "n_frames_actual", "width", "height"]
    df[int_cols] = df[int_cols].replace("", None).astype("Int64")
    df = df.sort_values(["split", "label", "video_relpath"], key=lambda s: s.map(SPLITS.index) if s.name == "split" else s)
    df = df.reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_csv, index=False)
    failures = df[df["status"] != "ok"]
    failures.to_csv(args.out / "extract_failures.csv", index=False)

    print(f"\nwrote {manifest_csv} ({len(df)} rows, {len(failures)} failed -> extract_failures.csv)")
    ok = df[df["status"] == "ok"]
    print(ok.groupby(["split", "class_name"], sort=False).size().unstack(fill_value=0).to_string())
    mismatch = ok[ok["n_frames_reported"].astype(int) != ok["n_frames_actual"].astype(int)]
    print(f"\nframe-count header mismatches (reported != actual): {len(mismatch)}/{len(ok)}")
    print(f"actual frames per clip: min={ok['n_frames_actual'].min()} median={ok['n_frames_actual'].median()} "
          f"max={ok['n_frames_actual'].max()}")


if __name__ == "__main__":
    main()
