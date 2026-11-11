import argparse
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES, DEFAULT_FRAME, ENCODER_TAGS, GENERIC_ENCODERS, FRAME_POSITIONS, FRAMES_DIR, MANIFEST_CSV, load_manifest
from dcond.outliers import METRICS
from dcond.stats import holm_bonferroni

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results" / "caer01"
RULES = {"top": "outlier_top_class", "median": "outlier_median_class", "top_or_small": "outlier_top_or_small"}
THUMB = 160
N_SHOW = 20


def run_dir(results: Path, enc: str, metric: str, frame: str, split: str, k: int) -> Path:
    return results / f"{enc}_{metric}_{frame}" / split / f"k{k}"


def load_assignments(results, enc, metric, frame, split, k) -> pd.DataFrame | None:
    p = run_dir(results, enc, metric, frame, split, k) / "assignments.csv"
    return pd.read_csv(p) if p.exists() else None


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else np.nan


def _tile(path: Path, caption: str, color) -> Image.Image:
    tile = Image.new("RGB", (THUMB, THUMB + 28), "white")
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((THUMB, THUMB))
        tile.paste(img, ((THUMB - img.width) // 2, (THUMB - img.height) // 2))
    except Exception:
        ImageDraw.Draw(tile).text((5, THUMB // 2), "missing frame", fill="red")
    ImageDraw.Draw(tile).text((3, THUMB + 2), caption, fill=color)
    return tile


def contact_sheet(blocks: list[tuple[str, list[tuple[Path, str]], str]], out: Path, per_row: int = 10) -> None:
    th, tw = THUMB + 28, THUMB
    rows_per_block = [max(1, int(np.ceil(len(items) / per_row))) for _, items, _ in blocks]
    sheet = Image.new("RGB", (per_row * tw, sum(20 + r * th for r in rows_per_block)), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for (title, items, color), nrows in zip(blocks, rows_per_block):
        draw.text((4, y + 4), title, fill="black")
        y += 20
        for i, (path, cap) in enumerate(items):
            sheet.paste(_tile(path, cap, color), ((i % per_row) * tw, y + (i // per_row) * th))
        y += nrows * th
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)


def frame_stats(manifest: pd.DataFrame, frame: str, frames_dir: Path, cache: Path) -> pd.DataFrame:
    prev = pd.read_csv(cache) if cache.exists() else pd.DataFrame(columns=["video_relpath"])
    todo = manifest[~manifest["video_relpath"].isin(prev["video_relpath"])]
    rows = []
    for r in todo.itertuples():
        img = cv2.imread(str(frames_dir / getattr(r, f"frame_{frame}")))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        rows.append({"video_relpath": r.video_relpath, "brightness": gray.mean(), "contrast": gray.std(),
                     "sharpness": cv2.Laplacian(gray, cv2.CV_64F).var()})
    if rows:
        prev = pd.concat([prev, pd.DataFrame(rows)], ignore_index=True)
        prev.to_csv(cache, index=False)
    return prev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoders", nargs="+", default=GENERIC_ENCODERS, choices=list(ENCODER_TAGS))
    parser.add_argument("--metric", default="cosine", choices=METRICS)
    parser.add_argument("--frame", default=DEFAULT_FRAME, choices=list(FRAME_POSITIONS))
    parser.add_argument("--split", default="train")
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--primary-k", type=int, default=1)
    parser.add_argument("--rule", default="top", choices=list(RULES))
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--no-sheets", action="store_true")
    args = parser.parse_args()

    res = args.results_dir
    out = res / "inspect" / f"{args.metric}_{args.frame}_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    manifest = manifest[manifest["split"] == args.split]
    frame_path = dict(zip(manifest["video_relpath"], manifest[f"frame_{args.frame}"]))
    flag = RULES[args.rule]

    summaries = []
    for p in sorted(res.glob("*_*_*/*/k*/summary.csv")):
        s = pd.read_csv(p)
        enc, metric, frame = p.parts[-4].rsplit("_", 2)
        summaries.append(s.assign(encoder=enc, metric=metric, frame=frame))
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(out / "overview.csv", index=False)

    runs = {(e, k): load_assignments(res, e, args.metric, args.frame, args.split, k)
            for e in args.encoders for k in args.ks}
    runs = {key: df for key, df in runs.items() if df is not None}
    if not runs:
        raise SystemExit(f"no stage-3 results for {args.metric}/{args.frame}/{args.split} under {res}")
    encoders = sorted({e for e, _ in runs})
    print(f"found runs: {sorted(runs)}")

    if not args.no_sheets:
        for (enc, k), df in sorted(runs.items()):
            for cls in CLASSES:
                d = df[df["class_name"] == cls].sort_values("dist_to_centroid")
                if d.empty:
                    continue
                cap = lambda r: f"{Path(r.video_relpath).stem} d={r.dist_to_centroid:.3f} c{r.cluster} zs:{r.zs_pred[:4]}"
                far = [(args.frames_dir / frame_path.get(r.video_relpath, ""), cap(r)) for r in d.iloc[::-1][:N_SHOW].itertuples()]
                near = [(args.frames_dir / frame_path.get(r.video_relpath, ""), cap(r)) for r in d[:N_SHOW].itertuples()]
                contact_sheet([(f"{cls} | {enc} {args.metric} k={k} | {N_SHOW} MOST OUTLYING (n={len(d)})", far, "red"),
                               (f"{cls} | {N_SHOW} MOST TYPICAL", near, "darkgreen")],
                              out / "sheets" / f"{enc}_k{k}_{cls}.jpg")
        print(f"contact sheets -> {out / 'sheets'}")

    stab = []
    def outlier_set(df, col, cls):
        return set(df.loc[(df["class_name"] == cls) & df[col], "video_relpath"])
    for cls in CLASSES:
        for enc in encoders:
            for k1, k2 in combinations([k for k in args.ks if (enc, k) in runs], 2):
                a, b = outlier_set(runs[enc, k1], flag, cls), outlier_set(runs[enc, k2], flag, cls)
                stab.append({"comparison": "k", "encoder": enc, "a": f"k{k1}", "b": f"k{k2}", "class_name": cls,
                             "jaccard": jaccard(a, b), "n_a": len(a), "n_b": len(b)})
            for k in args.ks:
                if (enc, k) in runs:
                    a, b = outlier_set(runs[enc, k], RULES["top"], cls), outlier_set(runs[enc, k], RULES["median"], cls)
                    stab.append({"comparison": "rule", "encoder": enc, "a": f"top@k{k}", "b": f"median@k{k}",
                                 "class_name": cls, "jaccard": jaccard(a, b), "n_a": len(a), "n_b": len(b)})
            other_frames = {f: load_assignments(res, enc, args.metric, f, args.split, args.primary_k)
                            for f in FRAME_POSITIONS if f != args.frame}
            for f, df_f in other_frames.items():
                if df_f is not None and (enc, args.primary_k) in runs:
                    a, b = outlier_set(runs[enc, args.primary_k], flag, cls), outlier_set(df_f, flag, cls)
                    stab.append({"comparison": "frame", "encoder": enc, "a": args.frame, "b": f, "class_name": cls,
                                 "jaccard": jaccard(a, b), "n_a": len(a), "n_b": len(b)})
        for k in args.ks:
            for e1, e2 in combinations([e for e in encoders if (e, k) in runs], 2):
                a, b = outlier_set(runs[e1, k], flag, cls), outlier_set(runs[e2, k], flag, cls)
                stab.append({"comparison": "encoder", "encoder": "", "a": f"{e1}@k{k}", "b": f"{e2}@k{k}",
                             "class_name": cls, "jaccard": jaccard(a, b), "n_a": len(a), "n_b": len(b)})
    stab = pd.DataFrame(stab)
    stab.to_csv(out / "stability.csv", index=False)
    if not stab.empty:
        print("\nmean Jaccard of outlier sets (rule=%s):" % args.rule)
        print(stab.groupby(["comparison", "encoder", "a", "b"])["jaccard"].mean().round(3).to_string())

    prim = {e: runs[e, args.primary_k] for e in encoders if (e, args.primary_k) in runs}
    if len(prim) >= 2:
        merged = None
        for e, df in prim.items():
            part = df[["video_relpath", "class_name", flag, "pct_in_class", "zs_pred"]].rename(
                columns={flag: f"out_{e}", "pct_in_class": f"pct_{e}", "zs_pred": f"zs_pred_{e}"})
            merged = part if merged is None else merged.merge(part, on=["video_relpath", "class_name"])
        merged["n_encoders_flagging"] = merged[[f"out_{e}" for e in prim]].sum(axis=1)
        cons = merged[merged["n_encoders_flagging"] == len(prim)].copy()
        cons["frame_path"] = cons["video_relpath"].map(frame_path)
        cons.to_csv(out / "consensus_outliers.csv", index=False)
        print(f"\nconsensus outliers (flagged by all {len(prim)} encoders, k={args.primary_k}, rule={args.rule}):")
        print(merged.groupby("class_name")["n_encoders_flagging"].agg(
            n="size", flagged_by_all=lambda s: (s == len(prim)).sum(),
            flagged_by_any=lambda s: (s > 0).sum()).to_string())
        if not args.no_sheets:
            for cls in CLASSES:
                c = cons[cons["class_name"] == cls]
                if len(c):
                    items = [(args.frames_dir / r.frame_path, Path(r.video_relpath).stem) for r in c.itertuples()]
                    contact_sheet([(f"{cls} | consensus outliers ({len(c)}) — all of {list(prim)} at k={args.primary_k}",
                                    items, "red")], out / "sheets" / f"consensus_{cls}.jpg")

    fstats = frame_stats(manifest, args.frame, args.frames_dir, args.manifest.parent / f"frame_stats_{args.frame}.csv")
    meta = manifest[["video_relpath", "n_frames_actual", "fps"]].merge(fstats, on="video_relpath", how="left")
    meta["duration_s"] = meta["n_frames_actual"] / meta["fps"].replace(0, np.nan)
    rows = []
    for e, df in prim.items():
        d = df.merge(meta, on="video_relpath", how="left")
        o, i = d[d[flag]], d[~d[flag]]
        tests = []
        for col in ["zs_correct", "closer_to_other_class"]:
            table = [[int(o[col].sum()), int((~o[col].astype(bool)).sum())],
                     [int(i[col].sum()), int((~i[col].astype(bool)).sum())]]
            _, p = stats.fisher_exact(table)
            tests.append({"feature": col, "test": "fisher", "outlier_mean": o[col].mean(), "inlier_mean": i[col].mean(), "p": p})
        for col in ["brightness", "contrast", "sharpness", "n_frames_actual", "duration_s"]:
            a, b = o[col].dropna(), i[col].dropna()
            p = stats.mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan
            tests.append({"feature": col, "test": "mann-whitney", "outlier_mean": a.mean(), "inlier_mean": b.mean(),
                          "outlier_median": a.median(), "inlier_median": b.median(), "p": p})
        valid = [t for t in tests if not np.isnan(t["p"])]
        for t, (p_holm, rej) in zip(valid, holm_bonferroni([t["p"] for t in valid])):
            t.update(p_holm=p_holm, significant_holm=rej)
        rows += [{"encoder": e, "k": args.primary_k, "rule": args.rule, "n_outliers": len(o), "n_inliers": len(i), **t}
                 for t in tests]
    chars = pd.DataFrame(rows)
    chars.to_csv(out / "characteristics.csv", index=False)
    if not chars.empty:
        print("\noutliers vs inliers:")
        print(chars[["encoder", "feature", "outlier_mean", "inlier_mean", "p", "p_holm", "significant_holm"]]
              .round(4).to_string(index=False))
    print(f"\nall outputs -> {out}")


if __name__ == "__main__":
    main()
