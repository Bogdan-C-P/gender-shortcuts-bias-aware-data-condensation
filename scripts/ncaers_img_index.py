import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "ncaers" / "manifest.csv")
    ap.add_argument("--face-manifest", type=Path, default=PROJECT_ROOT / "data" / "ncaers_face" / "manifest.csv")
    ap.add_argument("--images", type=Path, default=PROJECT_ROOT / "data" / "ncaers" / "images")
    ap.add_argument("--face-frames", type=Path, default=PROJECT_ROOT / "data" / "ncaers_face" / "frames")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "ncaers_img")
    a = ap.parse_args()
    m = pd.read_csv(a.manifest, keep_default_na=False)
    f = pd.read_csv(a.face_manifest, keep_default_na=False).set_index("video_relpath")["frame_p50"]
    missing = ~m["video_relpath"].isin(f.index)
    if missing.any():
        raise SystemExit(f"{missing.sum()} images have no face-crop row, e.g. {m.loc[missing, 'video_relpath'].head(3).tolist()}")
    a.out.mkdir(parents=True, exist_ok=True)
    for name, target in [("full", a.images), ("face", a.face_frames)]:
        link = a.out / name
        if not link.exists():
            link.symlink_to(target.resolve(), target_is_directory=True)
    for split, g in m.groupby("split"):
        idx = pd.DataFrame({"video_relpath": g["video_relpath"], "split": split, "label": g["label"], "j": 0,
                            "frame_idx": g["frame_idx"], "full": g["frame_p50"], "face": f.reindex(g["video_relpath"]).values})
        missing_files = [p for p in idx["face"].head(50) if not (a.out / "face" / p).exists()]
        assert not missing_files, f"face crops not found under {a.out / 'face'}: {missing_files[:3]}"
        idx.to_csv(a.out / f"index_{split}.csv", index=False)
        print(f"index_{split}.csv: {len(idx)} images")


if __name__ == "__main__":
    main()
