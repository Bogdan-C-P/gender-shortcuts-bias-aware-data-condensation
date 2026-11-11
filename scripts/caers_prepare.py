import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES

RAW = PROJECT_ROOT / "data" / "caers_raw" / "CAER-S"
OUT = PROJECT_ROOT / "data" / "caers"
ALIASES = {"angry": "Anger"}


def class_of(folder: str) -> str:
    name = ALIASES.get(folder.lower(), folder)
    match = [c for c in CLASSES if c.lower() == name.lower()]
    if not match:
        raise ValueError(f"unknown class folder {folder!r}")
    return match[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for orig_split in ["train", "test"]:
        for d in sorted(p for p in (args.raw / orig_split).iterdir() if p.is_dir()):
            cls = class_of(d.name)
            files = sorted(d.glob("*.png"))
            others = [p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() != ".png"]
            if others:
                print(f"WARNING {d}: {len(others)} non-png files ignored, e.g. {others[:3]}")
            for f in files:
                rows.append({"orig_split": orig_split, "orig_class_dir": d.name, "class_name": cls,
                             "label": CLASSES.index(cls), "video_stem": f.stem,
                             "frame_p50": f"{orig_split}/{d.name}/{f.name}"})
    df = pd.DataFrame(rows)
    df["split"] = df["orig_split"]
    rng = np.random.default_rng(args.seed)
    for cls, g in df[df["orig_split"] == "train"].groupby("class_name"):
        n_val = int(round(args.val_frac * len(g)))
        df.loc[rng.choice(g.index.to_numpy(), size=n_val, replace=False), "split"] = "validation"
    df["video_relpath"] = df["split"] + "/" + df["class_name"] + "/" + df["video_stem"] + ".png"
    assert df["video_relpath"].is_unique
    sizes = [Image.open(args.raw / p).size for p in df["frame_p50"]]
    df["width"], df["height"] = [s[0] for s in sizes], [s[1] for s in sizes]
    df["frame_p25"], df["frame_p75"] = "", ""
    df["status"] = "ok"
    order = {"train": 0, "validation": 1, "test": 2}
    df = df.sort_values(["split", "label", "video_stem"], key=lambda s: s.map(order) if s.name == "split" else s)
    cols = ["split", "class_name", "label", "video_relpath", "video_stem", "frame_p25", "frame_p50", "frame_p75",
            "orig_split", "orig_class_dir", "width", "height", "status"]
    args.out.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(args.out / "manifest.csv", index=False)
    print(f"wrote {args.out / 'manifest.csv'}: {len(df)} images")
    print(pd.crosstab(df["class_name"], df["split"]).reindex(index=CLASSES).to_string())
    print("image sizes:", pd.Series([f"{w}x{h}" for w, h in sizes]).value_counts().head(5).to_dict())


if __name__ == "__main__":
    main()
