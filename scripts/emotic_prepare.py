import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFile
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.emotic import CATEGORIES, DEFAULT_EMOTIC_ROOT, SPLITS, multi_hot, parse_split
from experiments.caer01_inspect import contact_sheet

MAX_SIDE = 768
MIN_BOX = 4


def letterbox(img: Image.Image) -> Image.Image:
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    if side > MAX_SIDE:
        canvas = canvas.resize((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
    return canvas


def open_image(src: Path) -> tuple[Image.Image | None, str]:
    try:
        return Image.open(src).convert("RGB"), ""
    except OSError as e:
        if "truncated" not in str(e):
            return None, f"unreadable:{type(e).__name__}"
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        return Image.open(src).convert("RGB"), "truncated_image"
    except Exception as e:
        return None, f"unreadable:{type(e).__name__}"
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def process_image(args) -> list[dict]:
    image_relpath, persons, root, out, force, keep_mismatch = args
    root, out = Path(root), Path(out)
    results = []
    src = root / image_relpath
    if not src.exists():
        return [{"person_id": p["person_id"], "status": "missing_image", "issues": "", "view_image": "",
                 "view_person": ""} for p in persons]
    img, img_issue = open_image(src)
    if img is None:
        return [{"person_id": p["person_id"], "status": "unreadable_image", "issues": img_issue, "view_image": "",
                 "view_person": ""} for p in persons]
    W, H = img.size
    view_image = "image/" + str(Path(image_relpath).with_suffix(".jpg")).replace("\\", "/")
    dst = out / "views" / view_image
    if force or not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        letterbox(img).save(dst, quality=95)
    for p in persons:
        issues = [img_issue] if img_issue else []
        mismatch = (p["img_w"], p["img_h"]) != (W, H)
        if mismatch:
            issues.append("size_mismatch")
        x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
        cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            issues.append("bbox_clipped")
        r = {"person_id": p["person_id"], "issues": "|".join(issues), "view_image": view_image,
             "actual_w": W, "actual_h": H, "box_w": cx2 - cx1, "box_h": cy2 - cy1,
             "box_area_frac": max(0, cx2 - cx1) * max(0, cy2 - cy1) / (W * H)}
        if mismatch and not keep_mismatch:
            results.append({**r, "status": "size_mismatch", "view_person": ""})
            continue
        if cx2 - cx1 < MIN_BOX or cy2 - cy1 < MIN_BOX:
            results.append({**r, "status": "bad_bbox", "view_person": ""})
            continue
        view_person = f"person/{p['split']}/{p['person_id']}.jpg"
        dst = out / "views" / view_person
        if force or not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            letterbox(img.crop((round(cx1), round(cy1), round(cx2), round(cy2)))).save(dst, quality=95)
        results.append({**r, "status": "ok", "view_person": view_person})
    return results


def previews(df: pd.DataFrame, root: Path, out: Path) -> None:
    rng = np.random.default_rng(0)
    prev = out / "preview"
    prev.mkdir(parents=True, exist_ok=True)
    ok = df[df["status"] == "ok"]
    s = ok.iloc[rng.choice(len(ok), size=min(40, len(ok)), replace=False)]
    contact_sheet([("random person crops (letterboxed)", [(out / "views" / r.view_person, r.categories[:24])
                                                          for r in s.itertuples()], "black")], prev / "person_crops.jpg")
    def draw_boxes(img_rel: str, name: str) -> Path:
        persons = df[df["image_relpath"] == img_rel]
        img = Image.open(root / img_rel).convert("RGB")
        draw = ImageDraw.Draw(img)
        for p in persons.itertuples():
            ok = p.status == "ok"
            draw.rectangle([p.x1, p.y1, p.x2, p.y2], outline="red" if ok else "blue", width=4 if ok else 2)
        path = prev / name / (img_rel.replace("/", "_") + ".jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, quality=90)
        return path

    tiles = [(draw_boxes(img_rel, "boxes"), f"{(df['image_relpath'] == img_rel).sum()} person(s)")
             for img_rel in pd.unique(s["image_relpath"])[:16]]
    contact_sheet([("annotated boxes (red = used, blue = excluded) — red must sit on people", tiles, "black")],
                  prev / "boxes.jpg")
    mism = df[df["status"] == "size_mismatch"]
    if len(mism):
        tiles = [(draw_boxes(img_rel, "size_mismatch"),
                  f"csv {int(r.img_w)}x{int(r.img_h)} / file {int(r.actual_w)}x{int(r.actual_h)}")
                 for img_rel, r in mism.groupby("image_relpath").first().head(16).iterrows()]
        contact_sheet([("EXCLUDED: CSV size != file size (blue = raw CSV box)", tiles, "black")],
                      prev / "size_mismatch.jpg")
    print(f"previews -> {prev}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotic-root", type=Path, default=DEFAULT_EMOTIC_ROOT)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "emotic")
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--limit", type=int, default=0, help="first N persons per split (smoke runs)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-size-mismatch", action="store_true",
                        help="keep persons whose CSV image size != file size (raw coordinates; see docstring)")
    args = parser.parse_args()
    root = args.emotic_root.expanduser().resolve()

    df = pd.concat([parse_split(root / f"{s}.csv", s) for s in args.splits], ignore_index=True)
    if args.limit:
        df = df.groupby("split", sort=False).head(args.limit).reset_index(drop=True)
    print(f"{len(df)} annotated persons in {df['image_relpath'].nunique()} images under {root}")

    jobs = [(img, g.to_dict("records"), str(root), str(args.out), args.force, args.keep_size_mismatch)
            for (split, img), g in df.groupby(["split", "image_relpath"], sort=False)]
    results = []
    with Pool(args.workers) as pool:
        for r in tqdm(pool.imap_unordered(process_image, jobs, chunksize=16), total=len(jobs), desc="views"):
            results += r
    df = df.merge(pd.DataFrame(results), on="person_id", how="left")
    df = df.sort_values(["split", "person_id"], key=lambda s: s.map(SPLITS.index) if s.name == "split" else s)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "manifest.csv", index=False)

    print(f"\nwrote {args.out / 'manifest.csv'}")
    print("status:", df.groupby(["split", "status"]).size().to_dict())
    print("issues:", df["issues"].str.split("|").explode().replace("", np.nan).dropna().value_counts().to_dict())
    print("categories per person:", df["n_categories"].value_counts().sort_index().to_dict())
    counts = pd.DataFrame(multi_hot(df["categories"]), columns=CATEGORIES).groupby(df["split"].values).sum().T
    print("\npersons per category:\n" + counts[[s for s in SPLITS if s in counts.columns]].to_string())
    ok = df[df["status"] == "ok"]
    print(f"\nperson box: median {ok['box_w'].median():.0f}x{ok['box_h'].median():.0f} px, "
          f"median {100 * ok['box_area_frac'].median():.1f}% of the image")
    previews(df, root, args.out)


if __name__ == "__main__":
    main()
