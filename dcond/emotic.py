import ast
import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMOTIC_ROOT = Path(os.environ.get("EMOTIC_ROOT", PROJECT_ROOT / "data" / "emotic_raw")).expanduser()
DATA_DIR = PROJECT_ROOT / "data" / "emotic"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
EMB_DIR = DATA_DIR / "embeddings"
SPLITS = ["train", "val", "test"]
VIEWS = ["image", "person"]

CATEGORIES = [
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion", "Confidence", "Disapproval", "Disconnection",
    "Disquietment", "Doubt/Confusion", "Embarrassment", "Engagement", "Esteem", "Excitement", "Fatigue", "Fear",
    "Happiness", "Pain", "Peace", "Pleasure", "Sadness", "Sensitivity", "Suffering", "Surprise", "Sympathy",
    "Yearning",
]
CATEGORY_PROMPTS = [f"a photo of a person feeling {c.lower().replace('/', ' or ')}" for c in CATEGORIES]
COLUMNS = ["index", "folder", "filename", "image_size", "bbox", "categories", "vad", "gender", "age"]


def read_split_csv(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    if not raw.iloc[0, 0].strip().lstrip("-").isdigit():
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns {COLUMNS}, got {raw.shape[1]}")
    raw.columns = COLUMNS
    return raw


def parse_split(path: Path, split: str) -> pd.DataFrame:
    raw = read_split_csv(path)
    size = raw["image_size"].map(ast.literal_eval)
    bbox = raw["bbox"].map(ast.literal_eval)
    vad = raw["vad"].map(ast.literal_eval)
    cats = raw["categories"].map(ast.literal_eval)
    unknown = sorted({c for cs in cats for c in cs} - set(CATEGORIES))
    if unknown:
        raise ValueError(f"{path}: categories not in dcond.emotic.CATEGORIES: {unknown}")
    df = pd.DataFrame({
        "person_id": [f"{split}_{int(i):05d}" for i in raw["index"]],
        "split": split, "folder": raw["folder"], "filename": raw["filename"],
        "image_relpath": raw["folder"].str.rstrip("/") + "/" + raw["filename"],
        "img_w": [s[0] for s in size], "img_h": [s[1] for s in size],
        "x1": [b[0] for b in bbox], "y1": [b[1] for b in bbox], "x2": [b[2] for b in bbox], "y2": [b[3] for b in bbox],
        "categories": ["|".join(c) for c in cats], "n_categories": [len(c) for c in cats],
        "valence": [v[0] for v in vad], "arousal": [v[1] for v in vad], "dominance": [v[2] for v in vad],
        "gender": raw["gender"], "age": raw["age"],
    })
    if df["person_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate index values")
    return df


def multi_hot(categories: pd.Series) -> np.ndarray:
    idx = {c: i for i, c in enumerate(CATEGORIES)}
    out = np.zeros((len(categories), len(CATEGORIES)), dtype=np.int8)
    for r, s in enumerate(categories):
        for c in str(s).split("|"):
            if c:
                out[r, idx[c]] = 1
    return out


def load_manifest(path: Path = MANIFEST_CSV, ok_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    return df[df["status"] == "ok"].reset_index(drop=True) if ok_only else df


def embedding_path(split: str, view: str, tag: str, emb_dir: Path = EMB_DIR) -> Path:
    return Path(emb_dir) / f"emotic_{split}_{view}_{tag}.npz"
