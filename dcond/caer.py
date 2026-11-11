import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAER_ROOT = Path(os.environ.get("CAER_ROOT", PROJECT_ROOT / "data" / "caer_raw" / "CAER")).expanduser()
DATA_DIR = PROJECT_ROOT / "data" / "caer"
FRAMES_DIR = DATA_DIR / "frames"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
EMB_DIR = DATA_DIR / "embeddings"

CLASSES = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
SPLITS = ["train", "validation", "test"]
FRAME_POSITIONS = {"p25": 0.25, "p50": 0.50, "p75": 0.75}
DEFAULT_FRAME = "p50"

CLASS_PROMPTS = [f"a photo of a person feeling {c.lower()}" for c in CLASSES]
CLASS_PROMPTS[CLASSES.index("Neutral")] = "a photo of a person with a neutral expression"


ENCODER_TAGS = {
    "clip": "clip-vit-large-patch14",
    "siglip": "siglip-base-patch16-224",
    "evaclip": "EVA02-B-16_merged2b_s8b_b131k",
    "fer_vit": "vit-face-expression",
    "clip_fer": "clip-vit-large-patch14_fer2013",
    "emotionclip": "EmotionCLIP_ViT-B-32",
}
GENERIC_ENCODERS = ["clip", "siglip", "evaclip"]
EMOTION_ENCODERS = ["fer_vit", "clip_fer", "emotionclip"]


def frame_relpath(split: str, class_name: str, video_stem: str, frame: str) -> str:
    return f"{split}/{class_name}/{video_stem}_{frame}.png"


def load_manifest(path: Path = MANIFEST_CSV, ok_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"video_stem": str})
    if ok_only:
        df = df[df["status"] == "ok"].reset_index(drop=True)
    return df


def embedding_path(split: str, frame: str, tag: str, emb_dir: Path = EMB_DIR) -> Path:
    return Path(emb_dir) / f"caer_{split}_{frame}_{tag}.npz"
