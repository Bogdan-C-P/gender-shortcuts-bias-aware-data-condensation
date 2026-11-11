from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def _extract_features(output: torch.Tensor | object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    return output.pooler_output


class ClipEncoder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(self, images: list[Path | Image.Image], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(images), batch_size), desc="encode images"):
            batch = [
                img if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
                for img in images[i : i + batch_size]
            ]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            feats = _extract_features(self.model.get_image_features(**inputs))
            out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc="encode texts"):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            feats = _extract_features(self.model.get_text_features(**inputs))
            out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def save_embeddings(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as f:
        return {k: f[k] for k in f.files}
