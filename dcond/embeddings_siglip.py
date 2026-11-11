from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import SiglipModel, SiglipProcessor

from dcond.embeddings import _extract_features

DEFAULT_MODEL = "google/siglip-base-patch16-224"


class SiglipEncoder:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SiglipModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = SiglipProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def encode_images(self, images: list[Path | Image.Image], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(images), batch_size), desc="encode images (siglip)"):
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
        for i in tqdm(range(0, len(texts), batch_size), desc="encode texts (siglip)"):
            batch = texts[i : i + batch_size]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding="max_length", truncation=True
            ).to(self.device)
            feats = _extract_features(self.model.get_text_features(**inputs))
            out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)
