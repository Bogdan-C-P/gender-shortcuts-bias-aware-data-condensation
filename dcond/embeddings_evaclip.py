from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

DEFAULT_MODEL_NAME = "EVA02-B-16"
DEFAULT_PRETRAINED = "merged2b_s8b_b131k"


class EvaClipEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        pretrained: str = DEFAULT_PRETRAINED,
        device: str | None = None,
    ):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.no_grad()
    def encode_images(self, images: list[Path | Image.Image], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(images), batch_size), desc="encode images (eva-clip)"):
            batch = [
                img if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
                for img in images[i : i + batch_size]
            ]
            pixel_values = torch.stack([self.preprocess(img) for img in batch]).to(self.device)
            feats = self.model.encode_image(pixel_values)
            out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(texts), batch_size), desc="encode texts (eva-clip)"):
            batch = texts[i : i + batch_size]
            tokens = self.tokenizer(batch).to(self.device)
            feats = self.model.encode_text(tokens)
            out.append(feats.cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)
