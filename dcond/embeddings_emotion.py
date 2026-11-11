from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

FER_VIT_MODEL = "trpakov/vit-face-expression"
CLIP_FER_VISION = "tanganke/clip-vit-large-patch14_fer2013"
CLIP_BASE = "openai/clip-vit-large-patch14"


def _load(images, i, batch_size):
    return [img if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
            for img in images[i : i + batch_size]]


class FerVitEncoder:
    def __init__(self, model_name: str = FER_VIT_MODEL, device: str | None = None):
        from transformers import AutoImageProcessor, ViTForImageClassification

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]

    @torch.no_grad()
    def encode_images_with_logits(self, images, batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        feats, logits = [], []
        for i in tqdm(range(0, len(images), batch_size), desc="encode images (fer-vit)"):
            inputs = self.processor(images=_load(images, i, batch_size), return_tensors="pt").to(self.device)
            cls = self.model.vit(**inputs).last_hidden_state[:, 0]
            feats.append(cls.cpu().numpy())
            logits.append(self.model.classifier(cls).cpu().numpy())
        return np.concatenate(feats).astype(np.float32), np.concatenate(logits).astype(np.float32)

    def encode_images(self, images, batch_size: int = 64) -> np.ndarray:
        return self.encode_images_with_logits(images, batch_size)[0]

    def logits_for(self, logits: np.ndarray, target_classes: list[str]) -> np.ndarray:
        alias = {"anger": "angry", "happiness": "happy", "sadness": "sad"}
        own = {l.lower(): j for j, l in enumerate(self.labels)}
        cols = []
        for c in target_classes:
            key = alias.get(c.lower(), c.lower())
            cols.append(logits[:, own[key]] if key in own else np.full(len(logits), -np.inf, dtype=np.float32))
        return np.stack(cols, axis=1)


class ClipFerEncoder:

    def __init__(self, device: str | None = None):
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import CLIPModel, CLIPProcessor

        from dcond.embeddings import _extract_features

        self._extract = _extract_features
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(CLIP_BASE)
        tuned = load_file(hf_hub_download(CLIP_FER_VISION, "model.safetensors"))
        tuned = {k.removeprefix("vision_model."): v for k, v in tuned.items() if k != "vision_model.embeddings.position_ids"}
        target = self.model.vision_model.state_dict()
        missing, unexpected = sorted(set(target) - set(tuned)), sorted(set(tuned) - set(target))
        if missing or unexpected:
            raise RuntimeError(f"{CLIP_FER_VISION} does not match {CLIP_BASE}'s vision tower: "
                               f"missing {missing[:5]} ({len(missing)}), unexpected {unexpected[:5]} ({len(unexpected)})")
        self.model.vision_model.load_state_dict(tuned, strict=True)
        del tuned
        self.model = self.model.to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_BASE)

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 64) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(images), batch_size), desc="encode images (clip-fer)"):
            inputs = self.processor(images=_load(images, i, batch_size), return_tensors="pt").to(self.device)
            out.append(self._extract(self.model.get_image_features(**inputs)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            inputs = self.processor(text=texts[i : i + batch_size], return_tensors="pt", padding=True,
                                    truncation=True).to(self.device)
            out.append(self._extract(self.model.get_text_features(**inputs)).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


EMOTIONCLIP_CKPT = Path(__file__).resolve().parents[1] / "models" / "emotionclip_model.pt"


class EmotionClipEncoder:

    uses_subject_box = True

    def __init__(self, ckpt_path: Path | str | None = None, device: str | None = None):
        import os

        from dcond.third_party.emotionclip.base import EmotionCLIP
        from dcond.third_party.emotionclip.tokenizer import tokenize

        ckpt_path = Path(ckpt_path or os.environ.get("EMOTIONCLIP_CKPT", EMOTIONCLIP_CKPT))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"{ckpt_path} — download the authors' checkpoint (see this module's comment) "
                                    "or set EMOTIONCLIP_CKPT")
        self._tokenize = tokenize
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        config = Path(__file__).resolve().parent / "third_party" / "emotionclip" / "ViT-B-32.json"
        self.model = EmotionCLIP(backbone_config=str(config), backbone_checkpoint=None, video_len=8)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
        self.model = self.model.to(self.device).eval()

    @staticmethod
    def _prep(img: Image.Image, box) -> tuple[torch.Tensor, torch.Tensor]:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as TF

        W, H = img.size
        img = TF.resize(img, 224, interpolation=InterpolationMode.BICUBIC)
        w, h = img.size
        mask = torch.zeros(h, w)
        if box is None:
            mask[:] = 1
        else:
            x1, y1, x2, y2 = box
            mask[max(0, int(y1 * h)):min(h, int(np.ceil(y2 * h))), max(0, int(x1 * w)):min(w, int(np.ceil(x2 * w)))] = 1
        img = TF.center_crop(img, 224)
        mask = TF.center_crop(mask, 224)
        x = TF.normalize(TF.to_tensor(img), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return x, mask.float()

    @torch.no_grad()
    def encode_images(self, images, batch_size: int = 64, boxes=None) -> np.ndarray:
        out = []
        for i in tqdm(range(0, len(images), batch_size), desc="encode images (emotionclip)"):
            batch = _load(images, i, batch_size)
            bb = boxes[i : i + batch_size] if boxes is not None else [None] * len(batch)
            xs, ms = zip(*(self._prep(img, b) for img, b in zip(batch, bb)))
            feats = self.model.encode_image(torch.stack(xs).to(self.device), torch.stack(ms).to(self.device))
            out.append(feats.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            tokens = self._tokenize(texts[i : i + batch_size]).to(self.device)
            out.append(self.model.encode_text(tokens).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)
