import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from dcond.caer import PROJECT_ROOT

MULTI_DIR = PROJECT_ROOT / "data" / "caer_multi"
DEDUP_JSON = PROJECT_ROOT / "experiments" / "results" / "caer03_mlp_probe" / "dedup_ids.json"
ALLTEST_JSON = PROJECT_ROOT / "experiments" / "results" / "caer_alltest_no_dedup.json"
EVAL = {"name": "dedup", "short": "dedup", "label": "test without near-twins"}
SPLIT_NAMES = ["train", "validation", "test"]


def load_multi(split: str, k: int = 15, multi_dir: Path = MULTI_DIR) -> dict:
    with np.load(Path(multi_dir) / f"caer_{split}_k{k}.npz", allow_pickle=True) as f:
        d = {key: f[key] for key in f.files}
    lg = d["fer_logits"].astype(np.float64)
    lg = lg - np.nanmax(lg, axis=-1, keepdims=True)
    p = np.exp(lg)
    d["fer_p"] = p / p.sum(-1, keepdims=True)
    d["video_relpath"] = d["video_relpath"].astype(str)
    return d


def load_ft(split: str, ft_dir: Path) -> dict:
    with np.load(Path(ft_dir) / f"caer_{split}_k15.npz", allow_pickle=True) as f:
        emb, valid = f["emb"].astype(np.float16), f["valid"].astype(bool)
        d = {"clip_full": emb, "clip_face": emb, "face_ok": valid, "label": f["label"],
             "video_relpath": f["video_relpath"].astype(str)}
    d["fer_logits"] = np.full(valid.shape + (7,), np.nan, np.float32)
    d["fer_p"] = d["fer_logits"].astype(np.float64)
    return d


def load_dedup(path: Path = DEDUP_JSON) -> dict[str, set]:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} — pass {ALLTEST_JSON.name} explicitly for the full-test protocol")
    d = json.loads(Path(path).read_text())
    dup = {s: set(d.get(s, [])) for s in ["validation", "test"]}
    if not dup["validation"] and not dup["test"]:
        EVAL.update(name="all", short="full test", label="full test, all 2,637 clips (twins included)")
        print("evaluation protocol: FULL validation for selection, FULL test (twins included)")
    return dup


@dataclass
class FrameSet:
    X: np.ndarray
    valid: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    fer_p: np.ndarray
    keep: np.ndarray
    extra: dict = field(default_factory=dict, repr=False)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def frame_sets(data: dict[str, dict], variant: str, dup: dict[str, set]) -> dict[str, FrameSet]:
    out = {}
    for s, d in data.items():
        full = d["clip_full"].astype(np.float32)
        face = d["clip_face"].astype(np.float32)
        ok = d["face_ok"].astype(bool)
        rows = np.ones(len(d["label"]), bool)
        if variant == "full":
            X, valid = full, np.ones_like(ok)
        elif variant == "face":
            rows = ok.any(1)
            X, valid = face, ok
        elif variant == "both":
            mean_face = _l2((face * ok[..., None]).sum(1)) * ok.any(1, keepdims=True)
            face_filled = np.where(ok[..., None], face, mean_face[:, None, :])
            X, valid = np.concatenate([full, face_filled], axis=-1), np.ones_like(ok)
        else:
            raise ValueError(variant)
        ids = d["video_relpath"][rows]
        keep = ~np.isin(ids, list(dup[s])) if s in dup else np.ones(len(ids), bool)
        out[s] = FrameSet(X[rows], valid[rows], d["label"][rows].astype(int), ids, d["fer_p"][rows], keep,
                          {"face_ok": ok[rows]})
    return out


def uniform_idx(k: int, n: int = 15) -> np.ndarray:
    return np.round((np.arange(k) + 0.5) * n / k - 0.5).astype(int)


def mean_rep(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = w.astype(np.float32)
    m = (X * w[..., None]).sum(1) / np.maximum(w.sum(1, keepdims=True), 1e-8)
    return _l2(m).astype(np.float32)


def mid_idx(valid: np.ndarray) -> np.ndarray:
    k = valid.shape[1]
    return (np.abs(np.arange(k) - k // 2)[None, :] + np.where(valid, 0, 1000)).argmin(1)


class CosineHead(torch.nn.Module):

    def __init__(self, d: int, n_cls: int, s: float = 16.0, m: float = 0.0):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(n_cls, d) * 0.01)
        self.s, self.m = s, m

    def forward(self, f: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        cos = torch.nn.functional.linear(torch.nn.functional.normalize(f.float(), dim=-1),
                                         torch.nn.functional.normalize(self.W, dim=-1)).clamp(-1 + 1e-7, 1 - 1e-7)
        if y is None or self.m == 0 or not self.training:
            return self.s * cos
        sin = torch.sqrt((1.0 - cos * cos).clamp_min(1e-7))
        phi = cos * math.cos(self.m) - sin * math.sin(self.m)
        target = torch.where(cos > math.cos(math.pi - self.m), phi, cos - math.sin(math.pi - self.m) * self.m)
        onehot = torch.nn.functional.one_hot(y, cos.shape[1]).bool()
        return self.s * torch.where(onehot, target, cos)


def ncm_predict(X_tr: np.ndarray, y_tr: np.ndarray, X: np.ndarray, n_cls: int = 7) -> np.ndarray:
    mu, sd = X_tr.mean(0), X_tr.std(0) + 1e-6
    Z_tr, Z = _l2((X_tr - mu) / sd), _l2((X - mu) / sd)
    C = _l2(np.stack([Z_tr[y_tr == c].mean(0) for c in range(n_cls)]))
    return (Z @ C.T).argmax(1)


def nn1_predict(X_tr: np.ndarray, y_tr: np.ndarray, X: np.ndarray, chunk: int = 2048) -> np.ndarray:
    Z_tr, Z = _l2(X_tr), _l2(X)
    return np.concatenate([y_tr[(Z[i : i + chunk] @ Z_tr.T).argmax(1)] for i in range(0, len(Z), chunk)])
