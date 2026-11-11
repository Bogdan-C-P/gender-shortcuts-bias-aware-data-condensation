from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from dcond.caer import CLASSES, ENCODER_TAGS, PROJECT_ROOT, SPLITS, embedding_path
from dcond.embeddings import load_embeddings

EMB_DIRS = {"full": PROJECT_ROOT / "data" / "caer" / "embeddings",
            "face": PROJECT_ROOT / "data" / "caer_face" / "embeddings"}


@dataclass
class SplitData:
    X: np.ndarray
    y: np.ndarray
    ids: np.ndarray
    raw: dict = field(repr=False)


def load_variant(variant: str, enc: str, frame: str = "p50", emb_dir: Path | None = None) -> dict[str, SplitData]:
    emb_dir = emb_dir or EMB_DIRS[variant]
    out = {}
    for s in SPLITS:
        d = load_embeddings(embedding_path(s, frame, ENCODER_TAGS[enc], emb_dir))
        out[s] = SplitData(d["image_embeds"].astype(np.float32), d["label"].astype(int),
                           d["video_relpath"].astype(str), d)
    return out


def nn_similarity(X: np.ndarray, X_train: np.ndarray, chunk: int = 4096) -> np.ndarray:
    return np.concatenate([(X[i : i + chunk] @ X_train.T).max(axis=1) for i in range(0, len(X), chunk)])


def near_duplicate_ids(refs: list[dict[str, SplitData]], threshold: float = 0.95) -> dict[str, set]:
    dup = {"validation": set(), "test": set()}
    for ref in refs:
        for s in dup:
            sim = nn_similarity(ref[s].X, ref["train"].X)
            dup[s] |= set(ref[s].ids[sim >= threshold])
    return dup


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict:
    return {"acc": accuracy_score(y, pred), "bal_acc": balanced_accuracy_score(y, pred)}


class Classifier:

    def __init__(self, net: torch.nn.Module, mean: torch.Tensor, std: torch.Tensor, device: str):
        self.net, self.mean, self.std, self.device = net, mean, std, device

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        x = (torch.as_tensor(X, dtype=torch.float32, device=self.device) - self.mean) / self.std
        return torch.softmax(self.net(x), dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


def _build(d_in: int, hidden: tuple, dropout: float, n_out: int) -> torch.nn.Module:
    layers, d = [], d_in
    for h in hidden:
        layers += [torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
        d = h
    layers.append(torch.nn.Linear(d, n_out))
    return torch.nn.Sequential(*layers)


def train_classifier(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray, y_va: np.ndarray, *,
                     hidden: tuple = (), dropout: float = 0.0, l2: float = 0.0, lr: float = 1e-3,
                     batch_size: int = 256, max_epochs: int = 300, patience: int = 25, seed: int = 0,
                     device: str | None = None, sample_weight: np.ndarray | None = None) -> tuple[Classifier, float, int]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_cls = len(CLASSES)
    Xt = torch.as_tensor(X_tr, device=device)
    yt = torch.as_tensor(y_tr, device=device, dtype=torch.long)
    mean, std = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
    Xt = (Xt - mean) / std
    counts = np.bincount(y_tr, minlength=n_cls).astype(np.float32)
    w = torch.as_tensor(len(y_tr) / (n_cls * np.maximum(counts, 1)), device=device)

    net = _build(X_tr.shape[1], hidden, dropout, n_cls).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    weights = [m.weight for m in net if isinstance(m, torch.nn.Linear)]
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    sw = None if sample_weight is None else torch.as_tensor(sample_weight, device=device, dtype=torch.float32)
    clf = Classifier(net, mean, std, device)

    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(len(yt), generator=g).to(device)
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad(set_to_none=True)
            if sw is None:
                loss = loss_fn(net(Xt[idx]), yt[idx])
            else:
                per = torch.nn.functional.cross_entropy(net(Xt[idx]), yt[idx], reduction="none")
                wi = w[yt[idx]] * sw[idx]
                loss = (wi * per).sum() / wi.sum().clamp_min(1e-8)
            if l2:
                loss = loss + l2 * sum((W ** 2).sum() for W in weights)
            loss.backward()
            opt.step()
        score = balanced_accuracy_score(y_va, clf.predict(X_va))
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return clf, best, best_epoch


LINEAR_GRID = [dict(hidden=(), dropout=0.0, l2=l2, lr=lr) for l2 in [0.0, 1e-4, 1e-3, 1e-2] for lr in [1e-3, 1e-2]]
MLP_GRID = [dict(hidden=(h, h), dropout=p, l2=l2, lr=1e-3)
            for h in [512, 1024] for p in [0.2, 0.5] for l2 in [0.0, 1e-4]]
GRIDS = {"linear": LINEAR_GRID, "mlp": MLP_GRID}


def config_name(cfg: dict) -> str:
    h = "x".join(map(str, cfg["hidden"])) or "none"
    return f"hidden={h},dropout={cfg['dropout']},l2={cfg['l2']},lr={cfg['lr']}"


def select_config(model: str, X_tr, y_tr, X_va, y_va, seed: int = 0) -> tuple[dict, list[dict]]:
    rows = []
    for cfg in GRIDS[model]:
        _, val, ep = train_classifier(X_tr, y_tr, X_va, y_va, seed=seed, **cfg)
        rows.append({"config": config_name(cfg), "val_bal_acc": val, "best_epoch": ep, **{"cfg": cfg}})
    best = max(rows, key=lambda r: r["val_bal_acc"])
    return best["cfg"], [{k: v for k, v in r.items() if k != "cfg"} for r in rows]
