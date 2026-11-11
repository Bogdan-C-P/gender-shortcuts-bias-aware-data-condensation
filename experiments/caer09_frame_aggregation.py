import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES
from dcond.caer_eval import config_name, select_config, train_classifier
from dcond.caer_frames import DEDUP_JSON, EVAL, MULTI_DIR, SPLIT_NAMES, frame_sets, load_dedup, load_ft, load_multi, mean_rep, uniform_idx
from dcond.plotting import INK_2, SURFACE, apply_style, plt

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "caer09_frame_aggregation"
MEAN_MODELS = ["mean_linear", "mean_mlp", "fer_conf_linear", "fer_expr_linear"]
SET_MODELS = ["attn_linear", "attn_mlp", "transformer"]
MODELS = MEAN_MODELS + SET_MODELS
NEUTRAL = CLASSES.index("Neutral")
TEST_K = [1, 3, 5, 15]


class SetModel(torch.nn.Module):
    def __init__(self, d: int, kind: str, dropout: float, k: int = 15, n_cls: int = len(CLASSES)):
        super().__init__()
        self.kind = kind
        if kind.startswith("attn"):
            self.V, self.U, self.w = torch.nn.Linear(d, 128), torch.nn.Linear(d, 128), torch.nn.Linear(128, 1)
            hid = [torch.nn.Linear(d, 512), torch.nn.ReLU(), torch.nn.Dropout(dropout)] if kind == "attn_mlp" else []
            self.head = torch.nn.Sequential(torch.nn.Dropout(dropout), *hid, torch.nn.Linear(512 if hid else d, n_cls))
        else:
            self.proj = torch.nn.Linear(d, 256)
            self.pos = torch.nn.Parameter(torch.zeros(1, k, 256))
            self.cls = torch.nn.Parameter(torch.zeros(1, 1, 256))
            layer = torch.nn.TransformerEncoderLayer(256, 4, 512, dropout, batch_first=True, norm_first=True)
            self.enc = torch.nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = torch.nn.Sequential(torch.nn.LayerNorm(256), torch.nn.Linear(256, n_cls))

    def forward(self, x, mask):
        if self.kind.startswith("attn"):
            a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1).masked_fill(~mask, -1e4)
            a = torch.softmax(a, dim=1)
            return self.head((a[..., None] * x).sum(1)), a
        h = torch.cat([self.cls.expand(len(x), -1, -1), self.proj(x) + self.pos], dim=1)
        pad = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), ~mask], dim=1)
        return self.head(self.enc(h, src_key_padding_mask=pad)[:, 0]), None


class SetClassifier:
    def __init__(self, net, mean, std, device):
        self.net, self.mean, self.std, self.device = net, mean, std, device

    @torch.no_grad()
    def predict(self, X, mask, bs: int = 1024):
        self.net.eval()
        preds, attn = [], []
        for i in range(0, len(X), bs):
            x = (torch.as_tensor(X[i : i + bs], device=self.device) - self.mean) / self.std
            m = torch.as_tensor(mask[i : i + bs], device=self.device)
            lg, a = self.net(x * m[..., None], m)
            preds.append(lg.argmax(1).cpu().numpy())
            attn.append(a.cpu().numpy() if a is not None else np.zeros(m.shape, np.float32))
        return np.concatenate(preds), np.concatenate(attn)


def train_set_model(kind, tr, va, *, dropout, lr, seed, frame_drop=0.3, bs=128, max_epochs=150, patience=15):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    X = torch.as_tensor(tr.X, device=dev)
    V = torch.as_tensor(tr.valid, device=dev)
    frames = X[V]
    mean, std = frames.mean(0), frames.std(0).clamp_min(1e-6)
    X = (X - mean) / std * V[..., None]
    y = torch.as_tensor(tr.y, device=dev, dtype=torch.long)
    counts = np.bincount(tr.y, minlength=len(CLASSES)).astype(np.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(len(tr.y) / (len(CLASSES) * np.maximum(counts, 1)), device=dev))
    net = SetModel(X.shape[-1], kind, dropout, k=X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    clf = SetClassifier(net, mean, std, dev)
    Xva, Mva, yva = va.X[va.keep], va.valid[va.keep], va.y[va.keep]
    best, best_state, bad = -1.0, None, 0
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(len(y), generator=g).to(dev)
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            m = V[idx] & (torch.rand(V[idx].shape, device=dev) > frame_drop)
            m = torch.where(m.any(1, keepdim=True), m, V[idx])
            opt.zero_grad(set_to_none=True)
            lg, _ = net(X[idx] * m[..., None], m)
            loss_fn(lg, y[idx]).backward()
            opt.step()
        score = balanced_accuracy_score(yva, clf.predict(Xva, Mva)[0])
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return clf, best


def soft_weights(score: np.ndarray, valid: np.ndarray, temp: float = 0.1) -> np.ndarray:
    s = np.where(valid & np.isfinite(score), score / temp, -np.inf)
    s = s - s.max(1, keepdims=True)
    w = np.exp(s)
    w = np.nan_to_num(w / w.sum(1, keepdims=True))
    none = w.sum(1) == 0
    w[none] = valid[none]
    return w


def model_weights(model: str, f) -> np.ndarray:
    if model == "fer_conf_linear":
        return soft_weights(np.nanmax(f.fer_p, -1), f.valid & f.extra["face_ok"])
    if model == "fer_expr_linear":
        return soft_weights(1.0 - f.fer_p[..., NEUTRAL], f.valid & f.extra["face_ok"])
    return f.valid.astype(np.float32)


def test_masks(valid: np.ndarray) -> dict[int, np.ndarray]:
    from experiments.caer08_frame_budget import uniform_mask

    return {k: (valid.copy() if k >= valid.shape[1] else uniform_mask(valid, k)) for k in TEST_K}


def attention_stats(variant, model, seed, te, attn, pred) -> dict:
    valid = te.valid
    n_valid = valid.sum(1)
    a = attn * valid
    ent = -(np.where(a > 0, a * np.log(np.maximum(a, 1e-12)), 0)).sum(1)
    ok = te.extra["face_ok"] & valid
    fer_pred = np.where(ok, np.nan_to_num(te.fer_p, nan=-1).argmax(-1), -1)
    out = {"variant": variant, "model": model, "seed": seed,
           "effective_frames_mean": float(np.exp(ent).mean()), "valid_frames_mean": float(n_valid.mean())}
    for name, target in [("label", te.y), ("predicted", pred)]:
        shows = (fer_pred == target[:, None]) & ok
        mixed = shows.any(1) & (~shows & valid).any(1)
        ratio = (a[mixed] * shows[mixed]).sum(1) / shows[mixed].sum(1) * n_valid[mixed]
        out[f"attn_ratio_frames_showing_{name}"] = float(ratio.mean())
        out[f"n_clips_mixed_{name}"] = int(mixed.sum())
    p_lab = np.take_along_axis(np.nan_to_num(te.fer_p, nan=0.0), te.y[:, None, None], axis=2)[..., 0]
    rhos = [spearmanr(a[i][ok[i]], p_lab[i][ok[i]])[0] for i in np.flatnonzero(ok.sum(1) >= 5)]
    out["spearman_attn_vs_fer_p_label_mean"] = float(np.nanmean(rhos)) if rhos else np.nan
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-dir", type=Path, default=MULTI_DIR)
    parser.add_argument("--variants", nargs="+", default=["full", "face", "both"], choices=["full", "face", "both"])
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--ft-dir", type=Path, default=None,
                        help="use fine-tuned embeddings (data/caer_ft/<variant>_<loss>_s<seed>, caer12) instead of "
                             "frozen CLIP; pass the matching single --variants full|face")
    parser.add_argument("--dedup-ids", type=Path, default=DEDUP_JSON)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dup = load_dedup(args.dedup_ids)
    if args.ft_dir:
        assert len(args.variants) == 1 and args.variants[0] in args.ft_dir.name, "one variant, matching the --ft-dir"
        data = {s: load_ft(s, args.ft_dir) for s in SPLIT_NAMES}
        print(f"fine-tuned embeddings from {args.ft_dir}")
    else:
        data = {s: load_multi(s, 15, args.multi_dir) for s in SPLIT_NAMES}

    rows, att_rows, pos_rows = [], [], []
    for variant in args.variants:
        fs = frame_sets(data, variant, dup)
        tr, va, te = fs["train"], fs["validation"], fs["test"]
        tmask = test_masks(te.valid)
        print(f"\n=== {variant}: train {len(tr.y)}, val dedup {va.keep.sum()}, test dedup {te.keep.sum()}/{len(te.y)}, dim {tr.X.shape[-1]}")
        for model in args.models:
            if model in MEAN_MODELS:
                kind = "mlp" if model == "mean_mlp" else "linear"
                Xtr = mean_rep(tr.X, model_weights(model, tr))
                Xva = mean_rep(va.X, model_weights(model, va))[va.keep]
                cfg, _ = select_config(kind, Xtr, tr.y, Xva, va.y[va.keep])
                desc = config_name(cfg)
            else:
                grid = [dict(dropout=p, lr=lr) for p in [0.2, 0.5] for lr in ([3e-4, 1e-4] if model == "transformer" else [1e-3, 3e-4])]
                scores = [train_set_model(model, tr, va, seed=0, **c)[1] for c in grid]
                cfg = grid[int(np.argmax(scores))]
                desc = f"dropout={cfg['dropout']},lr={cfg['lr']}"
            for seed in range(args.seeds):
                if model in MEAN_MODELS:
                    clf, _, _ = train_classifier(Xtr, tr.y, Xva, va.y[va.keep], seed=seed, **cfg)
                else:
                    clf, _ = train_set_model(model, tr, va, seed=seed, **cfg)
                for k, m in tmask.items():
                    if model in MEAN_MODELS:
                        w = model_weights(model, te) * m
                        w = np.where(w.sum(1, keepdims=True) > 0, w, m)
                        pred, attn = clf.predict(mean_rep(te.X, w)), None
                    else:
                        pred, attn = clf.predict(te.X, m)
                    rows.append({"variant": variant, "model": model, "config": desc, "seed": seed, "test_k": k,
                                 "test_all_bal_acc": balanced_accuracy_score(te.y, pred),
                                 "test_all_acc": accuracy_score(te.y, pred),
                                 "test_dedup_bal_acc": balanced_accuracy_score(te.y[te.keep], pred[te.keep]),
                                 "test_dedup_acc": accuracy_score(te.y[te.keep], pred[te.keep])})
                    if k == 15 and model.startswith("attn"):
                        att_rows.append(attention_stats(variant, model, seed, te, attn, pred))
                        a = attn * te.valid
                        prof = a.sum(0) / np.maximum(te.valid.sum(0), 1) * te.valid.sum(1).mean()
                        pos_rows += [{"variant": variant, "model": model, "seed": seed, "position": j,
                                      "attention_x_n_frames": float(prof[j])} for j in range(len(prof))]
            r = pd.DataFrame([x for x in rows if x["variant"] == variant and x["model"] == model])
            line = r.groupby("test_k")["test_dedup_bal_acc"].mean()
            print(f"  {model:16s} [{desc}] dedup bal acc by test frames: " + ", ".join(f"k={k} {line[k]:.3f}" for k in TEST_K))

    runs = pd.DataFrame(rows)
    runs.to_csv(args.out_dir / "runs.csv", index=False)
    agg = runs.groupby(["variant", "model", "config", "test_k"], sort=False)[
        ["test_all_bal_acc", "test_all_acc", "test_dedup_bal_acc", "test_dedup_acc"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b.replace('std', 'sd')}" for a, b in agg.columns]
    summary = agg.reset_index()
    for v in summary["variant"].unique():
        b = summary[(summary["variant"] == v) & (summary["model"] == "mean_linear") & (summary["test_k"] == 15)]
        if len(b):
            sv = summary["variant"] == v
            summary.loc[sv, "delta_vs_mean_linear_k15"] = summary.loc[sv, "test_dedup_bal_acc_mean"] - float(b["test_dedup_bal_acc_mean"].iloc[0])
    summary.round(4).to_csv(args.out_dir / "summary.csv", index=False)
    att = pd.DataFrame(att_rows)
    pos = pd.DataFrame(pos_rows)
    if len(att):
        att.groupby(["variant", "model"]).mean(numeric_only=True).drop(columns="seed").round(4).to_csv(args.out_dir / "attention.csv")
        pos.groupby(["variant", "model", "position"]).mean(numeric_only=True).drop(columns="seed").round(4).to_csv(args.out_dir / "attention_position.csv")
        print("\nattention (test, 15 frames; ratio 1.0 = no preference for frames where fer_vit sees the label):")
        print(att.groupby(["variant", "model"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())
    figure(summary, args.out_dir / "aggregation.png")
    if len(pos):
        attention_figure(att, pos, args.out_dir / "attention.png")
    pd.set_option("display.width", 250)
    print("\ndedup balanced accuracy, mean over seeds (rows = model, cols = test frames):")
    print(summary.pivot_table(index=["variant", "model"], columns="test_k", values="test_dedup_bal_acc_mean", sort=False).round(3).to_string())
    print(f"\n-> {args.out_dir}")


def figure(summary: pd.DataFrame, out: Path) -> None:
    variants = list(summary["variant"].unique())
    names = {"full": "full frames", "face": "face crops", "both": "full frame + face"}
    colors = {k: c for k, c in zip(TEST_K, ["#cde2fb", "#9ec5f4", "#3987e5", "#0d366b"])}
    fig, axes = plt.subplots(len(variants), 1, figsize=(10, 2.9 * len(variants)), sharey=True, squeeze=False)
    for ax, v in zip(axes[:, 0], variants):
        d = summary[summary["variant"] == v]
        models = list(dict.fromkeys(d["model"]))
        for i, k in enumerate(TEST_K):
            dk = d[d["test_k"] == k].set_index("model").reindex(models)
            xs = np.arange(len(models)) + (i - 1.5) * 0.2
            ax.bar(xs, dk["test_dedup_bal_acc_mean"], width=0.19, color=colors[k], edgecolor=SURFACE, linewidth=0.8,
                   yerr=dk["test_dedup_bal_acc_sd"], error_kw=dict(ecolor=INK_2, elinewidth=0.7, capsize=1.5), label=f"test on {k} frame{'s' if k > 1 else ''}")
            if k == 15:
                for x, val in zip(xs, dk["test_dedup_bal_acc_mean"]):
                    ax.text(x, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=6, color=INK_2)
        base = d[(d["model"] == "mean_linear") & (d["test_k"] == 15)]["test_dedup_bal_acc_mean"]
        if len(base):
            ax.axhline(float(base.iloc[0]), color=INK_2, linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(models)), models)
        ax.set_ylim(0.1, None)
        ax.set_ylabel(f"balanced acc ({EVAL['short']})")
        ax.set_title(f"{names[v]} — dashed: mean of 15 frames + linear probe", loc="left", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(TEST_K), frameon=False, fontsize=7)
    fig.suptitle("CAER — aggregating a clip's frames: mean vs weighted vs learned attention (mean ± sd over seeds)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def attention_figure(att: pd.DataFrame, pos: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#e34948", "#898781"]
    keys = list(pos.groupby(["variant", "model"]).groups)
    for c, (v, m) in zip(colors, keys):
        d = pos[(pos["variant"] == v) & (pos["model"] == m)].groupby("position")["attention_x_n_frames"].mean()
        axes[0].plot(d.index, d.values, marker="o", markersize=3, color=c, linewidth=1.4, label=f"{v} / {m}")
    axes[0].axhline(1.0, color=INK_2, linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("frame position in the clip (0 = start, 7 = middle)")
    axes[0].set_ylabel("attention × number of frames\n(1 = uniform)")
    axes[0].set_title("where in the clip does attention go?", loc="left", fontsize=9)
    axes[0].legend(fontsize=6)
    a = att.groupby(["variant", "model"]).mean(numeric_only=True).reset_index()
    labels = [f"{v}\n{m}" for v, m in zip(a["variant"], a["model"])]
    xs = np.arange(len(a))
    axes[1].bar(xs - 0.2, a["attn_ratio_frames_showing_label"], width=0.38, color="#2a78d6", edgecolor=SURFACE, label="frames where fer_vit sees the LABEL")
    axes[1].bar(xs + 0.2, a["attn_ratio_frames_showing_predicted"], width=0.38, color="#eb6834", edgecolor=SURFACE, label="frames where fer_vit sees the PREDICTED class")
    axes[1].axhline(1.0, color=INK_2, linestyle="--", linewidth=0.8)
    axes[1].set_xticks(xs, labels, fontsize=7)
    axes[1].set_ylabel("their attention / uniform")
    axes[1].set_title("does attention favour frames that show an emotion?", loc="left", fontsize=9)
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
