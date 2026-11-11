import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES
from dcond.caer_eval import config_name, train_classifier
from dcond.caer_frames import (DEDUP_JSON, EVAL, MULTI_DIR, SPLIT_NAMES, CosineHead, frame_sets, load_dedup, load_multi, mean_rep,
                               mid_idx, ncm_predict, nn1_predict)
from dcond.plotting import INK_2, SURFACE, apply_style, plt

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "caer11_mlp_features"
PROBE_GRID = [dict(hidden=(), dropout=0.0, l2=l2, lr=lr) for l2 in [0.0, 1e-3] for lr in [1e-3, 1e-2]]
LOSS_COLORS = {"ce": "#2a78d6", "arcface": "#eb6834", "normface": "#1baf7a"}


class FeatureMLP(torch.nn.Module):
    def __init__(self, d_in: int, depth: int, loss: str, width: int = 1024, dropout: float = 0.2, emb: int = 256,
                 s: float = 16.0, m: float = 0.2):
        super().__init__()
        self.blocks = torch.nn.ModuleList()
        d = d_in
        for _ in range(depth):
            self.blocks.append(torch.nn.Sequential(torch.nn.Linear(d, width), torch.nn.ReLU()))
            d = width
        self.drop = torch.nn.Dropout(dropout)
        self.loss = loss
        if loss == "ce":
            self.emb, self.head = None, torch.nn.Linear(d, len(CLASSES))
        else:
            self.emb = torch.nn.Linear(d, emb)
            self.head = CosineHead(emb, len(CLASSES), s=s, m=m if loss == "arcface" else 0.0)

    def forward(self, x, y=None, return_feats=False):
        feats = []
        for b in self.blocks:
            x = b(x)
            feats.append(x)
            x = self.drop(x)
        if self.emb is None:
            out = self.head(x)
        else:
            e = self.emb(x)
            feats.append(e)
            out = self.head(e, y)
        return (out, feats) if return_feats else out


def train_mlp(tr, va, depth, loss, train_frames, seed, lr=1e-3, max_epochs=None, patience=None):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    if train_frames == "mid":
        X, y = tr.X[np.arange(len(tr.y)), mid_idx(tr.valid)], tr.y
    else:
        ii, jj = np.nonzero(tr.valid)
        X, y = tr.X[ii, jj], tr.y[ii]
    Xt = torch.as_tensor(X, device=dev)
    mean, std = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
    Xt = (Xt - mean) / std
    yt = torch.as_tensor(y, device=dev, dtype=torch.long)
    counts = np.bincount(y, minlength=len(CLASSES)).astype(np.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(len(y) / (len(CLASSES) * np.maximum(counts, 1)), device=dev))
    net = FeatureMLP(X.shape[1], depth, loss).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bs = 256 if train_frames == "mid" else 1024
    max_epochs = max_epochs or (300 if train_frames == "mid" else 80)
    patience = patience or (25 if train_frames == "mid" else 10)
    best, best_state, bad, ep_best = -1.0, None, 0, 0
    for epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(len(yt), generator=g).to(dev)
        for i in range(0, len(perm), bs):
            idx = perm[i : i + bs]
            opt.zero_grad(set_to_none=True)
            loss_fn(net(Xt[idx], yt[idx]), yt[idx]).backward()
            opt.step()
        net.eval()
        score = balanced_accuracy_score(va.y[va.keep], head_predict(net, mean, std, va.X[va.keep], va.valid[va.keep], "mean"))
        if score > best:
            best, bad, ep_best = score, 0, epoch
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, mean, std, best, ep_best


@torch.no_grad()
def run_frames(net, mean, std, X: np.ndarray, bs: int = 8192):
    dev = mean.device
    n, k, d = X.shape
    flat = X.reshape(-1, d)
    logits, feats = [], None
    for i in range(0, len(flat), bs):
        x = (torch.as_tensor(flat[i : i + bs], device=dev) - mean) / std
        lg, fs = net(x, return_feats=True)
        logits.append(lg.float().cpu().numpy())
        if feats is None:
            feats = [[] for _ in fs]
        for j, f in enumerate(fs):
            feats[j].append(f.float().cpu().numpy())
    return (np.concatenate(logits).reshape(n, k, -1), [np.concatenate(f).reshape(n, k, -1) for f in feats])


def head_predict(net, mean, std, X, valid, how):
    lg, _ = run_frames(net, mean, std, X)
    if how == "mid":
        return lg[np.arange(len(lg)), mid_idx(valid)].argmax(1)
    p = np.exp(lg - lg.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return (p * valid[..., None]).sum(1).argmax(1)


def evaluate_features(H: dict, fs: dict, seed: int) -> list[dict]:
    tr, va, te = fs["train"], fs["validation"], fs["test"]
    reps = {"mid": {s: H[s][np.arange(len(f.y)), mid_idx(f.valid)] for s, f in fs.items()},
            "mean15": {s: mean_rep(H[s], f.valid) for s, f in fs.items()}}
    out = []

    def add(rep, ev, pred, extra=""):
        out.append({"rep": rep, "eval": ev, "probe_config": extra,
                    "test_all_bal_acc": balanced_accuracy_score(te.y, pred), "test_all_acc": accuracy_score(te.y, pred),
                    "test_dedup_bal_acc": balanced_accuracy_score(te.y[te.keep], pred[te.keep]),
                    "test_dedup_acc": accuracy_score(te.y[te.keep], pred[te.keep])})

    R = reps["mean15"]
    scores = [train_classifier(R["train"], tr.y, R["validation"][va.keep], va.y[va.keep], seed=seed, **c)[1] for c in PROBE_GRID]
    cfg = PROBE_GRID[int(np.argmax(scores))]
    for rep, R in reps.items():
        clf, _, _ = train_classifier(R["train"], tr.y, R["validation"][va.keep], va.y[va.keep], seed=seed, **cfg)
        add(rep, "linear_probe", clf.predict(R["test"]), config_name(cfg))
    R = reps["mean15"]
    add("mean15", "ncm", ncm_predict(R["train"], tr.y, R["test"]))
    add("mean15", "nn1", nn1_predict(R["train"], tr.y, R["test"]))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-dir", type=Path, default=MULTI_DIR)
    parser.add_argument("--variants", nargs="+", default=["full", "face"], choices=["full", "face", "both"])
    parser.add_argument("--train-frames", nargs="+", default=["mid", "all"], choices=["mid", "all"])
    parser.add_argument("--losses", nargs="+", default=["ce", "arcface"], choices=["ce", "arcface", "normface"])
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--dedup-ids", type=Path, default=DEDUP_JSON)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dup = load_dedup(args.dedup_ids)
    data = {s: load_multi(s, 15, args.multi_dir) for s in SPLIT_NAMES}

    rows = []
    for variant in args.variants:
        fs = frame_sets(data, variant, dup)
        tr, va, te = fs["train"], fs["validation"], fs["test"]
        print(f"\n=== {variant}: train {len(tr.y)} clips / {tr.valid.sum()} frames, val dedup {va.keep.sum()}, "
              f"test dedup {te.keep.sum()}/{len(te.y)}")
        for seed in range(args.seeds):
            for r in evaluate_features({s: f.X for s, f in fs.items()}, fs, seed):
                rows.append({"variant": variant, "train_frames": "-", "loss": "-", "depth": 0, "seed": seed, "layer": "clip", **r})
        base = pd.DataFrame([x for x in rows if x["variant"] == variant and x["layer"] == "clip"])
        print("  CLIP (layer 0): " + ", ".join(f"{a}/{b} {v:.3f}" for (a, b), v in
                                               base.groupby(["eval", "rep"])["test_dedup_bal_acc"].mean().items()))
        for tf in args.train_frames:
            for loss in args.losses:
                for depth in args.depths:
                    for seed in range(args.seeds):
                        net, mean, std, val, ep = train_mlp(tr, va, depth, loss, tf, seed)
                        common = {"variant": variant, "train_frames": tf, "loss": loss, "depth": depth, "seed": seed,
                                  "mlp_val_bal_acc": val, "mlp_best_epoch": ep}
                        for how in ["mid", "mean"]:
                            pred = head_predict(net, mean, std, te.X, te.valid, how)
                            rows.append({**common, "layer": "head", "rep": "mid" if how == "mid" else "mean15", "eval": "mlp_head",
                                         "test_all_bal_acc": balanced_accuracy_score(te.y, pred),
                                         "test_all_acc": accuracy_score(te.y, pred),
                                         "test_dedup_bal_acc": balanced_accuracy_score(te.y[te.keep], pred[te.keep]),
                                         "test_dedup_acc": accuracy_score(te.y[te.keep], pred[te.keep])})
                        feats = {s: run_frames(net, mean, std, f.X)[1] for s, f in fs.items()}
                        names = [f"h{j + 1}" for j in range(depth)] + (["emb"] if loss != "ce" else [])
                        for j, name in enumerate(names):
                            for r in evaluate_features({s: feats[s][j] for s in fs}, fs, seed):
                                rows.append({**common, "layer": name, **r})
                        del feats
                    d = pd.DataFrame([x for x in rows if x["variant"] == variant and x["train_frames"] == tf and x["loss"] == loss
                                      and x["depth"] == depth])
                    m = d.groupby(["layer", "eval", "rep"], sort=False)["test_dedup_bal_acc"].mean()
                    print(f"  {tf}/{loss}/depth {depth}: " + ", ".join(f"{l}:{e}/{r} {v:.3f}" for (l, e, r), v in m.items()
                                                                     if (e, r) in [("linear_probe", "mean15"), ("mlp_head", "mean15"), ("ncm", "mean15")]))

    runs = pd.DataFrame(rows)
    runs.to_csv(args.out_dir / "runs.csv", index=False)
    keys = ["variant", "train_frames", "loss", "depth", "layer", "eval", "rep"]
    agg = runs.groupby(keys, sort=False)[["test_all_bal_acc", "test_all_acc", "test_dedup_bal_acc", "test_dedup_acc"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b.replace('std', 'sd')}" for a, b in agg.columns]
    summary = agg.reset_index()
    for v in summary["variant"].unique():
        for (ev, rep), _ in summary[summary["variant"] == v].groupby(["eval", "rep"]):
            b = summary[(summary["variant"] == v) & (summary["layer"] == "clip") & (summary["eval"] == ev) & (summary["rep"] == rep)]
            if len(b):
                sel = (summary["variant"] == v) & (summary["eval"] == ev) & (summary["rep"] == rep)
                summary.loc[sel, "delta_vs_clip"] = summary.loc[sel, "test_dedup_bal_acc_mean"] - float(b["test_dedup_bal_acc_mean"].iloc[0])
    summary.round(4).to_csv(args.out_dir / "summary.csv", index=False)
    figure(summary, args.out_dir / "mlp_features.png")
    pd.set_option("display.width", 250)
    print("\nlinear probe on the 15-frame mean, dedup balanced accuracy (mean over seeds):")
    s = summary[(summary["eval"] == "linear_probe") & (summary["rep"] == "mean15")]
    print(s.pivot_table(index=["variant", "train_frames", "loss", "depth"], columns="layer", values="test_dedup_bal_acc_mean",
                        sort=False).round(3).to_string())
    print(f"\n-> {args.out_dir}")


def figure(summary: pd.DataFrame, out: Path) -> None:
    s = summary[(summary["eval"] == "linear_probe")]
    variants = list(s["variant"].unique())
    tfs = [t for t in s["train_frames"].unique() if t != "-"]
    names = {"full": "full frames", "face": "face crops", "both": "full frame + face"}
    fig, axes = plt.subplots(len(variants), len(tfs), figsize=(6.2 * len(tfs), 3.3 * len(variants)), sharey=True, squeeze=False)
    for r, v in enumerate(variants):
        clip = s[(s["variant"] == v) & (s["layer"] == "clip")].set_index("rep")["test_dedup_bal_acc_mean"]
        for c, tf in enumerate(tfs):
            ax = axes[r][c]
            d = s[(s["variant"] == v) & (s["train_frames"] == tf)]
            cats = list(dict.fromkeys(f"d{dp}:{l}" for dp, l in zip(d["depth"], d["layer"])))
            cats = sorted(cats, key=lambda x: (int(x[1]), x.split(":")[1] == "emb", x))
            losses = list(dict.fromkeys(d["loss"]))
            for i, loss in enumerate(losses):
                dl = d[(d["loss"] == loss)]
                for rep, mk in [("mean15", None), ("mid", "_")]:
                    dr = dl[dl["rep"] == rep].assign(cat=lambda z: [f"d{dp}:{l}" for dp, l in zip(z["depth"], z["layer"])]).set_index("cat").reindex(cats)
                    xs = np.arange(len(cats)) + (i - (len(losses) - 1) / 2) * 0.36
                    if mk is None:
                        ax.bar(xs, dr["test_dedup_bal_acc_mean"], width=0.34, color=LOSS_COLORS[loss], edgecolor=SURFACE,
                               yerr=dr["test_dedup_bal_acc_sd"], error_kw=dict(ecolor=INK_2, elinewidth=0.7, capsize=1.5),
                               label=f"{loss}: probe on mean of 15 frames")
                    else:
                        ax.scatter(xs, dr["test_dedup_bal_acc_mean"], marker="_", s=90, color=INK_2, zorder=3,
                                   label="probe on the middle frame" if i == 0 else None)
            for rep, ls in [("mean15", "--"), ("mid", ":")]:
                if rep in clip:
                    ax.axhline(clip[rep], color=INK_2, linestyle=ls, linewidth=0.9, label=f"CLIP features, {rep}")
            ax.set_xticks(range(len(cats)), cats, fontsize=7)
            ax.set_ylim(0.2, None)
            ax.set_title(f"{names[v]} — MLP trained on {'middle frames' if tf == 'mid' else 'all frames'}", loc="left", fontsize=9)
        axes[r][0].set_ylabel(f"balanced acc ({EVAL['short']})")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=7)
    fig.suptitle("CAER — linear probe on features after d nonlinear layers (dX:hY = layer Y of a depth-X MLP; emb = ArcFace embedding)",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
