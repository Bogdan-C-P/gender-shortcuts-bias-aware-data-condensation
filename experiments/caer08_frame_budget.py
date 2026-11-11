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

from dcond.caer_eval import config_name, select_config, train_classifier
from dcond.caer_frames import DEDUP_JSON, EVAL, MULTI_DIR, SPLIT_NAMES, frame_sets, load_dedup, load_multi, mean_rep, uniform_idx
from dcond.plotting import INK_2, apply_style, plt

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "caer08_frame_budget"
SELECTIONS = ["uniform", "random", "medoid"]
SEL_COLORS = {"uniform": "#2a78d6", "random": "#eb6834", "medoid": "#1baf7a"}


def uniform_mask(valid: np.ndarray, k: int) -> np.ndarray:
    n, K = valid.shape
    mask = np.zeros_like(valid)
    if valid.all():
        mask[:, uniform_idx(min(k, K), K)] = True
        return mask
    for i in range(n):
        v = np.flatnonzero(valid[i])
        mask[i, v if len(v) <= k else v[uniform_idx(k, len(v))]] = True
    return mask


def random_mask(valid: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    r = rng.random(valid.shape)
    r[~valid] = -1
    top = np.argsort(-r, axis=1)[:, :k]
    mask = np.zeros_like(valid)
    mask[np.arange(len(valid))[:, None], top] = True
    return mask & valid


@torch.no_grad()
def medoid_mask(X: np.ndarray, valid: np.ndarray, k: int, iters: int = 15, chunk: int = 2048) -> np.ndarray:
    n, K, _ = X.shape
    if k >= K:
        return valid.copy()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    init = np.argsort(~uniform_mask(valid, k), axis=1, kind="stable")[:, :k]
    out = np.zeros_like(valid)
    for s in range(0, n, chunk):
        x = torch.nn.functional.normalize(torch.as_tensor(X[s : s + chunk], device=dev, dtype=torch.float32), dim=-1)
        v = torch.as_tensor(valid[s : s + chunk], device=dev)
        rows = torch.arange(len(x), device=dev)
        c = x[rows[:, None], torch.as_tensor(init[s : s + chunk], device=dev)]
        for _ in range(iters):
            sim = torch.einsum("bkd,bcd->bkc", x, c)
            a = torch.nn.functional.one_hot(sim.argmax(-1), k).float() * v[..., None]
            cnt = a.sum(1)
            new = torch.einsum("bkc,bkd->bcd", a, x)
            c = torch.where(cnt[..., None] > 0, torch.nn.functional.normalize(new, dim=-1), c)
        sim = torch.einsum("bkd,bcd->bck", x, c).masked_fill(~v[:, None, :], -9.0)
        chosen = torch.zeros_like(v)
        for j in range(k):
            idx = sim[:, j].masked_fill(chosen, -10.0).argmax(-1)
            chosen[rows, idx] = True
        m = (chosen & v).cpu().numpy()
        few = valid[s : s + chunk].sum(1) <= k
        m[few] = valid[s : s + chunk][few]
        out[s : s + chunk] = m
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-dir", type=Path, default=MULTI_DIR)
    parser.add_argument("--variants", nargs="+", default=["full", "face"], choices=["full", "face", "both"])
    parser.add_argument("--k-list", nargs="+", type=int, default=[1, 2, 3, 5, 8, 15])
    parser.add_argument("--selections", nargs="+", default=SELECTIONS, choices=SELECTIONS)
    parser.add_argument("--model", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--seeds", type=int, default=3)
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
        full_w = {s: f.valid for s, f in fs.items()}
        Xva15 = mean_rep(va.X, va.valid)[va.keep]
        cfg, _ = select_config(args.model, mean_rep(tr.X, tr.valid), tr.y, Xva15, va.y[va.keep])
        print(f"\n=== {variant}: train {len(tr.y)}, val dedup {va.keep.sum()}, test dedup {te.keep.sum()}/{len(te.y)}; "
              f"{args.model} [{config_name(cfg)}]")
        ref = {}
        for seed in range(args.seeds):
            ref[seed], _, _ = train_classifier(mean_rep(tr.X, tr.valid), tr.y, Xva15, va.y[va.keep], seed=seed, **cfg)
        medoids = {}
        for sel in args.selections:
            for k in args.k_list:
                for seed in range(args.seeds):
                    rng = np.random.default_rng(1000 * k + seed)
                    if sel == "uniform":
                        masks = {s: uniform_mask(f.valid, k) for s, f in fs.items()}
                    elif sel == "random":
                        masks = {s: random_mask(f.valid, k, rng) for s, f in fs.items()}
                    else:
                        if k not in medoids:
                            medoids[k] = {s: medoid_mask(f.X, f.valid, k) for s, f in fs.items()}
                        masks = medoids[k]
                    if k >= 15:
                        masks = full_w
                    Xte = mean_rep(te.X, masks["test"])
                    clf, _, _ = train_classifier(mean_rep(tr.X, masks["train"]), tr.y,
                                                 mean_rep(va.X, masks["validation"])[va.keep], va.y[va.keep], seed=seed, **cfg)
                    for regime, c in [("same", clf), ("test_only", ref[seed])]:
                        pred = c.predict(Xte)
                        rows.append({"variant": variant, "selection": sel, "k": k, "regime": regime, "seed": seed,
                                     "frames_used_test": float(masks["test"].sum(1).mean()),
                                     "test_all_bal_acc": balanced_accuracy_score(te.y, pred),
                                     "test_all_acc": accuracy_score(te.y, pred),
                                     "test_dedup_bal_acc": balanced_accuracy_score(te.y[te.keep], pred[te.keep]),
                                     "test_dedup_acc": accuracy_score(te.y[te.keep], pred[te.keep])})
                r = pd.DataFrame([x for x in rows if x["variant"] == variant and x["selection"] == sel and x["k"] == k])
                m = r.groupby("regime")["test_dedup_bal_acc"].mean()
                print(f"  {sel:8s} k={k:2d}: dedup bal acc same {m['same']:.3f}, test_only {m['test_only']:.3f}")

    runs = pd.DataFrame(rows)
    runs.to_csv(args.out_dir / "runs.csv", index=False)
    agg = runs.groupby(["variant", "selection", "k", "regime"], sort=False)[
        ["frames_used_test", "test_all_bal_acc", "test_all_acc", "test_dedup_bal_acc", "test_dedup_acc"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b.replace('std', 'sd')}" for a, b in agg.columns]
    summary = agg.reset_index().drop(columns="frames_used_test_sd")
    for v in summary["variant"].unique():
        sv = summary["variant"] == v
        base = summary[sv & (summary["selection"] == "uniform") & (summary["k"] == 1) & (summary["regime"] == "same")]
        top = summary[sv & (summary["k"] == max(args.k_list)) & (summary["regime"] == "same")]
        if len(base) and len(top):
            b, t = float(base["test_dedup_bal_acc_mean"].iloc[0]), float(top["test_dedup_bal_acc_mean"].iloc[0])
            summary.loc[sv, "delta_vs_k1"] = summary.loc[sv, "test_dedup_bal_acc_mean"] - b
            summary.loc[sv, "share_of_gain_k1_to_kmax"] = (summary.loc[sv, "test_dedup_bal_acc_mean"] - b) / max(t - b, 1e-9)
    summary.round(4).to_csv(args.out_dir / "summary.csv", index=False)
    figure(summary, args.out_dir / "frame_budget.png")
    pd.set_option("display.width", 250)
    print("\ndedup balanced accuracy (mean over seeds):")
    print(summary.pivot_table(index=["variant", "regime", "selection"], columns="k", values="test_dedup_bal_acc_mean",
                              sort=False).round(3).to_string())
    print(f"\n-> {args.out_dir}")


def figure(summary: pd.DataFrame, out: Path) -> None:
    variants = list(summary["variant"].unique())
    fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 3.6), sharey=True, squeeze=False)
    names = {"full": "full frames", "face": "face crops", "both": "full frame + face"}
    for ax, v in zip(axes[0], variants):
        for sel in summary["selection"].unique():
            for regime, ls in [("same", "-"), ("test_only", "--")]:
                d = summary[(summary["variant"] == v) & (summary["selection"] == sel) & (summary["regime"] == regime)].sort_values("k")
                if regime == "test_only" and sel != "uniform":
                    continue
                ax.errorbar(d["k"], d["test_dedup_bal_acc_mean"], yerr=d["test_dedup_bal_acc_sd"], color=SEL_COLORS[sel],
                            linestyle=ls, marker="o", markersize=3.5, linewidth=1.5, capsize=2,
                            label=f"{sel}" + (" (train on 15, test on k)" if regime == "test_only" else ""))
        ks = sorted(summary["k"].unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlabel("frames per clip (k)")
        ax.set_title(f"{names[v]}", loc="left")
    axes[0][0].set_ylabel(f"balanced accuracy ({EVAL['label']})")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=7)
    fig.suptitle("CAER — how many frames does a clip need? (linear probe on the mean of k frames; mean ± sd over seeds)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
