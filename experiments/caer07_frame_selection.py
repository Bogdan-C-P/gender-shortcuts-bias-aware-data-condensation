import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES
from dcond.caer_eval import config_name, select_config, train_classifier
from dcond.plotting import INK, INK_2, SURFACE, apply_style, plt

MULTI_DIR = PROJECT_ROOT / "data" / "caer_multi"
DEDUP_JSON = PROJECT_ROOT / "experiments" / "results" / "caer03_mlp_probe" / "dedup_ids.json"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "caer07_frame_selection"
TRAIN_STRATEGIES = ["mid", "random", "mean", "all", "fer_label", "fer_expressive", "fer_confident"]
TEST_STRATEGIES = ["mid", "random", "mean", "all", "fer_expressive", "fer_confident", "fer_label (oracle)"]
NEUTRAL = CLASSES.index("Neutral")
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load(split: str, k: int, multi_dir: Path) -> dict:
    with np.load(multi_dir / f"caer_{split}_k{k}.npz", allow_pickle=True) as f:
        d = {key: f[key] for key in f.files}
    lg = d["fer_logits"].astype(np.float64)
    lg = lg - np.nanmax(lg, axis=-1, keepdims=True)
    p = np.exp(lg)
    d["fer_p"] = p / p.sum(-1, keepdims=True)
    d["video_relpath"] = d["video_relpath"].astype(str)
    return d


def subset(d: dict, keep: np.ndarray) -> dict:
    return {k: v[keep] for k, v in d.items()}


def pick(strategy: str, d: dict, valid: np.ndarray, y: np.ndarray | None = None, rng=None) -> np.ndarray:
    n, k = valid.shape
    dist = np.abs(np.arange(k) - k // 2)[None, :] + np.where(valid, 0, 1000)
    mid = dist.argmin(1)
    if strategy == "mid":
        return mid
    if strategy == "random":
        r = rng.random((n, k))
        r[~valid] = -1
        return r.argmax(1)
    p = d["fer_p"]
    if strategy.startswith("fer_label"):
        score = p[np.arange(n), :, y]
    elif strategy == "fer_expressive":
        score = 1.0 - p[..., NEUTRAL]
    elif strategy == "fer_confident":
        score = p.max(-1)
    else:
        raise ValueError(strategy)
    score = np.where(valid & d["face_ok"] & np.isfinite(score), score, -np.inf)
    idx = score.argmax(1)
    none = ~np.isfinite(score.max(1))
    idx[none] = mid[none]
    return idx


def rep_mean(X: np.ndarray, valid: np.ndarray) -> np.ndarray:
    m = (X.astype(np.float32) * valid[..., None]).sum(1) / np.maximum(valid.sum(1, keepdims=True), 1)
    return (m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-8)).astype(np.float32)


def train_set(strategy, d, X, valid, rng):
    y = d["label"]
    if strategy == "mean":
        return rep_mean(X, valid), y
    if strategy == "all":
        ii, jj = np.nonzero(valid)
        return X[ii, jj].astype(np.float32), y[ii]
    idx = pick(strategy, d, valid, y, rng)
    return X[np.arange(len(y)), idx].astype(np.float32), y


def predict(strategy, clf, d, X, valid, rng) -> np.ndarray:
    y = d["label"]
    if strategy == "mean":
        return clf.predict(rep_mean(X, valid))
    if strategy == "all":
        ii, jj = np.nonzero(valid)
        prob = clf.predict_proba(X[ii, jj].astype(np.float32))
        acc = np.zeros((len(y), prob.shape[1]))
        np.add.at(acc, ii, prob)
        return acc.argmax(1)
    idx = pick(strategy.split(" ")[0], d, valid, y if strategy.startswith("fer_label") else None, rng)
    return clf.predict(X[np.arange(len(y)), idx].astype(np.float32))


def visibility(d: dict, split: str) -> pd.DataFrame:
    y, ok, p = d["label"], d["face_ok"], d["fer_p"]
    pred = np.where(ok, np.nan_to_num(p, nan=-1).argmax(-1), -1)
    k = ok.shape[1]
    has = ok.any(1)
    mid_ok = ok[:, k // 2]
    shows = pred == y[:, None]
    mean_p = np.nansum(np.where(ok[..., None], p, 0), 1) / np.maximum(ok.sum(1, keepdims=True), 1)
    rows = []
    for c, name in enumerate(CLASSES):
        m = (y == c) & has
        n_faces = ok[m].sum(1)
        distinct = np.array([len(set(pred[i][ok[i]])) for i in np.flatnonzero(m)])
        rows.append({"split": split, "class": name, "n_clips_with_face": int(m.sum()),
                     "frames_with_face_mean": float(n_faces.mean()),
                     "label_shown_middle_frame": float(shows[m & mid_ok, k // 2].mean()),
                     "label_shown_in_any_frame": float(shows[m].any(1).mean()),
                     "label_shown_in_most_frames": float((shows[m].sum(1) > n_faces / 2).mean()),
                     "expression_changes_within_clip": float((distinct >= 2).mean()),
                     "fer_zero_shot_mean_over_frames_correct": float((mean_p[m].argmax(1) == c).mean())})
    v = pd.DataFrame(rows)
    mm = has & mid_ok
    mid_pred = pred[mm, k // 2]
    print(f"  {split}: fer_vit zero-shot balanced acc — middle frame {balanced_accuracy_score(y[mm], mid_pred):.3f}, "
          f"mean over frames {balanced_accuracy_score(y[has], mean_p[has].argmax(1)):.3f}")
    return v


def heatmap(summary: pd.DataFrame, variant: str, out: Path) -> None:
    d = summary[summary["variant"] == variant].pivot(index="train", columns="test", values="test_dedup_bal_acc_mean")
    d = d.reindex(index=[t for t in TRAIN_STRATEGIES if t in d.index], columns=[t for t in TEST_STRATEGIES if t in d.columns])
    base = d.loc["mid", "mid"]
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue", SEQ)
    lo, hi = np.nanmin(d.values), np.nanmax(d.values)
    ax.imshow(d.values, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            dark = (v - lo) / max(hi - lo, 1e-9) > 0.55
            txt = f"{v:.3f}\n({v - base:+.3f})" if not (i == 0 and j == 0) else f"{v:.3f}\nbaseline"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if dark else INK)
    ax.set_xticks(range(d.shape[1]), d.columns, rotation=25, ha="right")
    ax.set_yticks(range(d.shape[0]), d.index)
    ax.grid(False)
    ax.set_xlabel("test: which frame(s) represent a clip")
    ax.set_ylabel("train: which frame(s) represent a clip")
    ax.set_title(f"CAER {'full frames' if variant == 'full' else 'face crops'} — balanced accuracy on the test clips "
                 "without near-twins\n(mean over seeds; in brackets: change vs middle frame for train and test)",
                 loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"frame_selection_{variant}.png", dpi=150)
    plt.close(fig)


def visibility_figure(v: pd.DataFrame, out: Path) -> None:
    d = v[v["split"] == "train"].set_index("class").reindex(CLASSES)
    cols = [("label_shown_middle_frame", "middle frame", "#2a78d6"), ("label_shown_in_any_frame", "at least one of 15 frames", "#eb6834"),
            ("label_shown_in_most_frames", "most frames", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    x = np.arange(len(CLASSES))
    for i, (c, lab, color) in enumerate(cols):
        xs = x + (i - 1) * 0.27
        ax.bar(xs, d[c], width=0.25, color=color, edgecolor=SURFACE, linewidth=1, label=lab)
        for xx, val in zip(xs, d[c]):
            ax.text(xx, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=6, color=INK_2)
    ax.set_xticks(x, CLASSES)
    ax.set_ylabel("share of train clips")
    ax.set_title("CAER — does fer_vit see the labelled emotion on the face? (train clips with a face)", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=3, loc="lower center")
    ax.set_ylim(0, 1.08)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "visibility.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi-dir", type=Path, default=MULTI_DIR)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--variants", nargs="+", default=["full", "face"], choices=["full", "face"])
    parser.add_argument("--train-strategies", nargs="+", default=TRAIN_STRATEGIES, choices=TRAIN_STRATEGIES)
    parser.add_argument("--model", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-epochs-all", type=int, default=100, help="epochs cap when every frame is a sample")
    parser.add_argument("--dedup-ids", type=Path, default=DEDUP_JSON)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dup = json.loads(args.dedup_ids.read_text()) if args.dedup_ids.exists() else {"validation": [], "test": []}
    data = {s: load(s, args.k, args.multi_dir) for s in ["train", "validation", "test"]}

    print("1. visibility of the labelled emotion (fer_vit):")
    vis = pd.concat([visibility(data[s], s) for s in ["train", "test"]])
    vis.round(4).to_csv(args.out_dir / "visibility.csv", index=False)
    print(vis[vis["split"] == "train"].round(3).to_string(index=False))
    visibility_figure(vis, args.out_dir)
    tr = data["train"]
    chosen = pick("fer_label", tr, tr["face_ok"], tr["label"])
    has = tr["face_ok"].any(1)
    pos = pd.crosstab(pd.Series(np.array(CLASSES)[tr["label"][has]], name="class"),
                      pd.Series(chosen[has], name="frame position (0..K-1)"), normalize="index")
    pos.round(3).to_csv(args.out_dir / "chosen_position.csv")
    print("\n  position of the frame that shows the label best (train, share per class):\n", pos.round(2).to_string())

    rows = []
    for variant in args.variants:
        key = "clip_full" if variant == "full" else "clip_face"
        ds = {}
        for s, d in data.items():
            keep = np.ones(len(d["label"]), bool) if variant == "full" else d["face_ok"].any(1)
            ds[s] = subset(d, keep)
        valid = {s: (np.ones_like(d["face_ok"]) if variant == "full" else d["face_ok"]) for s, d in ds.items()}
        va, te = ds["validation"], ds["test"]
        va_keep = ~np.isin(va["video_relpath"], dup["validation"])
        te_keep = ~np.isin(te["video_relpath"], dup["test"])
        va_idx = pick("mid", va, valid["validation"])
        Xva = va[key][np.arange(len(va["label"])), va_idx].astype(np.float32)[va_keep]
        yva = va["label"][va_keep]
        Xmid, ymid = train_set("mid", ds["train"], ds["train"][key], valid["train"], None)
        cfg, _ = select_config(args.model, Xmid, ymid, Xva, yva)
        print(f"\n=== {variant}: {len(ymid)} train clips, test dedup {te_keep.sum()}/{len(te_keep)}, "
              f"{args.model} [{config_name(cfg)}]")
        for ts in args.train_strategies:
            for seed in range(args.seeds):
                rng = np.random.default_rng(seed)
                Xtr, ytr = train_set(ts, ds["train"], ds["train"][key], valid["train"], rng)
                kw = dict(cfg, max_epochs=args.max_epochs_all) if ts == "all" else cfg
                clf, _, _ = train_classifier(Xtr, ytr, Xva, yva, seed=seed, **kw)
                for tst in TEST_STRATEGIES:
                    pred = predict(tst, clf, te, te[key], valid["test"], np.random.default_rng(100 + seed))
                    y = te["label"]
                    rows.append({"variant": variant, "train": ts, "test": tst, "seed": seed, "n_train_samples": len(ytr),
                                 "test_all_bal_acc": balanced_accuracy_score(y, pred),
                                 "test_dedup_bal_acc": balanced_accuracy_score(y[te_keep], pred[te_keep]),
                                 "test_dedup_acc": accuracy_score(y[te_keep], pred[te_keep])})
            r = pd.DataFrame([x for x in rows if x["variant"] == variant and x["train"] == ts])
            line = r.groupby("test")["test_dedup_bal_acc"].mean()
            print(f"  train {ts:15s} dedup bal acc: " + ", ".join(f"{t} {line[t]:.3f}" for t in TEST_STRATEGIES))

    runs = pd.DataFrame(rows)
    runs.to_csv(args.out_dir / "runs.csv", index=False)
    agg = runs.groupby(["variant", "train", "test"], sort=False)[["test_all_bal_acc", "test_dedup_bal_acc", "test_dedup_acc"]] \
        .agg(["mean", "std"])
    agg.columns = [f"{a}_{b.replace('std', 'sd')}" for a, b in agg.columns]
    summary = agg.reset_index()
    for v in summary["variant"].unique():
        base = summary[(summary["variant"] == v) & (summary["train"] == "mid") & (summary["test"] == "mid")]["test_dedup_bal_acc_mean"]
        if len(base):
            summary.loc[summary["variant"] == v, "delta_vs_mid_mid"] = summary["test_dedup_bal_acc_mean"] - float(base.iloc[0])
    summary.round(4).to_csv(args.out_dir / "summary.csv", index=False)
    for v in summary["variant"].unique():
        if {"mid"} <= set(summary[summary["variant"] == v]["train"]):
            heatmap(summary, v, args.out_dir)
    pd.set_option("display.width", 250)
    for v in summary["variant"].unique():
        print(f"\n{v}: dedup balanced accuracy (rows = train, cols = test), mean over seeds")
        print(summary[summary["variant"] == v].pivot(index="train", columns="test", values="test_dedup_bal_acc_mean")
              .reindex(index=args.train_strategies, columns=TEST_STRATEGIES).round(3).to_string())
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
