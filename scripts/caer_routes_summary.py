import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.plotting import BLUE, INK_2, MUTED, SURFACE, apply_style, plt

RES = PROJECT_ROOT / "experiments" / "results"


SUFFIX, METRIC = "", "test_dedup"


def pick(csv: str, variant: str, **eq) -> tuple[float, float, float]:
    folder, name = csv.split("/")
    path = RES / (folder.format(v=variant) + SUFFIX) / name
    if not path.exists():
        return np.nan, np.nan, np.nan
    d = pd.read_csv(path)
    d = d[d["variant"] == variant] if "variant" in d else d
    for k, v in eq.items():
        d = d[d[k] == v]
    if len(d) != 1:
        return np.nan, np.nan, np.nan
    r = d.iloc[0]
    return (float(r[f"{METRIC}_bal_acc_mean"]), float(r.get(f"{METRIC}_bal_acc_sd", np.nan)),
            float(r.get(f"{METRIC}_acc_mean", np.nan)))


ROUTES = [
    ("middle frame + linear (start)", "caer08_frame_budget/summary.csv", dict(selection="uniform", k=1, regime="same")),
    ("mean of 15 frames + linear", "caer09_frame_aggregation/summary.csv", dict(model="mean_linear", test_k=15)),
    ("mean of 15 frames + MLP", "caer09_frame_aggregation/summary.csv", dict(model="mean_mlp", test_k=15)),
    ("attention over frames + MLP", "caer09_frame_aggregation/summary.csv", dict(model="attn_mlp", test_k=15)),
    ("MLP features (1 layer, all frames) + linear", "caer11_mlp_features/summary.csv",
     dict(train_frames="all", loss="ce", depth=1, layer="h1", eval="linear_probe", rep="mean15")),
    ("fine-tuned ViT-L/14 (softmax) + linear", "caer12_finetune/summary.csv", dict(model="ce", eval="linear_probe", rep="mean15")),
    ("fine-tuned ViT-L/14 (ArcFace) + linear", "caer12_finetune/summary.csv", dict(model="arcface", eval="linear_probe", rep="mean15")),
    ("fine-tuned ViT-L/14 (softmax) + MLP on mean", "caer09_ft_{v}_ce/summary.csv", dict(model="mean_mlp", test_k=15)),
]


def main():
    global SUFFIX, METRIC, RES
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RES)
    parser.add_argument("--suffix", default="", help="'_alltest': read the *_alltest result folders (full validation + full test)")
    args = parser.parse_args()
    SUFFIX, RES = args.suffix, args.results
    METRIC = "test_all" if SUFFIX else "test_dedup"
    test_label = "full test, all 2,637 clips" if SUFFIX else "test without near-twins"
    out = RES / ("caer_routes" + SUFFIX)
    apply_style()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in ["full", "face"]:
        for label, csv, eq in ROUTES:
            m, sd, acc = pick(csv, variant, **eq)
            rows.append({"variant": variant, "route": label, f"{METRIC}_bal_acc_mean": m, f"{METRIC}_bal_acc_sd": sd,
                         f"{METRIC}_acc_mean": acc})
    both = pick("caer09_frame_aggregation/summary.csv", "both", model="mean_mlp", test_k=15)
    rows.append({"variant": "both", "route": "mean of 15 frames + MLP", f"{METRIC}_bal_acc_mean": both[0],
                 f"{METRIC}_bal_acc_sd": both[1], f"{METRIC}_acc_mean": both[2]})
    df = pd.DataFrame(rows)
    df.round(4).to_csv(out / "routes.csv", index=False)
    print(df.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    labels = [r[0] for r in ROUTES]
    vals = df[f"{METRIC}_bal_acc_mean"] + df[f"{METRIC}_bal_acc_sd"].fillna(0)
    lo = min(0.30, np.floor((df[f"{METRIC}_bal_acc_mean"].min() - 0.05) * 20) / 20)
    hi = max(0.56, np.ceil((vals.max() + 0.04) * 20) / 20)
    y = np.arange(len(labels))[::-1]
    for ax, v, title in zip(axes, ["full", "face"], ["full frames", "face crops"]):
        d = df[df["variant"] == v].set_index("route").reindex(labels)
        colors = [MUTED] + [BLUE] * (len(labels) - 1)
        m, sd = d[f"{METRIC}_bal_acc_mean"].to_numpy(), d[f"{METRIC}_bal_acc_sd"].fillna(0).to_numpy()
        ax.hlines(y, lo, np.nan_to_num(m, nan=lo), color="#e1e0d9", linewidth=1)
        ax.errorbar(m, y, xerr=sd, fmt="none", ecolor=INK_2, elinewidth=0.9, capsize=2.5, zorder=2)
        ax.scatter(m, y, s=64, c=colors, edgecolors=SURFACE, linewidths=2, zorder=3)
        for yy, val, e in zip(y, m, sd):
            ax.text(val + e + 0.006 if np.isfinite(val) else lo + 0.002, yy, f"{val:.3f}" if np.isfinite(val) else "not run",
                    va="center", fontsize=7.5, color=INK_2)
        ax.axvline(1 / 7, color=INK_2, linestyle=":", linewidth=0.8)
        ax.set_xlim(lo, hi)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"balanced accuracy, {test_label} (chance 0.14)")
    axes[0].set_yticks(y, labels, fontsize=8)
    note = f"full frame + face together, mean of 15 frames + MLP: {both[0]:.3f}" if np.isfinite(both[0]) else ""
    fig.suptitle(f"CAER — every route to better features ends at about the same level ({test_label})", x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.01, note, fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "routes.png", dpi=150)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
