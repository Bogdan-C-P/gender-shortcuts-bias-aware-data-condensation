import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES, DEFAULT_FRAME, EMOTION_ENCODERS, ENCODER_TAGS, GENERIC_ENCODERS
from dcond.caer_eval import (EMB_DIRS, config_name, evaluate, load_variant, near_duplicate_ids, select_config,
                             train_classifier)
from dcond.plotting import INK_2, SURFACE, apply_style, plt

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "caer03_mlp_probe"
MODEL_COLORS = {"linear": "#2a78d6", "mlp": "#eb6834"}


def parse_refs(refs: list[str]) -> list[tuple[str, str]]:
    out = []
    for r in refs:
        enc, variant = r.split(":")
        assert enc in ENCODER_TAGS and variant in EMB_DIRS, r
        out.append((enc, variant))
    return out


def reading(d_all: float, d_dedup: float, min_gain: float) -> str:
    if d_all > min_gain and d_dedup > min_gain:
        return "MLP helps on both: information present, not linearly separable"
    if d_all > min_gain:
        return "MLP helps only on test(all): memorization of scenes"
    if d_dedup > min_gain:
        return "MLP helps only on test(dedup)"
    return "no MLP gain: feature limit"


def figure(summary: pd.DataFrame, out: Path) -> None:
    variants = list(summary["variant"].unique())
    subsets = [("all", "test (all)"), ("dedup", "test without near-twins in train")]
    fig, axes = plt.subplots(len(variants), 2, figsize=(11, 3.3 * len(variants)), sharey=True, squeeze=False)
    for r, v in enumerate(variants):
        d = summary[summary["variant"] == v]
        encs = list(dict.fromkeys(d["encoder"]))
        for c, (sub, title) in enumerate(subsets):
            ax = axes[r][c]
            for i, m in enumerate(MODEL_COLORS):
                dm = d[d["model"] == m].set_index("encoder").loc[encs]
                xs = np.arange(len(encs)) + (i - 0.5) * 0.38
                vals, sds = dm[f"test_{sub}_bal_acc_mean"], dm[f"test_{sub}_bal_acc_sd"]
                ax.bar(xs, vals, width=0.36, color=MODEL_COLORS[m], edgecolor=SURFACE, linewidth=1, label=m,
                       yerr=sds, error_kw=dict(ecolor=INK_2, elinewidth=0.8, capsize=2))
                for x, val in zip(xs, vals):
                    ax.text(x, val + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=INK_2)
            ax.axhline(1 / len(CLASSES), color=INK_2, linestyle="--", linewidth=1)
            ax.set_xticks(range(len(encs)), encs)
            ax.set_title(f"{'full frames' if v == 'full' else 'face crops'} — {title}", loc="left")
        axes[r][0].set_ylabel("balanced accuracy")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, ["linear probe", "MLP (2 hidden layers)"], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("CAER — linear probe vs MLP (mean ± sd over seeds; dashed = chance)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=list(EMB_DIRS), choices=list(EMB_DIRS))
    parser.add_argument("--encoders", nargs="+", default=GENERIC_ENCODERS + EMOTION_ENCODERS,
                        choices=list(ENCODER_TAGS))
    parser.add_argument("--models", nargs="+", default=["linear", "mlp"], choices=["linear", "mlp"])
    parser.add_argument("--frame", default=DEFAULT_FRAME)
    parser.add_argument("--dup-ref", nargs="+", default=["clip:full"],
                        help="encoder:variant embeddings that define near-duplicates (union if several)")
    parser.add_argument("--dup-threshold", type=float, default=0.95)
    parser.add_argument("--dedup-ids", type=Path, default=None,
                        help="JSON {validation: [...], test: [...]} of ids to exclude (e.g. caers01_leakage), "
                             "instead of computing near-duplicates with --dup-ref")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--min-gain", type=float, default=0.02, help="balanced-acc gain that counts as 'helps'")
    parser.add_argument("--emb-dir-full", type=Path, default=EMB_DIRS["full"])
    parser.add_argument("--emb-dir-face", type=Path, default=EMB_DIRS["face"])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    apply_style()
    emb_dirs = {"full": args.emb_dir_full, "face": args.emb_dir_face}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.dedup_ids:
        given = json.loads(args.dedup_ids.read_text())
        dup = {s: set(given[s]) for s in ["validation", "test"]}
        print(f"excluded ids from {args.dedup_ids}: validation {len(dup['validation'])}, test {len(dup['test'])}")
        (args.out_dir / "dedup_ids.json").write_text(json.dumps({"source": str(args.dedup_ids), **given}, indent=1))
    else:
        refs = [load_variant(v, e, args.frame, emb_dirs[v]) for e, v in parse_refs(args.dup_ref)]
        dup = near_duplicate_ids(refs, args.dup_threshold)
        n_test = len(refs[0]["test"].ids)
        print(f"near-duplicates (cosine >= {args.dup_threshold} in {args.dup_ref}): "
              f"validation {len(dup['validation'])}/{len(refs[0]['validation'].ids)}, test {len(dup['test'])}/{n_test}")
        (args.out_dir / "dedup_ids.json").write_text(json.dumps(
            {"dup_ref": args.dup_ref, "threshold": args.dup_threshold, **{s: sorted(v) for s, v in dup.items()}}, indent=1))

    grid_rows, run_rows = [], []
    for v in args.variants:
        for e in args.encoders:
            data = load_variant(v, e, args.frame, emb_dirs[v])
            tr, va, te = data["train"], data["validation"], data["test"]
            va_keep = ~np.isin(va.ids, list(dup["validation"]))
            if va_keep.sum() < max(100, 0.05 * len(va_keep)):
                print(f"    WARNING: only {va_keep.sum()} leakage-free validation items — using all {len(va_keep)}")
                va_keep = np.ones(len(va.ids), bool)
            te_keep = ~np.isin(te.ids, list(dup["test"]))
            nn_pred = tr.y[(te.X @ tr.X.T).argmax(axis=1)]
            nn = {"all": evaluate(te.y, nn_pred), "dedup": evaluate(te.y[te_keep], nn_pred[te_keep])}
            print(f"--- {v} / {e}: train {len(tr.y)}, val dedup {va_keep.sum()}/{len(va.y)}, "
                  f"test dedup {te_keep.sum()}/{len(te.y)}; 1-NN bal acc all {nn['all']['bal_acc']:.3f} "
                  f"dedup {nn['dedup']['bal_acc']:.3f}")
            for m in args.models:
                cfg, grid = select_config(m, tr.X, tr.y, va.X[va_keep], va.y[va_keep])
                grid_rows += [{"variant": v, "encoder": e, "model": m, **g} for g in grid]
                for seed in range(args.seeds):
                    clf, val, ep = train_classifier(tr.X, tr.y, va.X[va_keep], va.y[va_keep], seed=seed, **cfg)
                    pred = clf.predict(te.X)
                    res_all, res_dd = evaluate(te.y, pred), evaluate(te.y[te_keep], pred[te_keep])
                    run_rows.append({"variant": v, "encoder": e, "model": m, "config": config_name(cfg), "seed": seed,
                                     "best_epoch": ep, "val_dedup_bal_acc": val,
                                     "test_all_acc": res_all["acc"], "test_all_bal_acc": res_all["bal_acc"],
                                     "test_dedup_acc": res_dd["acc"], "test_dedup_bal_acc": res_dd["bal_acc"],
                                     "n_test_dedup": int(te_keep.sum()),
                                     "nn1_all_bal_acc": nn["all"]["bal_acc"], "nn1_dedup_bal_acc": nn["dedup"]["bal_acc"]})
                    print(f"    {m} [{config_name(cfg)}] seed {seed}: test all {res_all['bal_acc']:.3f} "
                          f"(acc {res_all['acc']:.3f}), dedup {res_dd['bal_acc']:.3f} (acc {res_dd['acc']:.3f})")

    runs = pd.DataFrame(run_rows)
    runs.to_csv(args.out_dir / "runs.csv", index=False)
    pd.DataFrame(grid_rows).to_csv(args.out_dir / "grid.csv", index=False)
    metric_cols = ["val_dedup_bal_acc", "test_all_acc", "test_all_bal_acc", "test_dedup_acc", "test_dedup_bal_acc"]
    agg = runs.groupby(["variant", "encoder", "model", "config"], sort=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s.replace('std', 'sd')}" for c, s in agg.columns]
    summary = agg.reset_index().merge(
        runs.groupby(["variant", "encoder"], sort=False)[["n_test_dedup", "nn1_all_bal_acc", "nn1_dedup_bal_acc"]]
        .first().reset_index(), on=["variant", "encoder"])
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    if set(args.models) == {"linear", "mlp"}:
        figure(summary, args.out_dir / "mlp_vs_linear.png")
        p = summary.pivot_table(index=["variant", "encoder"], columns="model", sort=False,
                                values=["test_all_bal_acc_mean", "test_dedup_bal_acc_mean", "test_all_acc_mean"])
        verdict = pd.DataFrame({
            "linear_all": p[("test_all_bal_acc_mean", "linear")], "mlp_all": p[("test_all_bal_acc_mean", "mlp")],
            "linear_dedup": p[("test_dedup_bal_acc_mean", "linear")], "mlp_dedup": p[("test_dedup_bal_acc_mean", "mlp")],
            "mlp_all_plain_acc": p[("test_all_acc_mean", "mlp")]})
        verdict["delta_all"] = verdict["mlp_all"] - verdict["linear_all"]
        verdict["delta_dedup"] = verdict["mlp_dedup"] - verdict["linear_dedup"]
        verdict["reading"] = [reading(a, b, args.min_gain) for a, b in zip(verdict["delta_all"], verdict["delta_dedup"])]
        verdict.round(3).to_csv(args.out_dir / "verdict.csv")
        pd.set_option("display.width", 250)
        print("\nbalanced accuracy on test, mean over seeds (chance = %.3f):" % (1 / len(CLASSES)))
        print(verdict.round(3).to_string())
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
