import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES, DEFAULT_FRAME, ENCODER_TAGS, embedding_path
from dcond.caer_vlm import EMOTIONS, VLM_DIR, load_descriptions
from dcond.plotting import INK, INK_2, SURFACE, apply_style, plt

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "caer06_named_correlations"
DEDUP_JSON = PROJECT_ROOT / "experiments" / "results" / "caer03_mlp_probe" / "dedup_ids.json"
FER_ASSIGN = PROJECT_ROOT / "experiments" / "results" / "caer01_face_emotion" / "fer_vit_cosine_p50" / "train" / "k1" / "assignments.csv"
NON_EMOTION = ["gender", "age", "setting", "has_text", "people_bin", "issues"]
FEATURE_SETS = {
    "gender": ["gender"], "age": ["age"], "setting": ["setting"], "on-screen text": ["has_text"],
    "number of people": ["people_bin"], "image issues": ["issues"],
    "all non-emotion fields": NON_EMOTION,
    "action (has smiling/crying)": ["action"],
    "non-emotion + action": NON_EMOTION + ["action"],
    "Qwen emotion + expression (reference)": ["emotion", "expression"],
}
C_GRID = [0.01, 0.1, 1.0, 10.0]
VARIANT_COLORS = {"full": "#2a78d6", "face": "#eb6834"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def prepare(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = df.copy()
    tr = df["split"] == "train"
    for col in ["setting", "action", "expression"]:
        keep = df.loc[tr, col].value_counts().head(top_n).index
        df[col] = df[col].where(df[col].isin(keep), "other")
    df["people_bin"] = pd.cut(df["people"], [-2, 0, 1, 2, 3, 1000], labels=["unknown", "1", "2", "3", "4+"]).astype(str)
    df["has_text"] = np.where(df["has_text"], "yes", "no")
    return df


def design(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    parts = []
    for f in fields:
        if f == "issues":
            ex = df["issues"].explode().dropna()
            hot = pd.crosstab(ex.index.to_numpy(), ex.to_numpy())
            parts.append(hot.reindex(df.index, fill_value=0).add_prefix("issue="))
        else:
            parts.append(pd.get_dummies(df[f], prefix=f).astype(int))
    return pd.concat(parts, axis=1).astype(np.float32)


def part_a(dfs: dict, out: Path) -> pd.DataFrame:
    rows, cms = [], {}
    for v, df in dfs.items():
        for s in ["train", "validation", "test"]:
            d = df[df["split"] == s]
            pred = d["emotion"].where(d["emotion"] != "unclear", "neutral")
            rows.append({"variant": v, "split": s, "n": len(d), "plain_acc": accuracy_score(d["label"], d["emotion"]),
                         "balanced_acc": balanced_accuracy_score(d["label"], pred), "unclear": (d["emotion"] == "unclear").mean(),
                         "pct_called_neutral": (d["emotion"] == "neutral").mean()})
        d = df[df["split"] == "train"]
        cm = pd.crosstab(d["label"], d["emotion"], normalize="index").reindex(index=EMOTIONS, columns=EMOTIONS + ["unclear"],
                                                                              fill_value=0)
        cm.round(3).to_csv(out / f"A_qwen_confusion_{v}.csv")
        cms[v] = cm
    summary = pd.DataFrame(rows)
    summary.round(4).to_csv(out / "A_qwen_emotion_summary.csv", index=False)

    fig, axes = plt.subplots(1, len(cms), figsize=(6.2 * len(cms), 4.2), squeeze=False)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("blue", SEQ)
    for ax, (v, cm) in zip(axes[0], cms.items()):
        ax.imshow(cm.values, cmap=cmap, vmin=0, vmax=0.7, aspect="auto")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="white" if val > 0.35 else INK)
        ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=30, ha="right")
        ax.set_yticks(range(cm.shape[0]), cm.index)
        ax.grid(False)
        ax.set_xlabel("Qwen3-VL says")
        ax.set_ylabel("CAER label")
        bal = summary[(summary.variant == v) & (summary.split == "train")]["balanced_acc"].iloc[0]
        ax.set_title(f"{'full frames' if v == 'full' else 'face crops'} — share of each label (balanced acc {bal:.2f})",
                     loc="left")
    fig.tight_layout()
    fig.savefig(out / "qwen_confusion.png", dpi=150)
    plt.close(fig)
    return summary


def part_b(dfs: dict, out: Path) -> None:
    lift_rows, v_rows = [], []
    for v, df in dfs.items():
        d = df[df["split"] == "train"].reset_index(drop=True)
        for field in ["gender", "age", "setting", "has_text", "people_bin", "issues", "action"]:
            col = d.explode("issues")[["label", "issues"]].dropna() if field == "issues" else d[["label", field]]
            tab = pd.crosstab(col[field].to_numpy(), col["label"].to_numpy())
            chi2 = chi2_contingency(tab)[0]
            v_rows.append({"variant": v, "field": field, "cramers_v": np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))})
            n_cls = d["label"].value_counts()
            for val in tab.index[tab.sum(axis=1) >= 30]:
                for c in EMOTIONS:
                    a = int(tab.loc[val, c]) if c in tab.columns else 0
                    b = int(n_cls[c] - a)
                    cc = int(tab.loc[val].sum() - a)
                    dd = int(len(d) - n_cls[c] - cc)
                    if a < 10:
                        continue
                    _, p = fisher_exact([[a, b], [cc, dd]])
                    lift_rows.append({"variant": v, "field": field, "value": val, "class": c, "n_in_class": a,
                                      "pct_in_class": a / n_cls[c], "pct_other_classes": cc / (len(d) - n_cls[c]),
                                      "lift": (a / n_cls[c]) / (tab.loc[val].sum() / len(d)), "p": p})
    lifts = pd.DataFrame(lift_rows)
    lifts["p_bonferroni"] = (lifts["p"] * len(lifts)).clip(upper=1)
    lifts = lifts[lifts["p_bonferroni"] < 0.01].sort_values(["variant", "field", "lift"], ascending=[True, True, False])
    lifts.round(4).to_csv(out / "B_overrepresented_values.csv", index=False)
    pd.DataFrame(v_rows).round(4).to_csv(out / "B_cramers_v.csv", index=False)
    print("\nB  Cramér's V with the label (train):")
    print(pd.DataFrame(v_rows).pivot(index="field", columns="variant", values="cramers_v").round(3).to_string())


def fit_lr(Xtr, ytr, Xva, yva, seed=0):
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=seed).fit(Xtr, ytr)
        s = balanced_accuracy_score(yva, clf.predict(Xva))
        if best is None or s > best[0]:
            best = (s, C, clf)
    return best


def part_c(dfs: dict, dup: dict, n_perm: int, out: Path) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for v, df in dfs.items():
        tr, va, te = (df["split"] == s for s in ["train", "validation", "test"])
        va_keep = va & ~df["video_relpath"].isin(dup["validation"])
        te_dd = te & ~df["video_relpath"].isin(dup["test"])
        y = df["label"].to_numpy()
        for name, fields in FEATURE_SETS.items():
            X = design(df, fields).to_numpy()
            val, C, clf = fit_lr(X[tr], y[tr], X[va_keep], y[va_keep])
            pred = clf.predict(X)
            res = {"variant": v, "feature_set": name, "n_features": X.shape[1], "C": C, "val_dedup_bal_acc": val,
                   "test_all_bal_acc": balanced_accuracy_score(y[te], pred[te]),
                   "test_dedup_bal_acc": balanced_accuracy_score(y[te_dd], pred[te_dd]),
                   "test_dedup_plain_acc": accuracy_score(y[te_dd], pred[te_dd])}
            if n_perm and name in ["gender", "setting", "on-screen text", "all non-emotion fields"]:
                null = []
                for _ in range(n_perm):
                    yp = rng.permutation(y[tr])
                    m = LogisticRegression(C=C, class_weight="balanced", max_iter=3000).fit(X[tr], yp)
                    null.append(balanced_accuracy_score(y[te_dd], m.predict(X[te_dd])))
                null = np.array(null)
                res.update(perm_null_mean=null.mean(), perm_null_p95=np.quantile(null, 0.95),
                           p_perm=(1 + (null >= res["test_dedup_bal_acc"]).sum()) / (n_perm + 1))
            rows.append(res)
            print(f"  C {v:4s} {name:40s} dedup bal acc {res['test_dedup_bal_acc']:.3f}"
                  + (f"  (null 95% {res['perm_null_p95']:.3f}, p {res['p_perm']:.3f})" if "p_perm" in res else ""))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(out / "C_description_only_classifiers.csv", index=False)
    return res


def figure_c(res: pd.DataFrame, probe_ref: dict, out: Path) -> None:
    names = list(FEATURE_SETS)
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(names) + 1.6))
    ys = np.arange(len(names))
    for i, (v, color) in enumerate(VARIANT_COLORS.items()):
        d = res[res["variant"] == v].set_index("feature_set").reindex(names)
        y = ys + (i - 0.5) * 0.38
        ax.barh(y, d["test_dedup_bal_acc"], height=0.36, color=color, edgecolor=SURFACE, linewidth=1,
                label=f"Qwen descriptions of {'full frames' if v == 'full' else 'face crops'}")
        for yy, val in zip(y, d["test_dedup_bal_acc"]):
            ax.text(val + 0.004, yy, f"{val:.2f}", va="center", fontsize=7, color=INK_2)
    ax.axvline(1 / len(CLASSES), color=INK_2, linestyle="--", linewidth=1)
    ax.text(1 / len(CLASSES), len(names) - 0.35, " chance", fontsize=7, color=INK_2)
    for label, val in probe_ref.items():
        ax.axvline(val, color=INK, linestyle=":", linewidth=1)
        ax.text(val, len(names) - 0.35, f" {label} {val:.2f}", fontsize=7, color=INK_2)
    ax.set_xlim(0, max([0.45] + [v + 0.2 for v in probe_ref.values()]))
    ax.set_yticks(ys, names)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("balanced accuracy, test without near-twins (chance 0.14)")
    ax.set_title("CAER — how well do description fields alone predict the label?", loc="left")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "desc_classifier.png", dpi=150)
    plt.close(fig)


def part_d(dfs: dict, out: Path) -> None:
    if not FER_ASSIGN.exists():
        print(f"D skipped: {FER_ASSIGN} missing")
        return
    a = pd.read_csv(FER_ASSIGN)
    f = dfs["face"][dfs["face"]["split"] == "train"].merge(a[["video_relpath", "outlier_top_class", "zs_pred"]],
                                                         on="video_relpath")
    f["qwen_agrees"] = f["emotion"] == f["label"]
    f["fer_agrees"] = f["zs_pred"].str.lower() == f["label"]
    g = f.groupby("outlier_top_class").agg(n=("qwen_agrees", "size"), qwen_agrees_with_label=("qwen_agrees", "mean"),
                                           qwen_unclear=("emotion", lambda s: (s == "unclear").mean()))
    g.index = g.index.map({True: "fer_vit outlier (top 5%)", False: "other faces"})
    g.round(4).to_csv(out / "D_fer_outliers_vs_qwen.csv")
    joint = pd.crosstab(f["fer_agrees"], f["qwen_agrees"], normalize="all")
    joint.round(4).to_csv(out / "D_fer_vs_qwen_agreement.csv")
    both = ((~f["fer_agrees"]) & (~f["qwen_agrees"]) & (f["zs_pred"].str.lower() == f["emotion"])).mean()
    per_cls = f.assign(both_reject_same=(~f["fer_agrees"]) & (~f["qwen_agrees"]) & (f["zs_pred"].str.lower() == f["emotion"])) \
        .groupby("label")["both_reject_same"].mean()
    per_cls.round(4).to_csv(out / "D_both_models_agree_on_other_emotion_per_class.csv")
    hap = f[(f["label"] == "happy") & f["outlier_top_class"]]
    pd.DataFrame({"qwen_emotion": hap["emotion"].value_counts(), "share": hap["emotion"].value_counts(normalize=True)}) \
        .round(3).to_csv(out / "D_qwen_on_fer_happy_outliers.csv")
    print("\nD  fer_vit face outliers vs Qwen:\n", g.round(3).to_string())
    print(f"   faces where fer_vit and Qwen both reject the label and agree with each other: {both:.3f}")
    print("   per class:", per_cls.round(3).to_dict())


def part_e(dfs: dict, dup: dict, seeds: int, out: Path) -> None:
    from dcond.caer_eval import load_variant, select_config, train_classifier

    gender = dfs["full"].set_index("video_relpath")["gender"]
    rows, fp_rows = [], []
    tr_desc = dfs["full"][dfs["full"]["split"] == "train"]
    male_share = tr_desc[tr_desc["gender"] != "unclear"].groupby("label")["gender"].apply(lambda s: (s == "male").mean())
    for variant in ["full", "face"]:
        if not embedding_path("train", DEFAULT_FRAME, ENCODER_TAGS["clip"], {"full": PROJECT_ROOT / "data" / "caer" / "embeddings",
                                                                          "face": PROJECT_ROOT / "data" / "caer_face" / "embeddings"}[variant]).exists():
            print(f"E skipped for {variant}: no CLIP embeddings here (run on the Linux box)")
            continue
        data = load_variant(variant, "clip")
        tr, va, te = data["train"], data["validation"], data["test"]
        va_keep = ~np.isin(va.ids, list(dup["validation"]))
        te_keep = ~np.isin(te.ids, list(dup["test"]))
        cfg, _ = select_config("linear", tr.X, tr.y, va.X[va_keep], va.y[va_keep])
        g = gender.reindex(te.ids).fillna("unclear").to_numpy()
        for seed in range(seeds):
            clf, _, _ = train_classifier(tr.X, tr.y, va.X[va_keep], va.y[va_keep], seed=seed, **cfg)
            pred = clf.predict(te.X)
            for sub, m in [("all", np.ones(len(te.y), bool)), ("dedup", te_keep)]:
                for gg in ["male", "female"]:
                    mm = m & (g == gg)
                    rows.append({"variant": variant, "seed": seed, "test": sub, "gender": gg, "n": int(mm.sum()),
                                 "bal_acc": balanced_accuracy_score(te.y[mm], pred[mm]),
                                 **{f"recall_{c}": float((pred[mm & (te.y == i)] == i).mean()) for i, c in enumerate(EMOTIONS)}})
                for i, c in enumerate(EMOTIONS):
                    fp = {gg: float((pred[m & (g == gg) & (te.y != i)] == i).mean()) for gg in ["male", "female"]}
                    fp_rows.append({"variant": variant, "seed": seed, "test": sub, "class": c, "fp_rate_male": fp["male"],
                                    "fp_rate_female": fp["female"], "train_male_share_in_class": male_share.get(c, np.nan)})
    if not rows:
        return
    r = pd.DataFrame(rows).groupby(["variant", "test", "gender"]).mean(numeric_only=True).drop(columns="seed")
    r.round(4).to_csv(out / "E_probe_by_gender.csv")
    fp = pd.DataFrame(fp_rows).groupby(["variant", "test", "class"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    fp["fp_ratio_male_over_female"] = fp["fp_rate_male"] / fp["fp_rate_female"].clip(lower=1e-6)
    fp.round(4).to_csv(out / "E_false_positives_by_gender.csv", index=False)
    print("\nE  CLIP linear probe by perceived gender (mean over seeds):\n", r[["n", "bal_acc"] + [f"recall_{c}" for c in EMOTIONS]].round(3).to_string())
    for (v, sub), d in fp.groupby(["variant", "test"]):
        rho, p = spearmanr(d["fp_ratio_male_over_female"], d["train_male_share_in_class"])
        print(f"   {v}/{sub}: Spearman(FP male/female ratio, train male share of the class) = {rho:.2f} (p {p:.3f}, 7 classes)")
    for (v, sub), d in fp[fp["test"] == "dedup"].groupby(["variant", "test"]):
        print(f"\n   {v} dedup false-positive rates by gender:\n", d[["class", "fp_rate_male", "fp_rate_female",
                                                                       "train_male_share_in_class"]].round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, v in zip(axes, ["full", "face"]):
        d = fp[(fp["variant"] == v) & (fp["test"] == "dedup")].set_index("class").reindex(EMOTIONS)
        if d["fp_rate_male"].isna().all():
            continue
        x = np.arange(len(EMOTIONS))
        for k, (gg, color) in enumerate([("male", "#2a78d6"), ("female", "#eb6834")]):
            ax.bar(x + (k - 0.5) * 0.38, d[f"fp_rate_{gg}"], width=0.36, color=color, edgecolor=SURFACE, linewidth=1, label=gg)
        ax.set_xticks(x, EMOTIONS, rotation=30, ha="right")
        ax.set_title(f"{'full frames' if v == 'full' else 'face crops'} — how often the probe wrongly says each class",
                     loc="left", fontsize=9)
    axes[0].set_ylabel("false-positive rate (test, dedup)")
    axes[0].legend(frameon=False, title="perceived gender (Qwen)", fontsize=7, title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "probe_by_gender.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm-dir", type=Path, default=VLM_DIR)
    parser.add_argument("--variants", nargs="+", default=["full", "face"], choices=["full", "face"])
    parser.add_argument("--top-n", type=int, default=25, help="most frequent setting/action/expression values kept")
    parser.add_argument("--n-perm", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--dedup-ids", type=Path, default=DEDUP_JSON)
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dup = {k: set(v) for k, v in json.loads(args.dedup_ids.read_text())
           .items() if k in ["validation", "test"]} if args.dedup_ids.exists() else {"validation": set(), "test": set()}
    raw = {v: load_descriptions(v, args.vlm_dir) for v in args.variants}
    dfs = {v: prepare(d, args.top_n) for v, d in raw.items()}
    for v, d in dfs.items():
        print(f"{v}: {len(d)} descriptions, {(~d['parsed_ok']).sum()} unparsed, gender {d['gender'].value_counts().to_dict()}")

    a = part_a(dfs, args.out_dir)
    print("\nA  Qwen emotion vs CAER label:\n", a.round(3).to_string(index=False))
    part_b(dfs, args.out_dir)
    print("\nC  description-only classifiers:")
    res = part_c(dfs, dup, args.n_perm, args.out_dir)
    probe_ref = {}
    s3 = PROJECT_ROOT / "experiments" / "results" / "caer03_mlp_probe" / "summary.csv"
    if s3.exists():
        s = pd.read_csv(s3)
        s = s[(s["encoder"] == "clip") & (s["model"] == "linear") & (s["variant"] == "full")]
        if len(s):
            probe_ref["CLIP linear probe (full)"] = float(s["test_dedup_bal_acc_mean"].iloc[0])
    figure_c(res, probe_ref, args.out_dir)
    if "face" in dfs:
        part_d(dfs, args.out_dir)
    if not args.skip_probe and "full" in dfs:
        part_e(dfs, dup, args.seeds, args.out_dir)
    print(f"\n-> {args.out_dir}")


if __name__ == "__main__":
    main()
