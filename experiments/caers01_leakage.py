import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES, ENCODER_TAGS, SPLITS, embedding_path
from dcond.embeddings import load_embeddings
from dcond.plotting import INK_2, SURFACE, apply_style, plt

EMB = PROJECT_ROOT / "data" / "caers" / "embeddings"
MULTI = PROJECT_ROOT / "data" / "caer_multi"
OUT = PROJECT_ROOT / "experiments" / "results" / "caers01_leakage"
SPLIT_COLORS = {"train": "#2a78d6", "validation": "#1baf7a", "test": "#eb6834"}


def nearest(Q: np.ndarray, R: np.ndarray, device: str, chunk: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    Rt = torch.as_tensor(R, dtype=torch.float16, device=device)
    sims, idx = [], []
    for i in range(0, len(Q), chunk):
        q = torch.as_tensor(Q[i : i + chunk], dtype=torch.float16, device=device)
        s, j = (q @ Rt.T).max(dim=1)
        sims.append(s.float().cpu().numpy())
        idx.append(j.cpu().numpy())
    return np.concatenate(sims), np.concatenate(idx)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-dir", type=Path, default=EMB)
    parser.add_argument("--multi-dir", type=Path, default=MULTI)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--dup-thr", type=float, default=0.95)
    parser.add_argument("--match-thr", type=float, default=0.95)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    s = {sp: load_embeddings(embedding_path(sp, "p50", ENCODER_TAGS["clip"], args.emb_dir)) for sp in SPLITS}
    X = {sp: d["image_embeds"].astype(np.float32) for sp, d in s.items()}
    ids = {sp: d["video_relpath"].astype(str) for sp, d in s.items()}
    y = {sp: d["label"].astype(int) for sp, d in s.items()}

    nn_rows, summary = [], {"n": {sp: int(len(ids[sp])) for sp in SPLITS}}
    nn = {}
    for sp in ["validation", "test"]:
        sim, j = nearest(X[sp], X["train"], device)
        nn[sp] = sim
        nn_rows.append(pd.DataFrame({"split": sp, "video_relpath": ids[sp], "nn_train_sim": sim,
                                     "nn_train_id": ids["train"][j], "same_label": y[sp] == y["train"][j]}))
        summary[f"{sp}_share_nn_train_sim_ge"] = {t: float((sim >= t).mean()) for t in [0.9, 0.95, 0.98, 0.99]}
        summary[f"{sp}_1nn_label_acc"] = float((y[sp] == y["train"][j]).mean())
        print(f"1. {sp}: nearest CAER-S train image cosine >= 0.95 for {(sim >= 0.95).mean():.1%}, >= 0.99 for "
              f"{(sim >= 0.99).mean():.1%}; label of that nearest train image is right {summary[f'{sp}_1nn_label_acc']:.1%}")
    pd.concat(nn_rows).round(4).to_csv(args.out_dir / "nn_within.csv", index=False)

    match = None
    if all((args.multi_dir / f"caer_{sp}_k{args.k}.npz").exists() for sp in SPLITS):
        frames, clip_id, clip_split, clip_label = [], [], [], []
        for sp in SPLITS:
            with np.load(args.multi_dir / f"caer_{sp}_k{args.k}.npz", allow_pickle=True) as f:
                n, k = f["clip_full"].shape[:2]
                frames.append(f["clip_full"].reshape(n * k, -1))
                clip_id.append(np.repeat(f["video_relpath"].astype(str), k))
                clip_split.append(np.full(n * k, sp))
                clip_label.append(np.repeat(f["label"], k))
        F = np.concatenate(frames)
        clip_id, clip_split, clip_label = map(np.concatenate, [clip_id, clip_split, clip_label])
        rows = []
        for sp in SPLITS:
            sim, j = nearest(X[sp], F, device)
            rows.append(pd.DataFrame({"split": sp, "video_relpath": ids[sp], "label": np.array(CLASSES)[y[sp]],
                                      "caer_sim": sim, "caer_clip": clip_id[j], "caer_split": clip_split[j],
                                      "caer_label": np.array(CLASSES)[clip_label[j]]}))
        match = pd.concat(rows, ignore_index=True)
        match["matched"] = match["caer_sim"] >= args.match_thr
        match.round(4).to_csv(args.out_dir / "matches.csv", index=False)
        m = match[match["matched"]]
        summary["match_rate"] = match.groupby("split")["matched"].mean().round(4).to_dict()
        summary["matched_label_agreement"] = float((m["label"] == m["caer_label"]).mean())
        summary["caer_split_of_matched"] = {sp: g["caer_split"].value_counts(normalize=True).round(4).to_dict()
                                            for sp, g in m.groupby("split")}
        tr_clips = set(m[m["split"] == "train"]["caer_clip"])
        te_clips = set(m[m["split"] == "test"]["caer_clip"])
        summary["distinct_caer_clips"] = {sp: int(g["caer_clip"].nunique()) for sp, g in m.groupby("split")}
        summary["caer_clips_in_both_caers_train_and_test"] = len(tr_clips & te_clips)
        summary["test_images_whose_clip_is_also_in_train"] = float(m[m["split"] == "test"]["caer_clip"].isin(tr_clips).sum()
                                                                  / max((match["split"] == "test").sum(), 1))
        summary["images_per_caer_clip"] = m.groupby("caer_clip").size().describe().round(2).to_dict()
        print(f"2. match to CAER (cosine >= {args.match_thr}): {summary['match_rate']}; label agrees with the CAER clip "
              f"{summary['matched_label_agreement']:.1%}")
        print(f"   CAER split of matched images: {summary['caer_split_of_matched']}")
        print(f"   distinct CAER clips: {summary['distinct_caer_clips']}; clips in BOTH CAER-S train and test: "
              f"{summary['caer_clips_in_both_caers_train_and_test']}; test images whose clip also gives train images: "
              f"{summary['test_images_whose_clip_is_also_in_train']:.1%}")
    else:
        print(f"2. skipped: no {args.multi_dir}/caer_*_k{args.k}.npz (run scripts/caer_multiframe_embeddings.py)")

    excl = {}
    for sp in ["validation", "test"]:
        bad = set(ids[sp][nn[sp] >= args.dup_thr])
        if match is not None:
            tr_clips = set(match[(match["split"] == "train") & match["matched"]]["caer_clip"])
            g = match[(match["split"] == sp) & match["matched"]]
            bad |= set(g[g["caer_clip"].isin(tr_clips)]["video_relpath"])
        excl[sp] = sorted(bad)
        summary[f"{sp}_excluded"] = len(bad)
        print(f"3. {sp}: {len(bad)}/{len(ids[sp])} excluded, {len(ids[sp]) - len(bad)} left for a leakage-free evaluation")
    (args.out_dir / "dedup_ids.json").write_text(json.dumps({"rule": f"nn_train_sim >= {args.dup_thr} or matched CAER "
                                                                     f"clip (>= {args.match_thr}) also in CAER-S train",
                                                             **excl}, indent=1))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))

    fig, axes = plt.subplots(1, 2 if match is not None else 1, figsize=(10, 3.4), squeeze=False)
    ax = axes[0][0]
    bins = np.linspace(0.5, 1.0, 51)
    for sp in ["validation", "test"]:
        ax.hist(nn[sp], bins=bins, density=True, histtype="step", linewidth=2, color=SPLIT_COLORS[sp], label=sp)
    ax.axvline(args.dup_thr, color=INK_2, linestyle="--", linewidth=1)
    ax.set_xlabel("cosine to the nearest CAER-S train image (CLIP)")
    ax.set_ylabel("density")
    ax.set_title("CAER-S — how close is each image to train?", loc="left")
    ax.legend(frameon=False, fontsize=7)
    if match is not None:
        ax = axes[0][1]
        for sp in SPLITS:
            ax.hist(match[match["split"] == sp]["caer_sim"], bins=bins, density=True, histtype="step", linewidth=2,
                    color=SPLIT_COLORS[sp], label=f"CAER-S {sp}")
        ax.axvline(args.match_thr, color=INK_2, linestyle="--", linewidth=1)
        ax.set_xlabel("cosine to the nearest CAER frame (15 per clip)")
        ax.set_title("CAER-S -> CAER clips", loc="left")
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out_dir / "leakage.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
