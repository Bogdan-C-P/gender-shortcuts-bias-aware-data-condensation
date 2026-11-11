import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES
from dcond.caer_eval import config_name, select_config, train_classifier
from dcond.caer_frames import DEDUP_JSON, MULTI_DIR, SPLIT_NAMES, frame_sets, load_dedup, load_multi, mean_rep
from dcond.plotting import SURFACE, apply_style, plt

ASR_DIR = PROJECT_ROOT / "data" / "caer_asr"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "caer15_transcripts"
ENCODERS = {"mpnet": "sentence-transformers/all-mpnet-base-v2", "emoroberta": "j-hartmann/emotion-english-distilroberta-base"}
EMO_MAP = {"anger": "Anger", "disgust": "Disgust", "fear": "Fear", "joy": "Happy", "neutral": "Neutral", "sadness": "Sad",
           "surprise": "Surprise"}
TFIDF_C = [0.1, 0.3, 1.0, 3.0, 10.0]
SUBSETS = ["all", "twin_free", "has_text", "has_text_twin_free"]


def load_transcripts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    parts = df["video_relpath"].str.split("/", expand=True)
    df["split"], df["class_name"] = parts[0], parts[1]
    df["label"] = df["class_name"].map({c: i for i, c in enumerate(CLASSES)})
    if df["label"].isna().any():
        raise ValueError(f"unknown class folders: {sorted(set(df.loc[df['label'].isna(), 'class_name']))}")
    df["label"] = df["label"].astype(int)
    df["usable"] = df["usable"].astype(str).str.lower().eq("true")
    df["text"] = df["text"].astype(str)
    return df.set_index("video_relpath")


@torch.no_grad()
def encode(texts: list[str], enc: str, device: str, batch: int = 64) -> tuple[np.ndarray, np.ndarray | None]:
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    ckpt = ENCODERS[enc]
    tok = AutoTokenizer.from_pretrained(ckpt)
    cls = enc == "emoroberta"
    model = (AutoModelForSequenceClassification if cls else AutoModel).from_pretrained(ckpt).to(device).eval()
    order = None
    if cls:
        id2label = {i: EMO_MAP[l.lower()] for i, l in model.config.id2label.items()}
        order = [next(i for i, c in id2label.items() if c == name) for name in CLASSES]
    embs, zs = [], []
    for i in range(0, len(texts), batch):
        b = tok(texts[i : i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        out = model(**b, output_hidden_states=cls)
        h = out.hidden_states[-1] if cls else out.last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        e = (h * m).sum(1) / m.sum(1).clamp_min(1)
        embs.append(torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy())
        if cls:
            zs.append(torch.softmax(out.logits.float(), -1)[:, order].cpu().numpy())
    return np.concatenate(embs), (np.concatenate(zs) if cls else None)


def text_features(df: pd.DataFrame, enc: str, device: str, force: bool, cache_dir: Path) -> tuple[np.ndarray, np.ndarray | None]:
    cache = cache_dir / f"text_emb_{enc}.npz"
    if cache.exists() and not force:
        with np.load(cache, allow_pickle=True) as z:
            if list(z["video_relpath"].astype(str)) == list(df.index) and (z["text"].astype(str) == df["text"].to_numpy()).all():
                return z["emb"], (z["zs"] if z["zs"].size else None)
    print(f"  encoding {len(df)} transcripts with {ENCODERS[enc]} ...", flush=True)
    emb, zs = encode(df["text"].tolist(), enc, device)
    np.savez(cache, video_relpath=df.index.to_numpy(dtype=object), text=df["text"].to_numpy(dtype=object), emb=emb,
             zs=zs if zs is not None else np.zeros(0))
    return emb, zs


def scores(y, pred, keep) -> dict:
    yk, pk = y[keep], pred[keep]
    rec = {f"recall_{c}": float((pk[yk == i] == i).mean()) if (yk == i).any() else np.nan for i, c in enumerate(CLASSES)}
    return {"n": int(keep.sum()), "acc": accuracy_score(yk, pk), "bal_acc": balanced_accuracy_score(yk, pk), **rec}


def norm_text(t: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", "", t.lower()).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-csv", type=Path, default=ASR_DIR / "transcripts.csv")
    parser.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    parser.add_argument("--no-fusion", dest="fusion", action="store_false", help="skip the image reference and the fusion")
    parser.add_argument("--multi-dir", type=Path, default=MULTI_DIR)
    parser.add_argument("--dedup-ids", type=Path, default=DEDUP_JSON)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--force-encode", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dup = load_dedup(args.dedup_ids)
    df = load_transcripts(args.asr_csv)

    cov = df.assign(n_words_usable=df["n_words"].where(df["usable"])).groupby(["split", "class_name"]).agg(
        n=("label", "size"), usable_pct=("usable", "mean"), median_words_usable=("n_words_usable", "median"))
    cov["usable_pct"] = (100 * cov["usable_pct"]).round(1)
    cov.to_csv(args.out_dir / "coverage.csv")
    print("usable transcripts, % per class (train / validation / test):")
    print(cov["usable_pct"].unstack(0)[SPLIT_NAMES].to_string())
    tr_txt = df[(df["split"] == "train") & df["usable"]].assign(t=lambda d: d["text"].map(norm_text))
    tr_map = tr_txt.groupby("t")["label"].agg(lambda s: s.mode().iloc[0])
    te_u = df[(df["split"] == "test") & df["usable"]]
    tt = te_u["text"].map(norm_text)
    hit = tt.isin(tr_map.index)
    agree = (tr_map.reindex(tt[hit]).to_numpy() == te_u["label"][hit].to_numpy()).mean() if hit.any() else np.nan
    print(f"test clips with a usable transcript identical to a train transcript: {hit.mean():.1%} "
          f"({int(hit.sum())}/{len(te_u)}), same label in {agree:.1%} of those", flush=True)

    if args.fusion:
        data = {s: load_multi(s, 15, args.multi_dir) for s in SPLIT_NAMES}
        fs = frame_sets(data, "both", dup)
        ids = {s: fs[s].ids for s in SPLIT_NAMES}
        X_img = {s: mean_rep(fs[s].X, fs[s].valid) for s in SPLIT_NAMES}
        missing = sum((~pd.Index(ids[s]).isin(df.index)).sum() for s in SPLIT_NAMES)
        if missing:
            print(f"WARNING: {missing} clips of data/caer_multi have no transcript row (treated as no text)")
    else:
        ids = {s: df.index[df["split"] == s].to_numpy() for s in SPLIT_NAMES}
    sub = {s: df.reindex(ids[s]) for s in SPLIT_NAMES}
    y = {s: (sub[s]["label"].fillna(-1).astype(int).to_numpy() if not args.fusion else fs[s].y) for s in SPLIT_NAMES}
    has = {s: sub[s]["usable"].fillna(False).astype(bool).to_numpy() for s in SPLIT_NAMES}
    keep = {s: ~np.isin(ids[s], list(dup[s])) if s in dup else np.ones(len(ids[s]), bool) for s in SPLIT_NAMES}
    texts = {s: sub[s]["text"].fillna("").to_numpy() for s in SPLIT_NAMES}
    prior = np.bincount(y["train"], minlength=len(CLASSES)) / len(y["train"])
    vsel = keep["validation"] & has["validation"]
    masks = {"all": np.ones(len(y["test"]), bool), "twin_free": keep["test"], "has_text": has["test"],
             "has_text_twin_free": has["test"] & keep["test"]}
    print(f"clips: train {len(y['train'])} ({has['train'].mean():.1%} with text), validation {len(y['validation'])}, "
          f"test {len(y['test'])} ({has['test'].mean():.1%} with text, {masks['has_text_twin_free'].sum()} twin-free)", flush=True)

    rows, probs0, val_score = [], {}, {}

    def record(route: str, seed: int, p_test: np.ndarray, val: float) -> None:
        pred = p_test.argmax(1)
        for name, m in masks.items():
            rows.append({"route": route, "seed": seed, "test": name, "val_bal_acc": val, **scores(y["test"], pred, m)})
        if seed == 0:
            probs0[route] = p_test
        val_score.setdefault(route, []).append(val)
        r = scores(y["test"], pred, masks["twin_free"])
        print(f"    {route:24s} s{seed}: val {val:.3f} | twin-free test bal acc {r['bal_acc']:.3f} (acc {r['acc']:.3f}); "
              f"recall A/D/F {r['recall_Anger']:.2f}/{r['recall_Disgust']:.2f}/{r['recall_Fear']:.2f}", flush=True)

    def with_prior(p_text: np.ndarray, split: str) -> np.ndarray:
        p = np.tile(prior, (len(y[split]), 1))
        p[has[split]] = p_text
        return p

    print("\n== text: tfidf_lr", flush=True)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
    T = {s: vec.fit_transform(texts[s][has[s]]) if s == "train" else vec.transform(texts[s][has[s]]) for s in SPLIT_NAMES}
    best = None
    for C in TFIDF_C:
        lr = LogisticRegression(C=C, class_weight="balanced", max_iter=5000).fit(T["train"], y["train"][has["train"]])
        v = balanced_accuracy_score(y["validation"][vsel], lr.predict(vec.transform(texts["validation"][vsel])))
        if best is None or v > best[0]:
            best = (v, C, lr)
    v, C, lr = best
    print(f"  C={C}", flush=True)
    p = np.zeros((T["test"].shape[0], len(CLASSES)))
    p[:, lr.classes_] = lr.predict_proba(T["test"])
    record("tfidf_lr", 0, with_prior(p, "test"), v)
    vocab = np.array(vec.get_feature_names_out())
    top = [{"class": CLASSES[c], "rank": k + 1, "ngram": vocab[j], "coef": float(lr.coef_[i, j])}
           for i, c in enumerate(lr.classes_) for k, j in enumerate(np.argsort(-lr.coef_[i])[:20])]
    pd.DataFrame(top).to_csv(args.out_dir / "top_ngrams.csv", index=False)
    print("  top n-grams: " + "; ".join(f"{CLASSES[c]}: {', '.join(vocab[np.argsort(-lr.coef_[i])[:8]])}"
                                        for i, c in enumerate(lr.classes_)), flush=True)

    T_emb = {}
    for enc in args.encoders:
        emb, zs = text_features(df, enc, device, args.force_encode, args.asr_csv.parent)
        row = {v: i for i, v in enumerate(df.index)}
        idx = {s: np.array([row.get(i, -1) for i in ids[s]]) for s in SPLIT_NAMES}
        T_emb[enc] = {s: np.where((idx[s] >= 0)[:, None], emb[np.maximum(idx[s], 0)], 0).astype(np.float32) for s in SPLIT_NAMES}
        if zs is not None:
            print(f"\n== text: zeroshot_{enc}", flush=True)
            z = {s: zs[np.maximum(idx[s], 0)] for s in SPLIT_NAMES}
            v = balanced_accuracy_score(y["validation"][vsel], z["validation"][vsel].argmax(1))
            record(f"zeroshot_{enc}", 0, with_prior(z["test"][has["test"]], "test"), v)
        for model in ["linear", "mlp"]:
            route = f"{enc}_{model}"
            print(f"\n== text: {route}", flush=True)
            Xt = {s: T_emb[enc][s][has[s]] for s in SPLIT_NAMES}
            ytr, Xva, yva = y["train"][has["train"]], T_emb[enc]["validation"][vsel], y["validation"][vsel]
            cfg, _ = select_config(model, Xt["train"], ytr, Xva, yva)
            print(f"  [{config_name(cfg)}]", flush=True)
            for seed in range(args.seeds):
                clf, v, _ = train_classifier(Xt["train"], ytr, Xva, yva, seed=seed, **cfg)
                record(route, seed, with_prior(clf.predict_proba(Xt["test"]), "test"), v)

    if args.fusion:
        kv = keep["validation"]
        feats = {"image_both_mlp": X_img}
        for enc in args.encoders:
            feats[f"fusion_{enc}_mlp"] = {s: np.concatenate([X_img[s], T_emb[enc][s] * has[s][:, None],
                                                             has[s][:, None].astype(np.float32)], 1) for s in SPLIT_NAMES}
        for route, X in feats.items():
            print(f"\n== {route}", flush=True)
            cfg, _ = select_config("mlp", X["train"], y["train"], X["validation"][kv], y["validation"][kv])
            print(f"  [{config_name(cfg)}]", flush=True)
            for seed in range(args.seeds):
                clf, v, _ = train_classifier(X["train"], y["train"], X["validation"][kv], y["validation"][kv], seed=seed, **cfg)
                record(route, seed, clf.predict_proba(X["test"]), v)

    runs = pd.DataFrame(rows)
    runs.to_csv(args.out_dir / "runs.csv", index=False)
    num = [c for c in runs.columns if c not in ("route", "seed", "test")]
    s = runs.groupby(["route", "test"], sort=False)[num].mean()
    s = s.join(runs.groupby(["route", "test"], sort=False)[["bal_acc", "acc"]].std().add_suffix("_sd")).reset_index()
    s.round(4).to_csv(args.out_dir / "summary.csv", index=False)
    for route, p in probs0.items():
        if route == "image_both_mlp":
            continue
        d = args.out_dir / route
        d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "test_probs.npz", probs=p, label=y["test"], video_relpath=ids["test"].astype(object),
                 twin_free=keep["test"], has_text=has["test"])

    text_routes = [r for r in val_score if not r.startswith(("image", "fusion"))]
    best_text = max(text_routes, key=lambda r: np.mean(val_score[r]))
    fusion_routes = [r for r in val_score if r.startswith("fusion")]
    shown = (["image_both_mlp"] if "image_both_mlp" in val_score else []) + [best_text] + \
            ([max(fusion_routes, key=lambda r: np.mean(val_score[r]))] if fusion_routes else [])
    figure(s, shown, args.out_dir / "per_class_recall.png")
    pd.set_option("display.width", 250)
    cols = ["n", "acc", "bal_acc", "bal_acc_sd"] + [f"recall_{c}" for c in CLASSES]
    for t in ["twin_free", "has_text_twin_free", "all"]:
        print(f"\n--- test subset: {t} (mean over seeds) ---")
        print(s[s["test"] == t].set_index("route")[cols].round(3).to_string())
    print(f"\nbest text route on validation: {best_text}\n-> {args.out_dir}")


def figure(s: pd.DataFrame, routes: list[str], out: Path) -> None:
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), squeeze=False)
    for ax, t in zip(axes[0], ["twin_free", "has_text_twin_free"]):
        x = np.arange(len(CLASSES) + 1)
        w = 0.8 / len(routes)
        for i, r in enumerate(routes):
            d = s[(s["route"] == r) & (s["test"] == t)].iloc[0]
            vals = [d[f"recall_{c}"] for c in CLASSES] + [d["bal_acc"]]
            ax.bar(x + (i - (len(routes) - 1) / 2) * w, vals, width=w * 0.95, color=colors[i % 3], edgecolor=SURFACE,
                   linewidth=0.8, label=r)
        ax.set_xticks(x, CLASSES + ["balanced\nacc"], fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_ylabel("recall")
        ax.set_title({"twin_free": "twin-free test, every clip (no transcript -> train prior)",
                      "has_text_twin_free": "twin-free test, clips with a usable transcript"}[t], loc="left", fontsize=9)
    axes[0][0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
