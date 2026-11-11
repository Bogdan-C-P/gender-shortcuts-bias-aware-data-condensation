import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torchvision import transforms as T

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES
from dcond.caer_frames import DEDUP_JSON, EVAL, SPLIT_NAMES, CosineHead, FrameSet, load_dedup, uniform_idx
from dcond.plotting import INK_2, SURFACE, apply_style, plt
from experiments.caer11_mlp_features import evaluate_features

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "caer12_finetune"
IMG_DIR = PROJECT_ROOT / "data" / "caer_multi_img"
FEAT_DIR = PROJECT_ROOT / "data" / "caer_ft"
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
MODEL_COLORS = {"pretrained": "#898781", "ce": "#2a78d6", "normface": "#1baf7a", "arcface": "#eb6834"}


def clips_table(img_dir: Path, split: str, variant: str, k: int, limit: int) -> dict:
    idx = pd.read_csv(img_dir / f"index_{split}.csv", keep_default_na=False)
    idx = idx[idx["j"] >= 0]
    col = "full" if variant == "full" else "face"
    ids = np.array(sorted(idx["video_relpath"].unique()))
    if limit:
        lab = idx.drop_duplicates("video_relpath").set_index("video_relpath")["label"].reindex(ids)
        ids = np.array(sorted(pd.Series(ids, index=ids).groupby(lab.values).head(max(1, limit // len(CLASSES))).values))
    pos = {v: i for i, v in enumerate(ids)}
    paths = np.full((len(ids), k), "", dtype=object)
    labels = np.zeros(len(ids), int)
    for r in idx[idx["video_relpath"].isin(pos)].itertuples():
        i = pos[r.video_relpath]
        paths[i, r.j] = getattr(r, col)
        labels[i] = r.label
    valid = paths != ""
    rows = valid.any(1)
    return {"ids": ids[rows], "paths": paths[rows], "valid": valid[rows], "y": labels[rows], "root": img_dir / col}


class Images(torch.utils.data.Dataset):
    def __init__(self, items: list, root: Path, tf):
        self.items, self.root, self.tf = items, root, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, a, b = self.items[i]
        return self.tf(Image.open(self.root / path).convert("RGB")), a, b


class Model(torch.nn.Module):
    def __init__(self, name: str, loss: str, unfreeze: int, s: float, m: float):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection

        self.backbone = CLIPVisionModelWithProjection.from_pretrained(name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        vm = self.backbone.vision_model
        train_mods = list(vm.encoder.layers[len(vm.encoder.layers) - unfreeze:]) + [vm.post_layernorm, self.backbone.visual_projection] \
            if unfreeze > 0 else []
        for mod in train_mods:
            for p in mod.parameters():
                p.requires_grad = True
        d = self.backbone.config.projection_dim
        self.loss = loss
        self.head = torch.nn.Linear(d, len(CLASSES)) if loss == "ce" else \
            CosineHead(d, len(CLASSES), s=s, m=m if loss == "arcface" else 0.0)

    def embed(self, x):
        return self.backbone(pixel_values=x).image_embeds

    def forward(self, x, y=None):
        e = self.embed(x).float()
        return self.head(e) if self.loss == "ce" else self.head(e, y), e

    def trainable_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.named_parameters() if v.requires_grad}


def loader(items, root, tf, bs, workers, shuffle=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(Images(items, root, tf), batch_size=bs, shuffle=shuffle, num_workers=workers,
                                       pin_memory=True, drop_last=shuffle, generator=g, persistent_workers=False)


@torch.no_grad()
def run_frames(model, table, frames: np.ndarray | None, tf, bs, workers, dev, with_logits=True):
    model.eval()
    sel = table["valid"] if frames is None else (table["valid"] & frames)
    ii, jj = np.nonzero(sel)
    items = [(table["paths"][i, j], i, j) for i, j in zip(ii, jj)]
    n, k = table["valid"].shape
    emb, lg = None, np.zeros((n, k, len(CLASSES)), np.float32)
    for x, i, j in loader(items, table["root"], tf, bs, workers):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            logits, e = model(x.to(dev, non_blocking=True))
        e = torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy()
        if emb is None:
            emb = np.zeros((n, k, e.shape[1]), np.float16)
        emb[i.numpy(), j.numpy()] = e
        if with_logits:
            lg[i.numpy(), j.numpy()] = logits.float().cpu().numpy()
    return emb, lg, sel


def head_pred(lg, sel):
    p = np.exp(lg - lg.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return (p * sel[..., None]).sum(1).argmax(1)


def train_one(args, variant, loss, seed, tables, keep_va, dev, tf_train, tf_eval):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Model(args.model_name, loss, args.unfreeze, args.s, args.m).to(dev)
    tr, va = tables["train"], tables["validation"]
    counts = np.bincount(tr["y"], minlength=len(CLASSES)).astype(np.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.as_tensor(len(tr["y"]) / (len(CLASSES) * np.maximum(counts, 1)), device=dev))
    bb = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr}, {"params": model.head.parameters(), "lr": args.head_lr}],
                            weight_decay=args.wd)
    steps_per_epoch = max(1, len(tr["y"]) // args.batch_size)
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch * args.head_warmup_epochs

    def lr_scale(step):
        if step < warm:
            return 0.0
        t = (step - warm) / max(1, total - warm)
        return min(1.0, (step - warm + 1) / steps_per_epoch) * 0.5 * (1 + np.cos(np.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lr_scale, lambda s: 0.5 * (1 + np.cos(np.pi * s / max(1, total)))])
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    va_tab = {k: (v[keep_va] if isinstance(v, np.ndarray) else v) for k, v in va.items()}
    three = np.zeros_like(va_tab["valid"])
    three[:, uniform_idx(3, three.shape[1])] = True
    three |= ~(va_tab["valid"] & three).any(1, keepdims=True) & va_tab["valid"]
    best, best_state, bad, curves = -1.0, None, 0, []
    for epoch in range(args.epochs):
        model.train()
        r = rng.random(tr["valid"].shape)
        r[~tr["valid"]] = -1
        j = r.argmax(1)
        items = [(tr["paths"][i, j[i]], int(tr["y"][i]), 0) for i in range(len(j))]
        t0, losses = time.time(), []
        for x, y, _ in loader(items, tr["root"], tf_train, args.batch_size, args.workers, shuffle=True, seed=seed * 1000 + epoch):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits, _ = model(x, y)
            l = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(l.item())
        _, lg, sel = run_frames(model, va_tab, three, tf_eval, args.batch_size * 2, args.workers, dev)
        score = balanced_accuracy_score(va_tab["y"], head_pred(lg, sel))
        curves.append({"variant": variant, "model": loss, "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses)),
                       "val_dedup_bal_acc": score, "minutes": (time.time() - t0) / 60})
        print(f"    epoch {epoch}: loss {np.mean(losses):.3f}, val dedup bal acc {score:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        if score > best:
            best, bad, best_state = score, 0, model.trainable_state()
        elif epoch >= args.head_warmup_epochs:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state, strict=False)
    return model, best, best_state, curves


def evaluate(name, variant, seed, model, tables, keep, args, dev, tf_eval, with_head, reuse=False):
    out_dir = args.feat_dir / (f"{variant}_{name}" + ("" if name == "pretrained" else f"_s{seed}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fs, logits = {}, {}
    for s in SPLIT_NAMES:
        t = tables[s]
        f = out_dir / f"caer_{s}_k15.npz"
        if f.exists() and (name == "pretrained" or reuse):
            with np.load(f, allow_pickle=True) as z:
                emb, lg, sel = z["emb"], z["logits"], z["valid"]
                assert (z["video_relpath"].astype(str) == t["ids"]).all(), f"{f}: clip order differs from the image index"
        elif model is None:
            raise FileNotFoundError(f)
        else:
            emb, lg, sel = run_frames(model, t, None, tf_eval, args.batch_size * 2, args.workers, dev, with_logits=with_head)
            np.savez(f, emb=emb, logits=lg, valid=sel, label=t["y"], video_relpath=t["ids"].astype(object))
        fs[s] = FrameSet(emb.astype(np.float32), sel, t["y"], t["ids"], np.full(sel.shape + (len(CLASSES),), np.nan), keep[s])
        logits[s] = lg
    rows = []
    for ps in range(args.probe_seeds):
        for r in evaluate_features({s: f.X for s, f in fs.items()}, fs, ps):
            rows.append({"variant": variant, "model": name, "seed": seed, "probe_seed": ps, **r})
    if with_head:
        te = fs["test"]
        mid = np.zeros_like(te.valid)
        mid[np.arange(len(te.y)), (np.abs(np.arange(15) - 7)[None] + np.where(te.valid, 0, 99)).argmin(1)] = True
        for rep, sel in [("mid", mid), ("mean15", te.valid)]:
            pred = head_pred(logits["test"], sel)
            rows.append({"variant": variant, "model": name, "seed": seed, "probe_seed": -1, "rep": rep, "eval": "head",
                         "probe_config": "", "test_all_bal_acc": balanced_accuracy_score(te.y, pred),
                         "test_all_acc": accuracy_score(te.y, pred),
                         "test_dedup_bal_acc": balanced_accuracy_score(te.y[te.keep], pred[te.keep]),
                         "test_dedup_acc": accuracy_score(te.y[te.keep], pred[te.keep])})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", type=Path, default=IMG_DIR)
    parser.add_argument("--feat-dir", type=Path, default=FEAT_DIR)
    parser.add_argument("--variants", nargs="+", default=["full", "face"], choices=["full", "face"])
    parser.add_argument("--losses", nargs="+", default=["ce", "arcface"], choices=["ce", "arcface", "normface"])
    parser.add_argument("--model-name", default="openai/clip-vit-large-patch14")
    parser.add_argument("--unfreeze", type=int, default=4, help="last transformer blocks trained")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--head-warmup-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--s", type=float, default=16.0)
    parser.add_argument("--m", type=float, default=0.2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="seed ids to train (e.g. --seeds 1 2)")
    parser.add_argument("--merge", action="store_true",
                        help="keep the rows of runs.csv / curves.csv from earlier runs that this call does not redo")
    parser.add_argument("--probe-seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-clips", type=int, default=0, help="smoke runs: ~this many clips per split")
    parser.add_argument("--dedup-ids", type=Path, default=DEDUP_JSON)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--eval-runs", nargs="+", default=None,
                        help="no training: re-evaluate saved embeddings in --feat-dir, e.g. full_ce_s0 face_ce_s1 "
                             "(with --dedup-ids experiments/results/caer_alltest_no_dedup.json -> full-test protocol)")
    parser.add_argument("--summarize-only", action="store_true",
                        help="no training: rebuild summary.csv and the figures from runs.csv / curves.csv")
    args = parser.parse_args()
    apply_style()
    if args.summarize_only:
        load_dedup(args.dedup_ids)
        c = args.out_dir / "curves.csv"
        return summarize(pd.read_csv(args.out_dir / "runs.csv"), pd.read_csv(c) if c.exists() else pd.DataFrame(), args.out_dir)
    if args.limit_clips:
        args.feat_dir = args.feat_dir.with_name(args.feat_dir.name + "_smoke")
        args.out_dir = args.out_dir.with_name(args.out_dir.name + "_smoke")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dup = load_dedup(args.dedup_ids)
    norm = T.Normalize(CLIP_MEAN, CLIP_STD)
    tf_eval = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), norm])
    tf_train = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                          T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), norm])

    rows, curves = [], []
    old_runs, old_curves = pd.DataFrame(), pd.DataFrame()
    if args.merge and (args.out_dir / "runs.csv").exists():
        old_runs = pd.read_csv(args.out_dir / "runs.csv")
        old_curves = pd.read_csv(args.out_dir / "curves.csv") if (args.out_dir / "curves.csv").exists() else pd.DataFrame()
        redo = lambda d: (d["variant"].isin(args.variants) & (d["model"].isin(args.losses) & d["seed"].isin(args.seeds)
                                                             | (d["model"] == "pretrained")))
        old_runs = old_runs[~redo(old_runs)]
        old_curves = old_curves[~redo(old_curves)] if len(old_curves) else old_curves
        print(f"--merge: keeping {len(old_runs)} earlier rows ({sorted(set(zip(old_runs['variant'], old_runs['model'], old_runs['seed'])))})")

    def save():
        r = pd.concat([old_runs, pd.DataFrame(rows)], ignore_index=True)
        c = pd.concat([old_curves, pd.DataFrame(curves)], ignore_index=True)
        r.to_csv(args.out_dir / "runs.csv", index=False)
        if len(c):
            c.to_csv(args.out_dir / "curves.csv", index=False)
        return r, c

    for variant in args.variants:
        tables = {s: clips_table(args.img_dir, s, variant, 15, args.limit_clips) for s in SPLIT_NAMES}
        keep = {s: (~np.isin(t["ids"], list(dup[s])) if s in dup else np.ones(len(t["ids"]), bool)) for s, t in tables.items()}
        print(f"\n=== {variant}: clips train {len(tables['train']['y'])}, val dedup {keep['validation'].sum()}, "
              f"test dedup {keep['test'].sum()}/{len(keep['test'])}; frames train {tables['train']['valid'].sum()}", flush=True)
        cached = all((args.feat_dir / f"{variant}_pretrained" / f"caer_{s}_k15.npz").exists() for s in SPLIT_NAMES)
        model = None if cached else Model(args.model_name, "ce", 0, args.s, args.m).to(dev)
        rows += evaluate("pretrained", variant, 0, model, tables, keep, args, dev, tf_eval, with_head=False)
        del model
        show(rows, variant, "pretrained")
        if args.eval_runs:
            for run in args.eval_runs:
                v, loss, sd = run.rsplit("_", 2)
                if v != variant:
                    continue
                print(f"  --- {run}: re-evaluating the saved embeddings (no training)", flush=True)
                rows += evaluate(loss, variant, int(sd[1:]), None, tables, keep, args, dev, tf_eval, with_head=True, reuse=True)
                show(rows, variant, loss)
                save()
            continue
        for loss in args.losses:
            for seed in args.seeds:
                print(f"  --- {variant} / {loss} / seed {seed}", flush=True)
                model, val, state, cur = train_one(args, variant, loss, seed, tables, keep["validation"], dev, tf_train, tf_eval)
                curves += cur
                out = args.feat_dir / f"{variant}_{loss}_s{seed}"
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"state": state, "args": {k: str(v) for k, v in vars(args).items()}, "val_dedup_bal_acc": val},
                           out / "trainable.pt")
                rows += evaluate(loss, variant, seed, model, tables, keep, args, dev, tf_eval, with_head=True)
                del model
                torch.cuda.empty_cache()
                show(rows, variant, loss)
                save()

    summarize(*save(), args.out_dir)


def summarize(runs: pd.DataFrame, curves: pd.DataFrame, out_dir: Path) -> None:
    order = {"pretrained": 0, "ce": 1, "normface": 2, "arcface": 3}
    runs = runs.sort_values(["variant", "model"], key=lambda c: c.map(order) if c.name == "model" else c, kind="stable")
    metrics = [c for c in ["test_all_bal_acc", "test_all_acc", "test_dedup_bal_acc", "test_dedup_acc"] if c in runs]
    agg = runs.groupby(["variant", "model", "eval", "rep"], sort=False)[metrics].agg(["mean", "std"])
    agg.columns = [f"{a}_{b.replace('std', 'sd')}" for a, b in agg.columns]
    summary = agg.reset_index()
    for v in summary["variant"].unique():
        for (ev, rep), _ in summary[summary["variant"] == v].groupby(["eval", "rep"]):
            b = summary[(summary["variant"] == v) & (summary["model"] == "pretrained") & (summary["eval"] == ev) & (summary["rep"] == rep)]
            if len(b):
                sel = (summary["variant"] == v) & (summary["eval"] == ev) & (summary["rep"] == rep)
                summary.loc[sel, "delta_vs_pretrained"] = summary.loc[sel, "test_dedup_bal_acc_mean"] - float(b["test_dedup_bal_acc_mean"].iloc[0])
    summary.round(4).to_csv(out_dir / "summary.csv", index=False)
    figure(summary, out_dir / "finetune.png")
    if len(curves):
        curves_figure(curves, out_dir / "curves.png")
    pd.set_option("display.width", 250)
    print(summary.pivot_table(index=["variant", "model"], columns=["eval", "rep"], values="test_dedup_bal_acc_mean", sort=False).round(3).to_string())
    print(f"\n-> {out_dir}")


def show(rows, variant, name):
    d = pd.DataFrame([r for r in rows if r["variant"] == variant and r["model"] == name])
    m = d.groupby(["eval", "rep"])[["test_all_bal_acc", "test_dedup_bal_acc"]].mean()
    print(f"  {name}: " + "; ".join(f"{e}/{r} all {a:.3f} dedup {b:.3f}" for (e, r), (a, b) in m.iterrows()), flush=True)


def figure(summary: pd.DataFrame, out: Path) -> None:
    variants = list(summary["variant"].unique())
    cats = [("linear_probe", "mid"), ("linear_probe", "mean15"), ("ncm", "mean15"), ("nn1", "mean15"), ("head", "mean15")]
    labels = ["probe, middle frame", "probe, mean 15", "nearest class mean", "1-NN", "own head, mean 15"]
    names = {"full": "full frames", "face": "face crops"}
    fig, axes = plt.subplots(1, len(variants), figsize=(6.4 * len(variants), 3.6), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], variants):
        d = summary[summary["variant"] == v]
        models = list(dict.fromkeys(d["model"]))
        w = 0.8 / len(models)
        for i, m in enumerate(models):
            dm = d[d["model"] == m].set_index(["eval", "rep"]).reindex(cats)
            xs = np.arange(len(cats)) + (i - (len(models) - 1) / 2) * w
            ax.bar(xs, dm["test_dedup_bal_acc_mean"], width=w * 0.95, color=MODEL_COLORS.get(m, INK_2), edgecolor=SURFACE,
                   yerr=dm["test_dedup_bal_acc_sd"].fillna(0), error_kw=dict(ecolor=INK_2, elinewidth=0.7, capsize=1.5), label=m)
            for x, val in zip(xs, dm["test_dedup_bal_acc_mean"]):
                if np.isfinite(val):
                    ax.text(x, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=6, color=INK_2)
        ax.set_xticks(range(len(cats)), labels, fontsize=7)
        ax.set_ylim(0.1, None)
        ax.set_title(names[v], loc="left")
    axes[0][0].set_ylabel(f"balanced acc ({EVAL['short']})")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("CAER — fine-tuned CLIP ViT-L/14 (last blocks): pretrained vs softmax vs ArcFace", x=0.01, ha="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def curves_figure(c: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for (v, m, s), d in c.groupby(["variant", "model", "seed"]):
        ax.plot(d["epoch"], d["val_dedup_bal_acc"], marker="o", markersize=3, color=MODEL_COLORS.get(m, INK_2),
                linestyle="-" if v == "full" else "--", label=f"{v} / {m} / seed {s}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"val balanced acc ({EVAL['short']})")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
