import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES
from dcond.caer_frames import DEDUP_JSON, SPLIT_NAMES, load_dedup
from dcond.plotting import INK_2, SURFACE, apply_style, plt

DATA_DIR = PROJECT_ROOT / "data" / "caer_video"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "caer13_video_e2e"
TV_MODELS = ["r2plus1d_18", "r3d_18", "mc3_18"]
KINETICS_MEAN, KINETICS_STD = (0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
VIDEOMAE_CKPT = "MCG-NJU/videomae-base-finetuned-kinetics"
PUBLISHED = {"CAER-Net (face+context)": 0.7704, "CAER-Net face only": 0.7413, "CLIP-CAER": 0.8177}


def load_split(data_dir: Path, split: str, limit: int) -> pd.DataFrame:
    d = pd.read_csv(data_dir / f"index_{split}.csv", keep_default_na=False)
    d = d[d["n_frames"].astype(int) > 0].copy()
    d["n_frames"] = d["n_frames"].astype(int)
    d["rel"] = [str(Path(p).with_suffix("")) for p in d["video_relpath"]]
    d = d.sort_values("video_relpath").reset_index(drop=True)
    if limit:
        d = d.groupby("label", sort=False).head(max(1, limit // len(CLASSES))).reset_index(drop=True)
    return d


class Windows(torch.utils.data.Dataset):

    def __init__(self, clips: pd.DataFrame, items: list, args, train: bool):
        self.clips, self.items, self.args, self.train = clips, items, args, train
        mean, std = (IMAGENET_MEAN, IMAGENET_STD) if args.model == "videomae" else (KINETICS_MEAN, KINETICS_STD)
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.items)

    def _frames(self, kind: str, rel: str, idx: list[int], boxes, rng) -> torch.Tensor:
        a, size = self.args, self.args.size
        folder = "face" if kind == "face" else "full"
        imgs = []
        for j in idx:
            im = Image.open(a.data_dir / folder / rel / f"{j:03d}.jpg").convert("RGB")
            if kind == "context":
                cx, cy, bw, bh = boxes[j, :4] * [im.width, im.height, im.width, im.height]
                s = max(bw, bh) / 2
                arr = np.asarray(im).copy()
                arr[max(0, int(cy - s)) : int(cy + s), max(0, int(cx - s)) : int(cx + s)] = 0
                im = Image.fromarray(arr)
            short = round(size * 128 / 112)
            sc = short / min(im.size)
            imgs.append(np.asarray(im.resize((max(short, round(im.width * sc)), max(short, round(im.height * sc))), Image.BILINEAR)))
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div_(255)
        H, W = x.shape[-2:]
        if self.train:
            y0, x0 = rng.integers(0, H - size + 1), rng.integers(0, W - size + 1)
        else:
            y0, x0 = (H - size) // 2, (W - size) // 2
        x = x[..., y0 : y0 + size, x0 : x0 + size]
        if self.train and rng.random() < 0.5:
            x = x.flip(-1)
        return ((x - self.mean) / self.std).permute(1, 0, 2, 3).contiguous()

    def __getitem__(self, i):
        c, start = self.items[i]
        row = self.clips.iloc[c]
        n, T = int(row["n_frames"]), self.args.frames
        idx = [min(start + t, n - 1) for t in range(T)]
        rng = np.random.default_rng() if self.train else np.random.default_rng(0)
        boxes = np.load(self.args.data_dir / "boxes" / f"{row['rel']}.npy") if "context" in self.args.stream_list else None
        flip_seed = rng.integers(1 << 30)
        out = {}
        for s in self.args.stream_list:
            out[s] = self._frames(s, row["rel"], idx, boxes, np.random.default_rng(flip_seed))
        return out, int(row["label"]), c


def eval_items(clips: pd.DataFrame, T: int, stride: int) -> list:
    items = []
    for c, n in enumerate(clips["n_frames"]):
        starts = list(range(0, max(1, n - T + 1), stride))
        if n > T and starts[-1] != n - T:
            starts.append(n - T)
        items += [(c, s) for s in starts]
    return items


class VideoMAEBackbone(torch.nn.Module):
    def __init__(self, pretrained: bool, frames: int, size: int):
        super().__init__()
        from transformers import VideoMAEConfig, VideoMAEModel

        if pretrained:
            self.m = VideoMAEModel.from_pretrained(VIDEOMAE_CKPT)
        else:
            self.m = VideoMAEModel(VideoMAEConfig(num_frames=frames, image_size=size, hidden_size=64, num_hidden_layers=2,
                                                  num_attention_heads=2, intermediate_size=128))
        self.m.gradient_checkpointing_enable()
        self.dim = self.m.config.hidden_size
        self.norm = torch.nn.LayerNorm(self.dim)

    def forward(self, x):
        h = self.m(pixel_values=x.transpose(1, 2)).last_hidden_state
        return self.norm(h.mean(1))


def make_backbone(name: str, pretrained: bool, frames: int, size: int):
    if name in TV_MODELS:
        import torchvision.models.video as tvv

        m = getattr(tvv, name)(weights="KINETICS400_V1" if pretrained else None)
        d = m.fc.in_features
        m.fc = torch.nn.Identity()
        return m, d
    b = VideoMAEBackbone(pretrained, frames, size)
    return b, b.dim


class Net(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.streams = args.stream_list
        bbs, dims = zip(*[make_backbone(args.model, args.pretrained, args.frames, args.size) for _ in self.streams])
        self.bb = torch.nn.ModuleList(bbs)
        self.head = torch.nn.Sequential(torch.nn.Dropout(args.dropout), torch.nn.Linear(sum(dims), len(CLASSES)))

    def forward(self, xs: dict):
        return self.head(torch.cat([bb(xs[s]) for bb, s in zip(self.bb, self.streams)], dim=1))


@torch.no_grad()
def predict(net, clips, args, dev) -> np.ndarray:
    net.eval()
    items = eval_items(clips, args.frames, args.window_stride)
    dl = torch.utils.data.DataLoader(Windows(clips, items, args, train=False), batch_size=args.batch_size * 2,
                                     num_workers=args.workers, pin_memory=True)
    probs = np.zeros((len(clips), len(CLASSES)))
    count = np.zeros(len(clips))
    for xs, _, c in dl:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            p = torch.softmax(net({k: v.to(dev, non_blocking=True) for k, v in xs.items()}).float(), 1).cpu().numpy()
        np.add.at(probs, c.numpy(), p)
        np.add.at(count, c.numpy(), 1)
    return probs / np.maximum(count, 1)[:, None]


def scores(y, probs, keep=None) -> dict:
    keep = np.ones(len(y), bool) if keep is None else keep
    pred = probs.argmax(1)
    yk, pk = y[keep], pred[keep]
    rec = [float((pk[yk == c] == c).mean()) if (yk == c).any() else float("nan") for c in range(len(CLASSES))]
    return {"acc": accuracy_score(yk, pk), "bal_acc": balanced_accuracy_score(yk, pk), "n": int(keep.sum()),
            **{f"recall_{c}": r for c, r in zip(CLASSES, rec)}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model", default="r2plus1d_18", choices=TV_MODELS + ["videomae"])
    parser.add_argument("--streams", default="face", choices=["face", "full", "context", "face+context", "face+full"])
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--size", type=int, default=None, help="default 112 (torchvision) / 224 (videomae)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None, help="backbone lr; default 1e-4 (torchvision) / 3e-5 (videomae)")
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--balanced-loss", action="store_true", help="class-weighted cross-entropy (default: plain, as CAER-Net)")
    parser.add_argument("--select", default="acc", choices=["acc", "bal_acc"], help="validation metric for early stopping")
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-clips", type=int, default=0, help="smoke runs: ~this many clips per split")
    parser.add_argument("--dedup-ids", type=Path, default=DEDUP_JSON)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    args.size = args.size or (224 if args.model == "videomae" else 112)
    args.lr = args.lr or (3e-5 if args.model == "videomae" else 1e-4)
    args.stream_list = args.streams.split("+")
    if args.limit_clips:
        args.out_dir = args.out_dir.with_name(args.out_dir.name + "_smoke")
    run = args.run_name or f"{args.model}_{args.streams.replace('+', '-')}{'' if args.pretrained else '_scratch'}" \
                           f"{'_bal' if args.balanced_loss else ''}_s{args.seed}"
    out = args.out_dir / run
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dup = load_dedup(args.dedup_ids)
    clips = {s: load_split(args.data_dir, s, args.limit_clips) for s in SPLIT_NAMES}
    keep = {s: ~clips[s]["video_relpath"].isin(dup[s]).to_numpy() if s in dup else None for s in SPLIT_NAMES}
    y = {s: clips[s]["label"].to_numpy().astype(int) for s in SPLIT_NAMES}
    print(f"=== {run}: clips train {len(y['train'])}, val {len(y['validation'])} (twin-free {keep['validation'].sum()}), "
          f"test {len(y['test'])} (twin-free {keep['test'].sum()}); frames/clip median {clips['train']['n_frames'].median():.0f}",
          flush=True)

    net = Net(args).to(dev)
    head = list(net.head.parameters())
    opt = torch.optim.AdamW([{"params": net.bb.parameters(), "lr": args.lr}, {"params": head, "lr": args.head_lr}],
                            weight_decay=args.wd)
    steps = max(1, len(y["train"]) // args.batch_size)
    total = steps * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / steps) * 0.5 * (1 + np.cos(np.pi * s / total)))
    w = None
    if args.balanced_loss:
        cnt = np.bincount(y["train"], minlength=len(CLASSES)).astype(np.float32)
        w = torch.as_tensor(len(y["train"]) / (len(CLASSES) * np.maximum(cnt, 1)), device=dev)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    rng = np.random.default_rng(args.seed)
    best, best_state, bad, curves = -1.0, None, 0, []
    for epoch in range(args.epochs):
        net.train()
        n = clips["train"]["n_frames"].to_numpy()
        starts = (rng.random(len(n)) * np.maximum(n - args.frames + 1, 1)).astype(int)
        dl = torch.utils.data.DataLoader(Windows(clips["train"], list(zip(range(len(n)), starts)), args, train=True),
                                         batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers,
                                         pin_memory=True, generator=torch.Generator().manual_seed(args.seed * 100 + epoch))
        t0, losses = time.time(), []
        for xs, yy, _ in dl:
            xs = {k: v.to(dev, non_blocking=True) for k, v in xs.items()}
            yy = yy.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits = net(xs)
            loss = loss_fn(logits.float(), yy)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        pv = predict(net, clips["validation"], args, dev)
        va, vd = scores(y["validation"], pv), scores(y["validation"], pv, keep["validation"])
        curves.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": va["acc"], "val_bal_acc": va["bal_acc"],
                       "val_twinfree_acc": vd["acc"], "val_twinfree_bal_acc": vd["bal_acc"], "minutes": (time.time() - t0) / 60})
        print(f"  epoch {epoch}: loss {np.mean(losses):.3f}, val acc {va['acc']:.3f} (bal {va['bal_acc']:.3f}), "
              f"twin-free val acc {vd['acc']:.3f} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        score = va[args.select]
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    pd.DataFrame(curves).to_csv(out / "curves.csv", index=False)
    net.load_state_dict(best_state)
    pt = predict(net, clips["test"], args, dev)
    res = {"all": scores(y["test"], pt), "twin_free": scores(y["test"], pt, keep["test"])}
    np.savez(out / "test_probs.npz", probs=pt, label=y["test"], video_relpath=clips["test"]["video_relpath"].to_numpy(dtype=object),
             twin_free=keep["test"], confusion=confusion_matrix(y["test"], pt.argmax(1), labels=range(len(CLASSES))))
    meta = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out / "metrics.json").write_text(json.dumps({"run": run, "args": meta, "best_val": best, "epochs_run": len(curves), **res}, indent=1))
    print(f"  TEST {run}: full test acc {res['all']['acc']:.4f} (bal {res['all']['bal_acc']:.4f}); "
          f"twin-free acc {res['twin_free']['acc']:.4f} (bal {res['twin_free']['bal_acc']:.4f})", flush=True)
    summarize(args.out_dir)


def summarize(out_dir: Path) -> None:
    rows = []
    for f in sorted(out_dir.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        a = m["args"]
        rows.append({"run": m["run"], "model": a["model"], "streams": a["streams"], "pretrained": a["pretrained"],
                     "balanced_loss": a["balanced_loss"], "seed": a["seed"], "epochs_run": m["epochs_run"], "best_val": m["best_val"],
                     **{f"test_all_{k}": v for k, v in m["all"].items()}, **{f"test_twinfree_{k}": v for k, v in m["twin_free"].items()}})
    if not rows:
        return
    s = pd.DataFrame(rows)
    s.round(4).to_csv(out_dir / "summary.csv", index=False)
    print(s[["run", "test_all_acc", "test_all_bal_acc", "test_twinfree_acc", "test_twinfree_bal_acc"]].round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(s) + 3), 3.8))
    x = np.arange(len(s))
    for i, (col, lab, c) in enumerate([("test_all_acc", "full test, plain acc", "#2a78d6"),
                                       ("test_twinfree_acc", "twin-free test, plain acc", "#eb6834"),
                                       ("test_twinfree_bal_acc", "twin-free test, balanced acc", "#1baf7a")]):
        ax.bar(x + (i - 1) * 0.27, s[col], width=0.25, color=c, edgecolor=SURFACE, linewidth=1, label=lab)
        for xx, v in zip(x + (i - 1) * 0.27, s[col]):
            ax.text(xx, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    for (name, v), ls in zip(PUBLISHED.items(), [":", "--", "-."]):
        ax.axhline(v, color=INK_2, linestyle=ls, linewidth=0.8, label=f"{name}: {v:.3f} (plain, full test)")
    ax.axhline(0.593, color="#898781", linestyle="-", linewidth=0.8, label="our frozen CLIP best: 0.593 (plain, full test)")
    ax.set_xticks(x, s["run"], rotation=15, ha="right", fontsize=7)
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("accuracy")
    ax.set_title("CAER — end-to-end video models (selection on the full validation split)", loc="left", fontsize=9)
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "video_e2e.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
