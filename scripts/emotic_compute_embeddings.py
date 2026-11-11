import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import ENCODER_TAGS
from dcond.embeddings import l2_normalize, save_embeddings
from dcond.emotic import CATEGORY_PROMPTS, DATA_DIR, EMB_DIR, MANIFEST_CSV, SPLITS, VIEWS, embedding_path, load_manifest, multi_hot
from scripts.caer_compute_embeddings import build_encoder


def letterboxed_box(rows) -> list[tuple[float, float, float, float]]:
    out = []
    for r in rows.itertuples():
        W, H = float(r.actual_w), float(r.actual_h)
        side = max(W, H)
        ox, oy = (side - W) // 2, (side - H) // 2
        x1, y1, x2, y2 = max(0.0, float(r.x1)), max(0.0, float(r.y1)), min(W, float(r.x2)), min(H, float(r.y2))
        out.append(((x1 + ox) / side, (y1 + oy) / side, (x2 + ox) / side, (y2 + oy) / side))
    return out


def mean_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    aps = [average_precision_score(labels[:, j], scores[:, j]) for j in range(labels.shape[1]) if labels[:, j].any()]
    return float(np.mean(aps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=list(ENCODER_TAGS))
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--views", nargs="+", default=VIEWS, choices=VIEWS)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--views-dir", type=Path, default=DATA_DIR / "views")
    parser.add_argument("--emb-dir", type=Path, default=EMB_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    encoder, tag, model_id = build_encoder(args.encoder)
    assert tag == ENCODER_TAGS[args.encoder]
    text_embeds = l2_normalize(encoder.encode_texts(CATEGORY_PROMPTS)) if hasattr(encoder, "encode_texts") else None

    for split in args.splits:
        rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if rows.empty:
            print(f"{split}: no ok rows, skipped")
            continue
        labels = multi_hot(rows["categories"])
        vad = rows[["valence", "arousal", "dominance"]].to_numpy(dtype=np.float32)
        for view in args.views:
            out = embedding_path(split, view, tag, args.emb_dir)
            if out.exists() and not args.force:
                print(f"{out.name}: exists, skipped (--force to redo)")
                continue
            col = rows[f"view_{view}"]
            unique = list(dict.fromkeys(col))
            if getattr(encoder, "uses_subject_box", False) and view == "image":
                raw = encoder.encode_images([args.views_dir / p for p in col], batch_size=args.batch_size,
                                            boxes=letterboxed_box(rows))
            else:
                raw_unique = encoder.encode_images([args.views_dir / p for p in unique], batch_size=args.batch_size)
                pos = {p: i for i, p in enumerate(unique)}
                raw = raw_unique[[pos[p] for p in col]]
            image_embeds = l2_normalize(raw)
            arrays = dict(person_id=rows["person_id"].to_numpy(dtype=object), labels=labels, vad=vad,
                          image_embeds_raw=raw, image_embeds=image_embeds,
                          text_prompts=np.array(CATEGORY_PROMPTS, dtype=object), model_id=np.array(model_id, dtype=object))
            if text_embeds is not None:
                arrays["text_embeds"] = text_embeds
            save_embeddings(out, **arrays)
            zs = mean_ap(labels, image_embeds @ text_embeds.T) if text_embeds is not None else float("nan")
            print(f"{out.name}: {len(rows)} persons ({len(unique)} unique files), dim={raw.shape[1]}, "
                  f"zero-shot mAP={zs:.3f} (chance {labels.mean(axis=0)[labels.any(axis=0)].mean():.3f})")


if __name__ == "__main__":
    main()
