import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import (
    CLASS_PROMPTS, CLASSES, EMB_DIR, ENCODER_TAGS, FRAME_POSITIONS, FRAMES_DIR, MANIFEST_CSV, SPLITS, embedding_path,
    load_manifest,
)
from dcond.embeddings import l2_normalize, save_embeddings

CLIP_MODEL = "openai/clip-vit-large-patch14"


def build_encoder(name: str):
    if name == "clip":
        from dcond.embeddings import ClipEncoder

        return ClipEncoder(CLIP_MODEL), CLIP_MODEL.split("/")[-1], CLIP_MODEL
    if name == "siglip":
        from dcond.embeddings_siglip import DEFAULT_MODEL, SiglipEncoder

        return SiglipEncoder(DEFAULT_MODEL), DEFAULT_MODEL.split("/")[-1], DEFAULT_MODEL
    if name == "evaclip":
        from dcond.embeddings_evaclip import DEFAULT_MODEL_NAME, DEFAULT_PRETRAINED, EvaClipEncoder

        tag = f"{DEFAULT_MODEL_NAME}_{DEFAULT_PRETRAINED}"
        return EvaClipEncoder(DEFAULT_MODEL_NAME, DEFAULT_PRETRAINED), tag, f"open_clip:{tag}"
    if name == "fer_vit":
        from dcond.embeddings_emotion import FER_VIT_MODEL, FerVitEncoder

        return FerVitEncoder(), FER_VIT_MODEL.split("/")[-1], FER_VIT_MODEL
    if name == "clip_fer":
        from dcond.embeddings_emotion import CLIP_BASE, CLIP_FER_VISION, ClipFerEncoder

        return ClipFerEncoder(), CLIP_FER_VISION.split("/")[-1], f"{CLIP_FER_VISION} (vision) + {CLIP_BASE} (projection, text)"
    if name == "emotionclip":
        from dcond.embeddings_emotion import EmotionClipEncoder

        enc = EmotionClipEncoder()
        return enc, "EmotionCLIP_ViT-B-32", "EmotionCLIP (Zhang et al. CVPR 2023), authors' checkpoint"
    raise ValueError(name)


def encode(encoder, paths, batch_size, classes):
    if hasattr(encoder, "encode_images_with_logits"):
        raw, logits = encoder.encode_images_with_logits(paths, batch_size=batch_size)
        return raw, encoder.logits_for(logits, classes)
    return encoder.encode_images(paths, batch_size=batch_size), None


def zero_shot_accuracy(image_embeds: np.ndarray, text_embeds: np.ndarray, labels: np.ndarray) -> float:
    return float(((image_embeds @ text_embeds.T).argmax(axis=1) == labels).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, choices=list(ENCODER_TAGS))
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--frames", nargs="+", default=list(FRAME_POSITIONS), choices=list(FRAME_POSITIONS))
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--emb-dir", type=Path, default=EMB_DIR)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    encoder, tag, model_id = build_encoder(args.encoder)
    assert tag == ENCODER_TAGS[args.encoder], f"tag {tag} != dcond.caer.ENCODER_TAGS[{args.encoder}]"
    text_embeds = l2_normalize(encoder.encode_texts(CLASS_PROMPTS)) if hasattr(encoder, "encode_texts") else None

    for split in args.splits:
        rows = manifest[manifest["split"] == split].reset_index(drop=True)
        if rows.empty:
            print(f"{split}: no ok rows in manifest, skipped")
            continue
        assert (rows["class_name"] == [CLASSES[i] for i in rows["label"]]).all(), "label/class_name mismatch"
        for frame in args.frames:
            out = embedding_path(split, frame, tag, args.emb_dir)
            if out.exists() and not args.force:
                print(f"{out.name}: exists, skipped (--force to redo)")
                continue
            col = rows[f"frame_{frame}"]
            rows_f = rows[col.notna() & (col.astype(str) != "")].reset_index(drop=True)
            if len(rows_f) < len(rows):
                print(f"{out.name}: {len(rows) - len(rows_f)}/{len(rows)} clips have no {frame} frame, skipped")
            paths = [args.frames_dir / p for p in rows_f[f"frame_{frame}"]]
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} frames missing, e.g. {missing[0]} — rerun stage 1")
            raw, zs_logits = encode(encoder, paths, args.batch_size, CLASSES)
            assert len(raw) == len(rows_f)
            image_embeds = l2_normalize(raw)
            labels = rows_f["label"].to_numpy()
            arrays = dict(
                video_relpath=rows_f["video_relpath"].to_numpy(dtype=object),
                label=labels,
                class_name=rows_f["class_name"].to_numpy(dtype=object),
                image_embeds_raw=raw,
                image_embeds=image_embeds,
                text_prompts=np.array(CLASS_PROMPTS, dtype=object),
                model_id=np.array(model_id, dtype=object),
            )
            if text_embeds is not None:
                arrays["text_embeds"] = text_embeds
                zs_scores = image_embeds @ text_embeds.T
            else:
                arrays["zs_logits"] = zs_logits
                zs_scores = zs_logits
            save_embeddings(out, **arrays)
            acc = float((zs_scores.argmax(axis=1) == labels).mean())
            print(f"{out.name}: {len(rows_f)} frames, dim={raw.shape[1]}, zero-shot acc={acc:.3f} (chance 0.143)")


if __name__ == "__main__":
    main()
