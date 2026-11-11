import argparse
import base64
import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES, DEFAULT_FRAME, FRAME_POSITIONS, FRAMES_DIR, MANIFEST_CSV, SPLITS, load_manifest

DEFAULT_MODEL = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "caer05_vlm"
EMOTIONS = [c.lower() for c in CLASSES]
ISSUES = ["dark", "blurry", "letterbox", "text_overlay", "not_a_real_face", "occluded_face", "multiple_faces",
          "low_resolution"]

PROMPT = f"""This is a frame from a TV series{{crop_note}}. Describe it for a dataset audit.
Answer ONLY with one JSON object, no other text, with exactly these keys:
"caption": one sentence describing the image,
"setting": where the scene takes place (e.g. living room, kitchen, office, bar, street, car, hospital, bedroom, stage), or "unclear",
"people": number of clearly visible people (integer),
"main_person": the most prominent person: apparent gender, apparent age group, hair, clothing, accessories,
"action": what the main person is doing (e.g. talking, listening, eating, on the phone, hugging, crying, laughing),
"facial_expression": the main person's facial expression in a few words, or "face not visible",
"emotion": the emotion shown by the main person's face, one of {json.dumps(EMOTIONS + ["unclear"])},
"emotion_confidence": "low", "medium" or "high",
"on_screen_text": any visible text, subtitles, logos or credits, or "",
"image_issues": the items from {json.dumps(ISSUES)} that apply, or []"""


def prompt_for(tag: str) -> str:
    return PROMPT.replace("{crop_note}", ", cropped around a face" if tag == "face" else "")


def load_image(path: Path, max_side: int, min_side: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if min(w, h) * scale < min_side:
        scale = min_side / min(w, h)
    if scale != 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    return img


def parse(text: str) -> dict | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(text[i : j + 1])
    except json.JSONDecodeError:
        return None


class VllmBackend:
    def __init__(self, model: str, args):
        import os

        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
        from vllm import LLM, SamplingParams

        mm_kwargs = {"max_pixels": args.mm_max_pixels} if args.mm_max_pixels else None
        self.llm = LLM(model=model, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
                       max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
                       limit_mm_per_prompt={"image": 1}, mm_processor_kwargs=mm_kwargs, seed=0)
        self.params = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    @staticmethod
    def _data_url(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def generate(self, images: list, prompt: str) -> list[tuple[str, int]]:
        convs = [[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": self._data_url(im)}},
                                               {"type": "text", "text": prompt}]}] for im in images]
        outs = self.llm.chat(convs, self.params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False})
        return [(o.outputs[0].text, len(o.outputs[0].token_ids)) for o in outs]


class TransformersBackend:
    def __init__(self, model: str, args):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.model = AutoModelForImageTextToText.from_pretrained(model, dtype="auto", device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(model)
        self.max_new_tokens = args.max_new_tokens

    def generate(self, images: list, prompt: str) -> list[tuple[str, int]]:
        res = []
        for im in images:
            msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": prompt}]}]
            inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                        return_tensors="pt", enable_thinking=False).to(self.model.device)
            with self.torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            new = out[0, inputs["input_ids"].shape[1]:]
            res.append((self.processor.decode(new, skip_special_tokens=True), int(len(new))))
        return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="vllm", choices=["vllm", "transformers"])
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--frame", default=DEFAULT_FRAME, choices=list(FRAME_POSITIONS))
    parser.add_argument("--manifest", type=Path, default=MANIFEST_CSV)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--tag", default="", help="'face' for face crops (changes the prompt and the file name)")
    parser.add_argument("--per-class", type=int, default=0, help="pilot: N random frames per class per split")
    parser.add_argument("--max-side", type=int, default=640, help="downscale so the longest side is <= this")
    parser.add_argument("--min-side", type=int, default=224, help="upscale so the shortest side is >= this")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--chunk", type=int, default=64, help="frames per generate call (and per file append)")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="prompt (~350) + image (<=~400 at 640 px) + answer (<=320) fits in ~1100 tokens")
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--mm-max-pixels", type=int, default=640 * 640,
                        help="vLLM image budget (mm_processor_kwargs max_pixels); 0 = model default (huge)")
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-mem", type=float, default=0.90)
    parser.add_argument("--full-run-size", type=int, default=13175, help="frames in a full run, for the time projection")
    parser.add_argument("--out-dir", type=Path, default=None, help=f"default {OUT_DIR}/<model name>")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    col = f"frame_{args.frame}"
    manifest = manifest[manifest[col].notna() & (manifest[col].astype(str) != "")]
    out_dir = args.out_dir or OUT_DIR / args.model.rstrip("/").split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_{args.tag}" if args.tag else "") + (f"_pilot{args.per_class}" if args.per_class else "")
    prompt = prompt_for(args.tag)

    backend = None
    t_gen, n_gen, n_tok, n_fail = 0.0, 0, 0, 0
    agree = []
    for split in args.splits:
        rows = manifest[manifest["split"] == split]
        if args.per_class:
            rows = rows.sample(frac=1.0, random_state=0).groupby("class_name").head(args.per_class)
        rows = rows.reset_index(drop=True)
        out = out_dir / f"descriptions_{split}_{args.frame}{suffix}.jsonl"
        done = set()
        if out.exists():
            done = {json.loads(l)["video_relpath"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
        todo = rows[~rows["video_relpath"].isin(done)].reset_index(drop=True)
        print(f"{out.name}: {len(rows)} frames, {len(done)} already done, {len(todo)} to go")
        if todo.empty:
            continue
        if backend is None:
            backend = (VllmBackend if args.backend == "vllm" else TransformersBackend)(args.model, args)
        with out.open("a", encoding="utf-8") as f:
            for i in range(0, len(todo), args.chunk):
                part = todo.iloc[i : i + args.chunk]
                images = [load_image(args.frames_dir / p, args.max_side, args.min_side) for p in part[col]]
                t0 = time.perf_counter()
                gens = backend.generate(images, prompt)
                t_gen += time.perf_counter() - t0
                for (_, r), (text, ntok) in zip(part.iterrows(), gens):
                    parsed = parse(text)
                    n_gen, n_tok, n_fail = n_gen + 1, n_tok + ntok, n_fail + (parsed is None)
                    if parsed and isinstance(parsed.get("emotion"), str):
                        agree.append(parsed["emotion"].strip().lower() == r["class_name"].lower())
                    f.write(json.dumps({"video_relpath": r["video_relpath"], "class_name": r["class_name"],
                                        "model_id": args.model, "raw": text, "parsed": parsed,
                                        "n_out_tokens": ntok}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {min(i + args.chunk, len(todo))}/{len(todo)}  {n_gen / t_gen:.2f} img/s")

    if n_gen:
        sec = t_gen / n_gen
        print(f"\n{n_gen} frames in {t_gen / 60:.1f} min (model loading excluded): {sec:.2f} s/frame, "
              f"{n_tok / n_gen:.0f} output tokens/frame, JSON parse failures {n_fail}/{n_gen}")
        if agree:
            print(f"VLM 'emotion' field == CAER label: {sum(agree) / len(agree):.3f} (chance 0.143; n={len(agree)})")
        print(f"projected full run ({args.full_run_size} frames): {args.full_run_size * sec / 3600:.1f} h")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
