import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES, DEFAULT_FRAME, FRAMES_DIR, frame_relpath
from dcond.caer_vlm import VLM_DIR, load_descriptions

OUT_DIR = PROJECT_ROOT / "data" / "caer_gender_audit"


def draw_sample(desc: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    desc = desc[desc.gender.isin(["male", "female"])].copy()
    desc["cls"] = desc.video_relpath.str.split("/").str[1]
    cells = [(g, c) for g in ["male", "female"] for c in CLASSES]
    want = {cell: n // len(cells) for cell in cells}
    interleaved = [(g, c) for c in CLASSES for g in ["male", "female"]]
    for cell in interleaved[: n - sum(want.values())]:
        want[cell] += 1
    for g in ["male", "female"]:
        for c in CLASSES:
            have = int(((desc.gender == g) & (desc.cls == c)).sum())
            short = want[(g, c)] - have
            if short > 0:
                want[(g, c)] = have
                others = [cell for cell in cells if cell[0] == g and cell != (g, c)]
                for k in range(short):
                    want[others[k % len(others)]] += 1
    parts = []
    for (g, c), k in want.items():
        pool = desc[(desc.gender == g) & (desc.cls == c)]
        pick = pool.iloc[rng.choice(len(pool), size=k, replace=False)].copy()
        pick["weight"] = len(pool) / k
        parts.append(pick)
    s = pd.concat(parts)
    s = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
    s.insert(0, "item", np.arange(1, len(s) + 1))
    return s


def to_data_uri(path: Path, max_side: int) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CAER gender audit</title>
<style>
:root { --bg:#fafaf9; --ink:#1c1c1a; --muted:#6b6a66; --line:#d9d8d4; --accent:#2a78d6; --sel:#e8f0fb; }
@media (prefers-color-scheme: dark) { :root { --bg:#1b1b1a; --ink:#ecebe8; --muted:#a3a29e; --line:#3a3936; --accent:#6aa5ec; --sel:#22324a; } }
* { box-sizing:border-box; } body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.45 system-ui, sans-serif; }
main { max-width:980px; margin:0 auto; padding:16px; }
header { display:flex; flex-wrap:wrap; gap:12px; align-items:center; justify-content:space-between; }
.bar { height:6px; background:var(--line); border-radius:3px; overflow:hidden; flex:1 1 200px; }
.bar > div { height:100%; background:var(--accent); width:0; }
figure { margin:16px 0; text-align:center; } img { max-width:100%; max-height:62vh; border-radius:6px; }
.opts { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; }
button { font:inherit; padding:10px 16px; border:1px solid var(--line); background:transparent; color:var(--ink); border-radius:6px; cursor:pointer; }
button.on { background:var(--sel); border-color:var(--accent); }
.nav { display:flex; gap:8px; justify-content:space-between; margin-top:16px; }
.hint { color:var(--muted); font-size:13px; text-align:center; margin-top:8px; }
</style></head><body><main>
<header><strong>CAER perceived-gender audit</strong><span id="count"></span><div class="bar"><div id="prog"></div></div>
<button id="dl">Download CSV</button></header>
<p class="hint">Judge the <b>main person</b> of the frame (the one the scene is about). Do not guess from context you cannot see.</p>
<figure><img id="img" alt="frame"><figcaption id="cap" class="hint"></figcaption></figure>
<div class="opts">
  <button data-g="man">M &middot; man</button><button data-g="woman">W &middot; woman</button>
  <button data-g="unclear">U &middot; cannot tell</button><button id="amb">A &middot; main person ambiguous</button>
</div>
<div class="nav"><button id="prev">&larr; previous</button><button id="next">next &rarr;</button></div>
<p class="hint">Keys: M / W / U answer and go to the next frame, A toggles "ambiguous", arrows navigate. Answers are saved in this browser.</p>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = "caer_gender_audit_" + __TAG__;
let ans = {}; try { ans = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
let i = 0;
const $ = (id) => document.getElementById(id);
function save() { try { localStorage.setItem(KEY, JSON.stringify(ans)); } catch (e) {} }
function show() {
  const it = ITEMS[i], a = ans[it.item] || {};
  $("img").src = it.src; $("cap").textContent = "item " + it.item + " of " + ITEMS.length;
  document.querySelectorAll("[data-g]").forEach(b => b.classList.toggle("on", a.gender === b.dataset.g));
  $("amb").classList.toggle("on", !!a.ambiguous);
  const done = ITEMS.filter(x => ans[x.item] && ans[x.item].gender).length;
  $("count").textContent = done + " / " + ITEMS.length + " answered";
  $("prog").style.width = (100 * done / ITEMS.length) + "%";
}
function setG(g) { const it = ITEMS[i]; ans[it.item] = Object.assign({}, ans[it.item], {gender: g}); save(); if (i < ITEMS.length - 1) i++; show(); }
function toggleA() { const it = ITEMS[i]; const a = ans[it.item] || {}; ans[it.item] = Object.assign({}, a, {ambiguous: !a.ambiguous}); save(); show(); }
document.querySelectorAll("[data-g]").forEach(b => b.onclick = () => setG(b.dataset.g));
$("amb").onclick = toggleA;
$("prev").onclick = () => { if (i > 0) i--; show(); };
$("next").onclick = () => { if (i < ITEMS.length - 1) i++; show(); };
document.addEventListener("keydown", e => {
  const k = e.key.toLowerCase();
  if (k === "m") setG("man"); else if (k === "w") setG("woman"); else if (k === "u") setG("unclear");
  else if (k === "a") toggleA(); else if (e.key === "ArrowLeft") $("prev").click(); else if (e.key === "ArrowRight") $("next").click();
});
$("dl").onclick = () => {
  const rows = ["item,annotator_gender,ambiguous"].concat(ITEMS.map(it => {
    const a = ans[it.item] || {}; return [it.item, a.gender || "", a.ambiguous ? 1 : 0].join(","); }));
  const url = URL.createObjectURL(new Blob([rows.join("\\n") + "\\n"], {type: "text/csv"}));
  const l = document.createElement("a"); l.href = url; l.download = "gender_audit_annotations.csv"; l.click();
};
// start at the first unanswered item
i = Math.max(0, ITEMS.findIndex(x => !(ans[x.item] && ans[x.item].gender))); if (i < 0) i = 0;
show();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    ap.add_argument("--max-side", type=int, default=720)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    desc = load_descriptions("full", VLM_DIR)
    s = draw_sample(desc, a.n, a.seed)
    paths = []
    for r in s.itertuples():
        split, cls, name = r.video_relpath.split("/")
        paths.append(a.frames_dir / frame_relpath(split, cls, Path(name).stem, DEFAULT_FRAME))
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frames missing, e.g. {missing[0]}")
    s["frame"] = [str(p.relative_to(a.frames_dir)) for p in paths]
    s[["item", "video_relpath", "split", "cls", "frame", "gender", "weight", "people"]].rename(
        columns={"gender": "vlm_gender", "cls": "class_name"}).to_csv(a.out_dir / "key.csv", index=False)
    items = [{"item": int(r.item), "src": to_data_uri(p, a.max_side)} for r, p in zip(s.itertuples(), paths)]
    html = PAGE.replace("__ITEMS__", json.dumps(items)).replace("__TAG__", json.dumps(f"n{a.n}_seed{a.seed}"))
    (a.out_dir / "annotate.html").write_text(html, encoding="utf-8")
    print(s.groupby(["gender", "cls"]).size().unstack().to_string())
    print(f"-> {a.out_dir / 'annotate.html'} ({(a.out_dir / 'annotate.html').stat().st_size / 1e6:.1f} MB), key: {a.out_dir / 'key.csv'}")


if __name__ == "__main__":
    main()
