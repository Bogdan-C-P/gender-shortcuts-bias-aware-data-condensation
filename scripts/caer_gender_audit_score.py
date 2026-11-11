import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "experiments" / "results" / "caer_gender_audit"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(c - h), float(c + h)


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in np.union1d(a, b))
    return float((po - pe) / (1 - pe)) if pe < 1 else np.nan


def block(d: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> dict:
    agree = (d.vlm_gender == d.human).to_numpy()
    k, n = int(agree.sum()), len(d)
    lo, hi = wilson(k, n)
    w = d.weight.to_numpy()
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        boots.append(np.average(agree[i], weights=w[i]))
    out = {"n": n, "agreement": k / n if n else np.nan, "agreement_lo": lo, "agreement_hi": hi,
           "kappa": kappa(d.vlm_gender.to_numpy(), d.human.to_numpy()),
           "agreement_weighted": float(np.average(agree, weights=w)) if n else np.nan,
           "agreement_weighted_lo": float(np.percentile(boots, 2.5)), "agreement_weighted_hi": float(np.percentile(boots, 97.5))}
    for g in ["male", "female"]:
        m = (d.vlm_gender == g).to_numpy()
        out[f"precision_vlm_{g}"] = float(agree[m].mean()) if m.any() else np.nan
        out[f"n_vlm_{g}"] = int(m.sum())
    return out


DEFAULT_DIR = PROJECT_ROOT / "data" / "caer_gender_audit"


def find_annotations(given: Path | None) -> Path:
    if given is not None and given.exists():
        return given
    names = ["gender_audit_annotations*.csv"]
    places = [DEFAULT_DIR, Path.home() / "Downloads", Path.home() / "Descărcări"]
    found = sorted((p for d in places if d.is_dir() for n in names for p in d.glob(n)), key=lambda p: p.stat().st_mtime)
    if not found:
        raise SystemExit(f"annotations not found ({given or 'no path given'}); looked in {', '.join(map(str, places))}.\n"
                         "Annotate in annotate.html, press 'Download CSV', and pass the downloaded file's path.")
    print(f"using {found[-1]} (newest of {len(found)} found)")
    return found[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations", type=Path, nargs="?", default=None,
                    help="the CSV from 'Download CSV' (default: newest gender_audit_annotations*.csv in "
                         "data/caer_gender_audit or ~/Downloads)")
    ap.add_argument("--key", type=Path, default=DEFAULT_DIR / "key.csv")
    a = ap.parse_args()
    a.annotations = find_annotations(a.annotations)
    key = pd.read_csv(a.key)
    ann = pd.read_csv(a.annotations, keep_default_na=False)
    d = key.merge(ann, on="item", how="left")
    d["annotator_gender"] = d["annotator_gender"].fillna("")
    d["ambiguous"] = d["ambiguous"].fillna(0).astype(int).astype(bool)
    d["human"] = d.annotator_gender.map({"man": "male", "woman": "female", "unclear": "unclear"}).fillna("")
    missing = int((d.human == "").sum())
    if missing:
        print(f"WARNING: {missing} of {len(d)} items have no answer -- they are left out")
    d = d[d.human != ""]
    decided = d[d.human.isin(["male", "female"])]
    res = {"n_sample": len(key), "n_answered": len(d), "n_unclear": int((d.human == "unclear").sum()),
           "n_ambiguous": int(d.ambiguous.sum()),
           "all_decided": block(decided), "decided_not_ambiguous": block(decided[~decided.ambiguous])}
    res["confusion_vlm_by_human"] = pd.crosstab(decided.vlm_gender, decided.human).to_dict()
    res["per_class"] = {c: {"n": int(len(g)), "agreement": float((g.vlm_gender == g.human).mean()),
                            "unclear_or_ambiguous": int(((d.class_name == c) & ((d.human == "unclear") | d.ambiguous)).sum())}
                        for c, g in decided.groupby("class_name")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    dis = d[(d.human != d.vlm_gender)]
    dis[["item", "video_relpath", "class_name", "frame", "vlm_gender", "human", "ambiguous", "people"]].to_csv(OUT / "disagreements.csv", index=False)

    r = res["all_decided"]
    print(json.dumps(res, indent=1))
    print(f"\nSummary:\n  the annotator could decide {r['n']} of {res['n_answered']} "
          f"frames ({res['n_unclear']} 'cannot tell', {res['n_ambiguous']} marked ambiguous); on them the annotator and the VLM "
          f"agree on {100 * r['agreement']:.1f}% (95% CI {100 * r['agreement_lo']:.1f}-{100 * r['agreement_hi']:.1f}; "
          f"{100 * r['agreement_weighted']:.1f}% re-weighted to the whole dataset), Cohen's kappa = {r['kappa']:.2f}.")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
