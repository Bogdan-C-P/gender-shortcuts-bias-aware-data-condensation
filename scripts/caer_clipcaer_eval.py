import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcond.caer import CLASSES
from dcond.caer_frames import DEDUP_JSON, load_dedup

DATA = PROJECT_ROOT / "data" / "caer_clipcaer"


def rel_from_line(line: str) -> str:
    d = Path(line.split(" ")[0])
    return f"{d.parts[-3]}/{d.parts[-2]}/{d.parts[-1]}.avi"


def metrics(y, p, keep):
    y, pred = y[keep], p[keep].argmax(1)
    rec = [(pred[y == c] == c).mean() for c in range(len(CLASSES))]
    return {"war": 100 * (pred == y).mean(), "uar": 100 * np.mean(rec), "n": int(keep.sum()),
            **{f"recall_{c}": 100 * r for c, r in zip(CLASSES, rec)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    dup = load_dedup(DEDUP_JSON)
    rels = {s: [rel_from_line(l) for l in (DATA / f"{s}.txt").read_text().split("\n") if l.strip()]
            for s in ["validation", "test"]}
    tf = {s: ~np.isin(rels[s], list(dup[s])) for s in rels}

    rows = []
    for f in sorted(a.run.glob("test_probs_e*.npz")):
        e = int(f.stem[-2:])
        t = np.load(f)
        assert len(t["label"]) == len(rels["test"]), "test order/length mismatch"
        r = {"epoch": e}
        for name, keep in [("test", np.ones(len(t["label"]), bool)), ("test_twinfree", tf["test"])]:
            r.update({f"{name}_{k}": v for k, v in metrics(t["label"], t["probs"], keep).items()})
        v = a.run / f"val_probs_e{e:02d}.npz"
        if v.exists():
            v = np.load(v)
            r.update({f"val_{k}": x for k, x in metrics(v["label"], v["probs"], np.ones(len(v["label"]), bool)).items()})
            r.update({f"val_twinfree_{k}": x for k, x in metrics(v["label"], v["probs"], tf["validation"]).items()})
        rows.append(r)
    ep = pd.DataFrame(rows)
    ep.round(2).to_csv(a.run / "repro_epochs.csv", index=False)
    print(ep[["epoch", "val_uar", "test_war", "test_uar", "test_twinfree_war", "test_twinfree_uar"]].round(2).to_string(index=False))

    picks = {"paper code (best test UAR)": ep["test_uar"].idxmax(), "honest (best val UAR)": ep["val_uar"].idxmax(),
             "last epoch": ep.index[-1]}
    out = []
    for name, i in picks.items():
        r = ep.loc[i]
        out.append({"selection": name, "epoch": int(r["epoch"]), "full_test_war": r["test_war"], "full_test_uar": r["test_uar"],
                    "twinfree_war": r["test_twinfree_war"], "twinfree_uar": r["test_twinfree_uar"]})
    s = pd.DataFrame(out)
    s.round(2).to_csv(a.run / "repro_summary.csv", index=False)
    print("\npublished: CLIP-CAER 81.77 (labelled UAR), CAER-Net 77.04 (plain accuracy)\n")
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
