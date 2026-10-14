"""Train the reranker.

python3 src/train.py                      fit on holdout-train, score the fixed holdout
python3 src/train.py --cv                 5-fold CV (mean weighted score is the adoption criterion)
python3 src/train.py --full OUT.pkl       fit on all train rows and save the bundle
add --llm DIR to any mode to use out-of-fold LoRA scores (DIR/f*_fwd_s1.jsonl from lora_run.py, T9.3)
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from evaluate import dump_errors, score
from matcher import CatalogIndex, Ranker
from split import ROOT, folds, holdout, load_catalog, load_train


def load_llm_scores(directory: str) -> dict:
    """{(hangul_name, catalog_index): score}; each name was scored by the adapter of its own fold."""
    table = {}
    files = sorted(Path(directory).glob("f[0-9]_fwd_s1.jsonl"))
    if len(files) != 5:
        raise ValueError(f"expected 5 fold score files in {directory}, found {len(files)}")
    for f in files:
        for r in map(json.loads, open(f, encoding="utf-8")):
            table.update({(r["name"], c): s for c, s in zip(r["cands"], r["scores"])})
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", metavar="OUT.pkl")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--llm", metavar="DIR")
    args = ap.parse_args()
    start = time.time()
    catalog = load_catalog()
    index = CatalogIndex(catalog)
    train = load_train()
    llm = load_llm_scores(args.llm) if args.llm else None
    if args.full:
        ranker = Ranker().fit(train, index, llm)
        with open(args.full, "wb") as f:
            pickle.dump(ranker, f)
        print(f"saved {args.full} (use_llm={ranker.use_llm}) in {time.time() - start:.0f}s")
        return
    if args.cv:
        results = []
        for k, (tr, va) in enumerate(folds(train)):
            tr, va = train.iloc[tr], train.iloc[va].reset_index(drop=True)
            s = score(va, Ranker().fit(tr, index, llm).predict(va, index, use_call_sign=False, llm=llm))
            results.append(s)
            print(k, {a: round(b, 4) for a, b in s.items()}, flush=True)
        mean = {a: round(float(np.mean([s[a] for s in results])), 4) for a in results[0]}
        print("mean", mean, f"weighted sd {np.std([s['weighted'] for s in results]):.4f}", f"{time.time() - start:.0f}s")
        return
    tr, va = holdout(train)
    pred = Ranker().fit(tr, index, llm).predict(va, index, use_call_sign=False, llm=llm)
    print({k: round(v, 4) for k, v in score(va, pred).items()}, f"{time.time() - start:.0f}s")
    print("errors", dump_errors(va, pred, catalog, ROOT / "plans/scratch/holdout_errors.csv"))


if __name__ == "__main__":
    main()
