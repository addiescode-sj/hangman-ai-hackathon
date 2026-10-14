"""Holdout scoring (official weighted accuracy) and error dump."""

from pathlib import Path

import numpy as np
import pandas as pd

from split import ROOT, is_hangul


def score(truth: pd.DataFrame, pred: list[str]) -> dict[str, float]:
    hangul = truth["vessel_name"].map(is_hangul).to_numpy(bool)
    correct = truth["target_vessel_id"].to_numpy() == np.asarray(pred)
    h, n = float(correct[hangul].mean()), float(correct[~hangul].mean())
    return {"hangul": h, "non_hangul": n, "weighted": 0.7 * h + 0.3 * n, "top1": float(correct.mean())}


def dump_errors(truth: pd.DataFrame, pred: list[str], catalog: pd.DataFrame, path: str | Path) -> int:
    names = catalog.set_index("vessel_id")["vessel_name"]
    wrong = truth[truth["target_vessel_id"].to_numpy() != np.asarray(pred)].copy()
    wrong["pred_id"] = np.asarray(pred)[wrong.index]
    wrong["true_name"] = wrong["target_vessel_id"].map(names)
    wrong["pred_name"] = wrong["pred_id"].map(names)
    wrong.to_csv(path, index=False, encoding="utf-8-sig")
    return len(wrong)


def baseline() -> None:
    """T0.4: retrain the baseline RF on the holdout-train part and score it."""
    import sys
    from sklearn.ensemble import RandomForestClassifier

    sys.path.insert(0, str(ROOT / "baseline_submit/model"))
    from baseline_lib import CatalogIndex, make_training_pairs, predict_ids
    from split import holdout, load_catalog, load_train

    catalog = load_catalog()
    tr, va = holdout(load_train())
    index = CatalogIndex(catalog)
    x, y, recall = make_training_pairs(tr, index)
    model = RandomForestClassifier(n_estimators=60, max_depth=12, min_samples_leaf=10,
                                   class_weight="balanced_subsample", n_jobs=4, random_state=42).fit(x, y)
    pred = predict_ids(model, va, index)
    print("train retrieval recall", round(recall, 4), score(va, pred))
    dump_errors(va, pred, catalog, ROOT / "plans/scratch/baseline_errors.csv")


if __name__ == "__main__":
    baseline()
