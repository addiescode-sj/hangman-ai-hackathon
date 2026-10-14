"""Fixed holdout split: grouped by target vessel, stratified by Hangul name."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N_FOLDS = 5


def is_hangul(name: object) -> bool:
    return isinstance(name, str) and any("가" <= ch <= "힣" for ch in name)


def load_train() -> pd.DataFrame:
    return pd.read_csv(ROOT / "open/train.csv", encoding="utf-8-sig")


def load_catalog() -> pd.DataFrame:
    return pd.read_csv(ROOT / "open/vessel_catalog.csv", encoding="utf-8-sig")


def folds(train: pd.DataFrame):
    """Yield (train_idx, valid_idx) for all folds; fold 0 is the fixed holdout."""
    strata = train["vessel_name"].map(is_hangul).astype(int)
    splitter = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    yield from splitter.split(train, strata, train["target_vessel_id"])


def holdout(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr, va = next(folds(train))
    return train.iloc[tr].reset_index(drop=True), train.iloc[va].reset_index(drop=True)


if __name__ == "__main__":
    t = load_train()
    tr, va = holdout(t)
    assert not set(tr.target_vessel_id) & set(va.target_vessel_id)
    print(len(tr), len(va), int(va.vessel_name.map(is_hangul).sum()))
