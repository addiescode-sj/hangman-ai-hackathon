"""DACON inference entry point: phonetic-skeleton retrieval + HistGradientBoosting reranker."""

from pathlib import Path
import pickle
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "model"))
from matcher import CatalogIndex  # noqa: E402  (also needed to unpickle Ranker)


def run(
    test_path: str | Path = "data/test.csv",
    catalog_path: str | Path = "data/vessel_catalog.csv",
    model_path: str | Path = "model/ranker.pkl",
    output_path: str | Path = "output/submission.csv",
) -> Path:
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    with open(model_path, "rb") as f:
        ranker = pickle.load(f)
    if test["sample_id"].isna().any() or not test["sample_id"].is_unique:
        raise ValueError("test.csv sample_id must be non-null and unique")

    index = CatalogIndex(catalog)
    scorer = None
    if getattr(ranker, "use_llm", False):
        from llm_scorer import LLMScorer

        scorer = LLMScorer(str(Path(model_path).parent / "llm"))
    predictions = ranker.predict(test, index, llm=scorer)
    if len(predictions) != len(test) or not set(predictions).issubset(index.id_to_index):
        raise ValueError("Invalid catalog predictions")
    result = pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": predictions})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(result)} predictions to {output_path}")
    return output_path


if __name__ == "__main__":
    run()
