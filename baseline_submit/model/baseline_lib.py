"""Small, row-independent candidate retrieval and RandomForest features.

Only the official vessel catalog is indexed. Every query row is processed alone;
no feature depends on another evaluation row or on the evaluation distribution.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from heapq import nlargest
import unicodedata

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "name_exact",
    "name_bigram_jaccard",
    "name_trigram_jaccard",
    "name_prefix_ratio",
    "name_length_ratio",
    "gross_tonnage_gap",
    "international_gross_tonnage_gap",
    "best_tonnage_gap",
    "query_tonnage_missing",
    "catalog_tonnage_missing",
)
QUERY_COLUMNS = (
    "sample_id",
    "vessel_name",
    "gross_tonnage",
    "international_gross_tonnage",
)


def normalize_name(value: object) -> str:
    """Normalize spelling and spacing without converting Korean to English."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).upper()
    return "".join(ch for ch in value if ch.isalnum())


def grams(value: str, width: int) -> frozenset[str]:
    if len(value) < width:
        return frozenset((value,)) if value else frozenset()
    return frozenset(value[i : i + width] for i in range(len(value) - width + 1))


def positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None


def relative_gap(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 2.0
    return min(abs(left - right) / max(left, right, 1.0), 2.0)


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class CatalogIndex:
    """Immutable lookup structures built from the official public catalog."""

    def __init__(self, catalog: pd.DataFrame) -> None:
        needed = {"vessel_id", "vessel_name", "gross_tonnage"}
        if not needed.issubset(catalog.columns):
            raise ValueError(f"Catalog is missing: {sorted(needed - set(catalog.columns))}")
        self.catalog = catalog.reset_index(drop=True)
        if not self.catalog["vessel_id"].is_unique:
            raise ValueError("vessel_id must be unique in the catalog")

        self.ids = self.catalog["vessel_id"].astype(str).tolist()
        self.id_to_index = {vessel_id: i for i, vessel_id in enumerate(self.ids)}
        self.names = [normalize_name(v) for v in self.catalog["vessel_name"]]
        self.bigram_sets = [grams(v, 2) for v in self.names]
        self.trigram_sets = [grams(v, 3) for v in self.names]
        self.gt = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)

        self.exact_name: dict[str, list[int]] = defaultdict(list)
        self.trigram_postings: dict[str, list[int]] = defaultdict(list)
        for i, name in enumerate(self.names):
            if name:
                self.exact_name[name].append(i)
            for gram in self.trigram_sets[i]:
                self.trigram_postings[gram].append(i)

        valid = np.flatnonzero(np.isfinite(self.gt) & (self.gt > 0))
        ordering = np.argsort(self.gt[valid], kind="mergesort")
        self.sorted_gt_indices = valid[ordering]
        self.sorted_gt_values = self.gt[self.sorted_gt_indices]

    def _nearest_gt(self, value: float | None, count: int) -> list[int]:
        if value is None or len(self.sorted_gt_values) == 0:
            return []
        pos = int(np.searchsorted(self.sorted_gt_values, value))
        left, right = pos - 1, pos
        found: list[int] = []
        while len(found) < count and (left >= 0 or right < len(self.sorted_gt_values)):
            if right >= len(self.sorted_gt_values):
                choose_left = True
            elif left < 0:
                choose_left = False
            else:
                choose_left = abs(self.sorted_gt_values[left] - value) <= abs(
                    self.sorted_gt_values[right] - value
                )
            found.append(int(self.sorted_gt_indices[left if choose_left else right]))
            left -= int(choose_left)
            right += int(not choose_left)
        return found

    def candidates(
        self,
        row: dict,
        name_limit: int = 16,
        tonnage_limit: int = 12,
        max_candidates: int = 40,
    ) -> list[int]:
        """Retrieve likely catalog rows from one query row only."""
        query_name = normalize_name(row.get("vessel_name"))
        gross = positive_float(row.get("gross_tonnage"))
        international = positive_float(row.get("international_gross_tonnage"))
        output: list[int] = []
        used: set[int] = set()

        def append(indices: list[int]) -> None:
            for index in indices:
                if index not in used and len(output) < max_candidates:
                    output.append(index)
                    used.add(index)

        if query_name:
            exact = self.exact_name.get(query_name, [])
            append(sorted(exact, key=lambda i: (min(relative_gap(gross, positive_float(self.gt[i])),
                                                     relative_gap(international, positive_float(self.gt[i]))), i))[:name_limit])
            counts: Counter[int] = Counter()
            for gram in grams(query_name, 3):
                counts.update(self.trigram_postings.get(gram, ()))
            near_names = nlargest(name_limit, counts, key=lambda i: (counts[i], -i))
            append(near_names)

        append(self._nearest_gt(gross, tonnage_limit))
        if international != gross:
            append(self._nearest_gt(international, tonnage_limit))
        if not output:
            output.append(0)
        return output

    def pair_features(self, row: dict, candidate_index: int) -> list[float]:
        query_name = normalize_name(row.get("vessel_name"))
        catalog_name = self.names[candidate_index]
        gross = positive_float(row.get("gross_tonnage"))
        international = positive_float(row.get("international_gross_tonnage"))
        catalog_gt = positive_float(self.gt[candidate_index])
        gross_gap = relative_gap(gross, catalog_gt)
        international_gap = relative_gap(international, catalog_gt)
        prefix = 0
        for left, right in zip(query_name, catalog_name):
            if left != right:
                break
            prefix += 1
        return [
            float(bool(query_name) and query_name == catalog_name),
            jaccard(grams(query_name, 2), self.bigram_sets[candidate_index]),
            jaccard(grams(query_name, 3), self.trigram_sets[candidate_index]),
            prefix / max(len(query_name), len(catalog_name), 1),
            min(len(query_name), len(catalog_name)) / max(len(query_name), len(catalog_name), 1),
            gross_gap,
            international_gap,
            min(gross_gap, international_gap),
            float(gross is None and international is None),
            float(catalog_gt is None),
        ]


def make_training_pairs(
    train: pd.DataFrame, index: CatalogIndex, max_negatives: int = 20
) -> tuple[np.ndarray, np.ndarray, float]:
    """Positive pair plus retrieved negative pairs for every official Train row."""
    if "target_vessel_id" not in train.columns:
        raise ValueError("Training data needs target_vessel_id")
    features: list[list[float]] = []
    labels: list[int] = []
    retrieved = 0
    for row in train.to_dict("records"):
        true_id = row["target_vessel_id"]
        if true_id not in index.id_to_index:
            raise ValueError(f"Training target not in catalog: {true_id}")
        true_index = index.id_to_index[true_id]
        candidates = index.candidates(row)
        retrieved += int(true_index in candidates)
        negatives = [i for i in candidates if i != true_index][:max_negatives]
        for candidate_index in [true_index] + negatives:
            features.append(index.pair_features(row, candidate_index))
            labels.append(int(candidate_index == true_index))
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        retrieved / len(train),
    )


def predict_ids(
    model, queries: pd.DataFrame, index: CatalogIndex, batch_size: int = 200
) -> list[str]:
    """Predict one catalog ID per row without consulting other query rows."""
    missing = set(QUERY_COLUMNS) - set(queries.columns)
    if missing:
        raise ValueError(f"Query data is missing: {sorted(missing)}")
    predictions: list[str] = []
    for start in range(0, len(queries), batch_size):
        rows = queries.iloc[start : start + batch_size].to_dict("records")
        features: list[list[float]] = []
        candidate_lists: list[list[int]] = []
        for row in rows:
            candidates = index.candidates(row)
            candidate_lists.append(candidates)
            features.extend(index.pair_features(row, i) for i in candidates)
        matrix = np.asarray(features, dtype=np.float32)
        scores = model.predict_proba(matrix)[:, 1]
        offset = 0
        for candidates in candidate_lists:
            local = scores[offset : offset + len(candidates)]
            predictions.append(index.ids[candidates[int(np.argmax(local))]])
            offset += len(candidates)
    return predictions


def is_hangul_name(value: object) -> bool:
    return isinstance(value, str) and any("가" <= char <= "힣" for char in value)


def local_weighted_accuracy(truth: pd.DataFrame, predicted_ids: list[str]) -> dict[str, float]:
    if len(truth) != len(predicted_ids):
        raise ValueError("Prediction count differs from validation data")
    language = truth["vessel_name"].map(is_hangul_name).to_numpy(dtype=bool)
    correct = truth["target_vessel_id"].to_numpy() == np.asarray(predicted_ids)
    if not language.any() or language.all():
        raise ValueError("Validation must contain both language groups")
    hangul = float(correct[language].mean())
    non_hangul = float(correct[~language].mean())
    return {
        "hangul_accuracy": hangul,
        "non_hangul_accuracy": non_hangul,
        "weighted_accuracy": 0.7 * hangul + 0.3 * non_hangul,
    }
