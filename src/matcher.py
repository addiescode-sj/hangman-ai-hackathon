"""Catalog index: candidate retrieval for one query row at a time."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher
from heapq import nlargest

import numpy as np
import pandas as pd

from normalize import (en_skeletons, hull_numbers, is_hangul, normalize_name, positive_float, query_skeleton,
                       reference_tonnage, romanize)


def grams(value: str, width: int) -> frozenset[str]:
    if len(value) < width:
        return frozenset((value,)) if value else frozenset()
    return frozenset(value[i : i + width] for i in range(len(value) - width + 1))


def dice(left: frozenset[str], right: frozenset[str]) -> float:
    return 2 * len(left & right) / (len(left) + len(right)) if left and right else 0.0


def deletes(value: str) -> set[str]:
    """The string and every one-character deletion (SymSpell distance-1 keys)."""
    return {value} | {value[:i] + value[i + 1 :] for i in range(len(value))}


def tonnage_gap(ref: float | None, catalog_gt: float) -> float | None:
    return abs(catalog_gt - ref) / ref if ref and catalog_gt > 0 else None


class CatalogIndex:
    def __init__(self, catalog: pd.DataFrame) -> None:
        self.catalog = catalog.reset_index(drop=True)
        if not self.catalog["vessel_id"].is_unique:
            raise ValueError("vessel_id must be unique in the catalog")
        self.ids = self.catalog["vessel_id"].astype(str).tolist()
        self.id_to_index = {v: i for i, v in enumerate(self.ids)}
        self.names = [normalize_name(v) for v in self.catalog["vessel_name"]]
        self.trigrams = [grams(v, 3) for v in self.names]
        self.skeletons = [en_skeletons(v) for v in self.catalog["vessel_name"]]
        self.skeleton_bigrams = [tuple(grams(s, 2) for s in v) for v in self.skeletons]
        self.gt = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(float)

        self.exact_name: dict[str, list[int]] = defaultdict(list)
        self.trigram_postings: dict[str, list[int]] = defaultdict(list)
        self.skeleton_postings: dict[str, list[int]] = defaultdict(list)
        self.delete_postings: dict[str, list[int]] = defaultdict(list)
        for i, name in enumerate(self.names):
            if name:
                self.exact_name[name].append(i)
                for d in deletes(name):
                    self.delete_postings[d].append(i)
            for g in self.trigrams[i]:
                self.trigram_postings[g].append(i)
            for g in frozenset().union(*self.skeleton_bigrams[i]):
                self.skeleton_postings[g].append(i)

        self.flags = self.catalog["flag_name"].tolist()
        self.types = self.catalog["vessel_type"].tolist()
        self.numbers = [hull_numbers(v) for v in self.catalog["vessel_name"]]
        # Train targets almost never miss size/DWT (EDA T1.1), so missingness is a prior.
        self.size_missing = self.catalog["size"].isna().to_numpy(float)
        self.dwt_missing = self.catalog["deadweight_tonnage"].isna().to_numpy(float)

        valid = np.flatnonzero(self.gt > 0)
        self.sorted_gt_indices = valid[np.argsort(self.gt[valid], kind="mergesort")]
        self.sorted_gt_values = self.gt[self.sorted_gt_indices]

    def skeleton_similarity(self, query_bigrams: frozenset[str], i: int) -> float:
        return max(dice(query_bigrams, g) for g in self.skeleton_bigrams[i])

    def _nearest_gt(self, value: float | None, count: int) -> list[int]:
        if value is None:
            return []
        pos = int(np.searchsorted(self.sorted_gt_values, value))
        lo, hi = max(pos - count, 0), min(pos + count, len(self.sorted_gt_values))
        window = self.sorted_gt_indices[lo:hi]
        order = np.argsort(np.abs(self.gt[window] - value), kind="mergesort")[:count]
        return window[order].tolist()

    def _skeleton_candidates(self, row: dict, limit: int) -> list[int]:
        query = grams(query_skeleton(row.get("vessel_name")), 2)
        counts: Counter[int] = Counter()
        for g in query:
            counts.update(self.skeleton_postings.get(g, ()))
        ref = reference_tonnage(row)

        def score(i: int) -> float:
            value = self.skeleton_similarity(query, i)
            gap = tonnage_gap(ref, self.gt[i])
            if gap is not None:
                value += 0.5 if gap == 0 else 0.25 if gap <= 0.065 else -0.3
            return value

        # Ties broken by catalog index so results do not depend on PYTHONHASHSEED.
        pool = nlargest(2000, counts, key=lambda i: (counts[i], -i))
        return nlargest(limit, pool, key=lambda i: (score(i), -i))

    def candidates(self, row: dict, name_limit: int = 16, tonnage_limit: int = 12,
                   skeleton_limit: int = 40) -> list[int]:
        """Union of exact-name, name-trigram, tonnage and phonetic-skeleton retrieval."""
        name = normalize_name(row.get("vessel_name"))
        out: dict[int, None] = {}
        if name:
            out.update(dict.fromkeys(self.exact_name.get(name, [])))
            counts: Counter[int] = Counter()
            for g in grams(name, 3):
                counts.update(self.trigram_postings.get(g, ()))
            out.update(dict.fromkeys(nlargest(name_limit, counts, key=lambda i: (counts[i], -i))))
            # One-edit typos in short names share few trigrams (T9.2); sorted for determinism.
            out.update(dict.fromkeys(sorted({i for d in deletes(name) for i in self.delete_postings.get(d, ())})))
        for value in {positive_float(row.get("gross_tonnage")), positive_float(row.get("international_gross_tonnage"))}:
            out.update(dict.fromkeys(self._nearest_gt(value, tonnage_limit)))
        out.update(dict.fromkeys(self._skeleton_candidates(row, skeleton_limit)))
        if is_hangul(row.get("vessel_name")):
            out.update(dict.fromkeys(self._window_candidates(row, out)))
        return list(out) or [0]

    def _window_candidates(self, row: dict, taken: dict, limit: int = 5, width: float = 0.1) -> list[int]:
        """Hangul rows: best name matches inside the +-10% tonnage window (T9.1 recovers missed targets)."""
        ref = reference_tonnage(row)
        if ref is None:
            return []
        lo = int(np.searchsorted(self.sorted_gt_values, ref * (1 - width)))
        hi = int(np.searchsorted(self.sorted_gt_values, ref * (1 + width), "right"))
        window = [i for i in self.sorted_gt_indices[lo:hi].tolist() if i not in taken]
        raw = row.get("vessel_name")
        skeleton, roman = SequenceMatcher(), SequenceMatcher()
        skeleton.set_seq2(query_skeleton(raw))
        roman.set_seq2(romanize(raw))

        def ratio(matcher: SequenceMatcher, value: str) -> float:
            matcher.set_seq1(value)
            return matcher.ratio()

        by_skeleton = nlargest(limit, window, key=lambda i: (max(ratio(skeleton, s) for s in self.skeletons[i]), -i))
        by_roman = nlargest(limit, window, key=lambda i: (ratio(roman, self.names[i]), -i))
        return by_skeleton + by_roman


class CondTable:
    """Smoothed P(catalog value | query value) learned from training pairs."""

    def __init__(self, query: list, catalog: list, alpha: float = 5.0) -> None:
        query = [q if isinstance(q, str) else None for q in query]
        self.pair = Counter(zip(query, catalog))
        self.query = Counter(query)
        self.prior = {k: v / len(catalog) for k, v in Counter(catalog).items()}
        self.alpha = alpha

    def __call__(self, query: object, catalog: object) -> float:
        if not isinstance(query, str):
            return -1.0
        return (self.pair[(query, catalog)] + self.alpha * self.prior.get(catalog, 0.0)) / (self.query[query] + self.alpha)


def pair_features(index: CatalogIndex, flag: CondTable, vtype: CondTable, row: dict, cands: list[int]) -> np.ndarray:
    """One feature row per candidate; the last block is relative to the candidate set."""
    raw = row.get("vessel_name")
    q = normalize_name(raw)
    q2, q3 = grams(q, 2), grams(q, 3)
    qs = query_skeleton(raw)
    qsg = grams(qs, 2)
    hangul = float(is_hangul(raw))
    roman = romanize(raw)
    refs = (reference_tonnage(row), positive_float(row.get("gross_tonnage")),
            positive_float(row.get("international_gross_tonnage")))
    year = float(positive_float(row.get("arrival_year")) or 0.0)
    qn = hull_numbers(raw)
    out = []
    for i in cands:
        name, skeletons, cgt, cn = index.names[i], index.skeletons[i], index.gt[i], index.numbers[i]
        gaps = [min(g, 2.0) if (g := tonnage_gap(r, cgt)) is not None else 2.0 for r in refs]
        out.append([
            hangul, float(bool(q) and q == name), dice(q2, grams(name, 2)),
            len(q3 & index.trigrams[i]) / max(1, len(q3 | index.trigrams[i])),
            SequenceMatcher(None, q, name).ratio(), min(len(q), len(name)) / max(len(q), len(name), 1),
            index.skeleton_similarity(qsg, i), float(qs in skeletons),
            max(SequenceMatcher(None, qs, s).ratio() for s in skeletons), float(len(qs)),
            *gaps, float(gaps[0] == 0), float(gaps[0] <= 0.065), float(refs[0] is None), float(not cgt > 0),
            flag(row.get("flag_name"), index.flags[i]), vtype(row.get("vessel_type"), index.types[i]), year,
            index.size_missing[i], index.dwt_missing[i], float(index.flags[i] == "Unknown"),
            SequenceMatcher(None, roman, name).ratio(),
            # Hull numbers never conflict on true train pairs (T9.2).
            float(bool(qn) and qn == cn), float(bool(qn) and bool(cn) and qn != cn), float(bool(qn) != bool(cn)),
        ])
    x = np.asarray(out, dtype=np.float32)
    relative = [x[:, c] - x[:, c].max() for c in (2, 3, 4, 6, 8, 23)] + [x[:, 10] - x[:, 10].min()]
    return np.column_stack([x, *relative, np.full(len(x), len(x), dtype=np.float32)])


LLM_MISSING = -10.0


def llm_block(scores: np.ndarray | None, n: int) -> np.ndarray:
    """[score, score - best, rank] from the LoRA scorer for Hangul rows; zeros otherwise (T9.3)."""
    if scores is None:
        return np.zeros((n, 3), np.float32)
    s = np.asarray(scores, np.float32)
    rank = np.argsort(np.argsort(-s, kind="stable"), kind="stable").astype(np.float32)
    return np.column_stack([s, s - s.max(), rank])


class Ranker:
    """Candidate retrieval + HistGradientBoosting pointwise reranker.

    With `llm_scores` (out-of-fold {(hangul_name, catalog_index): score}) the model also learns LLM features,
    and `predict` then needs a `scorer(hangul_name, catalog_names) -> scores` (src/llm_scorer.py).
    """

    use_llm = False

    def features(self, index: CatalogIndex, row: dict, cands: list[int], llm) -> np.ndarray:
        x = pair_features(index, self.flag, self.vtype, row, cands)
        if not self.use_llm:
            return x
        scores = llm(row["vessel_name"], cands) if is_hangul(row.get("vessel_name")) else None
        return np.column_stack([x, llm_block(scores, len(cands))])

    def fit(self, train: pd.DataFrame, index: CatalogIndex, llm_scores: dict | None = None) -> "Ranker":
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.use_llm = llm_scores is not None
        llm = (lambda name, cands: [llm_scores.get((name, c), LLM_MISSING) for c in cands]) if self.use_llm else None
        targets = [index.id_to_index[t] for t in train["target_vessel_id"]]
        self.flag = CondTable(train["flag_name"].tolist(), [index.flags[t] for t in targets])
        self.vtype = CondTable(train["vessel_type"].tolist(), [index.types[t] for t in targets])
        # call_sign -> vessel_id is 1:1 in train (EDA T1.7); used only as an override at inference.
        self.call_sign = dict(zip(train["call_sign"], train["target_vessel_id"]))
        xs, ys = [], []
        for row, true in zip(train.to_dict("records"), targets):
            cands = index.candidates(row)
            if true not in cands:
                cands.append(true)
            xs.append(self.features(index, row, cands, llm))
            ys.append(np.asarray([c == true for c in cands], dtype=np.int8))
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                                    random_state=42).fit(np.vstack(xs), np.concatenate(ys))
        return self

    def predict(self, queries: pd.DataFrame, index: CatalogIndex, use_call_sign: bool = True,
                llm=None) -> list[str]:
        """`llm`: a scorer(name, catalog_names), or for CV the out-of-fold score dict used in fit."""
        if self.use_llm and llm is None:
            raise ValueError("this ranker was trained with LLM features; pass llm=")
        if isinstance(llm, dict):
            table = llm
            llm = lambda name, cands: [table.get((name, c), LLM_MISSING) for c in cands]  # noqa: E731
        elif llm is not None:
            scorer = llm
            llm = lambda name, cands: scorer(name, [index.catalog["vessel_name"].iat[c] for c in cands])  # noqa: E731
        out = []
        for row in queries.to_dict("records"):
            known = self.call_sign.get(row.get("call_sign")) if use_call_sign else None
            if known in index.id_to_index:
                out.append(known)
                continue
            cands = index.candidates(row)
            scores = self.model.predict_proba(self.features(index, row, cands, llm))[:, 1]
            out.append(index.ids[cands[int(np.argmax(scores))]])
        return out


def recall_report() -> None:
    """T3.4: candidate recall on the fixed holdout, by language."""
    from split import holdout, load_catalog, load_train

    index = CatalogIndex(load_catalog())
    _, va = holdout(load_train())
    rows = []
    for row in va.to_dict("records"):
        cands = index.candidates(row)
        rows.append((is_hangul(row["vessel_name"]), index.id_to_index[row["target_vessel_id"]] in cands, len(cands)))
    report = pd.DataFrame(rows, columns=["hangul", "recall", "n_candidates"]).groupby("hangul").mean()
    print(report.round(4))


if __name__ == "__main__":
    recall_report()
