from __future__ import annotations
import csv
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss
except Exception:
    faiss = None

try:
    import torch
except Exception:
    torch = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def text_hash(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_named_strings(values: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=VALUE, got {value!r}")
        name, item = value.split("=", 1)
        name, item = name.strip(), item.strip()
        if not name or not item:
            raise ValueError(f"Invalid NAME=VALUE: {value!r}")
        out[name] = item
    return out


def parse_named_ints(values: Sequence[str]) -> Dict[str, int]:
    return {k: int(v) for k, v in parse_named_strings(values).items()}


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, eps)).astype(np.float32)


def stable_dedup(texts: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for text in texts:
        value = str(text).strip()
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def first_existing(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of these keys exists: {list(keys)}")


def parse_gold_item(item: Any) -> List[int]:
    if isinstance(item, (int, np.integer)):
        return [int(item)]
    if isinstance(item, str) and item.strip().lstrip("-").isdigit():
        return [int(item.strip())]
    if isinstance(item, (list, tuple, np.ndarray)):
        values: List[int] = []
        for part in item:
            values.extend(parse_gold_item(part))
        return sorted(set(values))
    if isinstance(item, Mapping):
        for key in (
            "gold_indices", "positive_indices", "relevant_indices",
            "gold_index", "positive_index", "target_index", "target_idx",
            "index", "idx", "gt",
        ):
            if key in item:
                return parse_gold_item(item[key])
    raise TypeError(f"Unsupported gold format: {item!r}")


@dataclass
class RetrievalSplit:
    name: str
    queries: List[str]
    corpus: List[str]
    gold: List[List[int]]
    cluster_ids: Optional[np.ndarray] = None


def extract_split(data: Mapping[str, Any], split: str) -> RetrievalSplit:
    split_alias = "test" if split == "legacy_test" else split
    queries = [
        str(x).strip() for x in first_existing(
            data, (f"{split_alias}_queries", f"{split_alias}_query_texts")
        )
    ]
    corpus = [
        str(x).strip() for x in first_existing(
            data,
            (
                f"{split_alias}_task_corpus", f"{split_alias}_corpus",
                f"{split_alias}_candidates",
            ),
        )
    ]
    gold = [
        parse_gold_item(x) for x in first_existing(
            data, (f"{split_alias}_gt", f"{split_alias}_gold")
        )
    ]
    if len(queries) != len(gold):
        raise ValueError(f"{split}: queries={len(queries)} gold={len(gold)}")
    for qid, ids in enumerate(gold):
        if not ids:
            raise ValueError(f"{split}: query {qid} has no gold item")
        if any(idx < 0 or idx >= len(corpus) for idx in ids):
            raise IndexError(f"{split}: invalid gold for query {qid}: {ids}")

    cluster_ids: Optional[np.ndarray] = None
    items = data.get(f"{split_alias}_items")
    if isinstance(items, list) and len(items) == len(queries):
        candidates = (
            "dialog_id", "dialogue_id", "conversation_id", "conv_id",
            "transcript_id", "session_id",
        )
        values: List[str] = []
        valid = True
        for item in items:
            value: Any = None
            if isinstance(item, Mapping):
                for key in candidates:
                    if key in item:
                        value = item[key]
                        break
                if value is None:
                    for nested_key in ("pair", "item", "metadata", "meta"):
                        nested = item.get(nested_key)
                        if isinstance(nested, Mapping):
                            for key in candidates:
                                if key in nested:
                                    value = nested[key]
                                    break
                        if value is not None:
                            break
            if value is None:
                valid = False
                break
            values.append(str(value))
        if valid:
            cluster_ids = np.asarray(values, dtype=object)

    return RetrievalSplit(split, queries, corpus, gold, cluster_ids)


def get_source_texts(data: Mapping[str, Any], key: str = "emotion_corpus") -> List[str]:
    if key not in data:
        raise KeyError(f"Source key {key!r} not found; keys={list(data)[:40]}")
    values = stable_dedup(str(x).strip() for x in data[key])
    if len(values) < 2:
        raise ValueError("Source corpus must contain at least two texts")
    return values


class EmbeddingCache:
    def __init__(self, root: str | Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def path(
        self,
        model_name: str,
        model_path: str,
        label: str,
        texts: Sequence[str],
    ) -> Path:
        digest = hashlib.sha256()
        digest.update(str(Path(model_path).resolve()).encode("utf-8"))
        digest.update(label.encode("utf-8"))
        digest.update(text_hash(texts).encode("ascii"))
        return self.root / f"{safe_key(model_name)}__{safe_key(label)}__{digest.hexdigest()[:20]}.npy"

    def encode(
        self,
        model: SentenceTransformer,
        model_name: str,
        model_path: str,
        label: str,
        texts: Sequence[str],
        batch_size: int,
    ) -> np.ndarray:
        path = self.path(model_name, model_path, label, texts)
        if self.enabled and path.exists():
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape[0] == len(texts):
                print(f"[cache] loaded {path} {arr.shape}")
                return arr.astype(np.float32)
            print(f"[cache] invalid shape, recomputing: {path}")
        arr = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32)
        if self.enabled:
            np.save(path, arr)
            print(f"[cache] saved {path} {arr.shape}")
        return arr


def load_model(path: str, device: str):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required to load encoders")
    return SentenceTransformer(path, device=device)


def _orthonormalize(basis: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(basis, dtype=np.float64))
    return q.astype(np.float32)


def fit_raw_svd_basis(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    gram = (x64.T @ x64) / max(len(x64), 1)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    basis = _orthonormalize(vectors[:, :k])
    return basis, values


def fit_centered_pca_basis(x: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    mean = x64.mean(axis=0)
    centered = x64 - mean
    gram = (centered.T @ centered) / max(len(centered), 1)
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    basis = _orthonormalize(vectors[:, :k])
    return mean.astype(np.float32), basis, values


def project_remove(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    b = np.asarray(basis, dtype=np.float32)
    return (x - (x @ b) @ b.T).astype(np.float32)


def apply_remove_and_normalize(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return l2_normalize(project_remove(x, basis))


def apply_center_pca_and_normalize(x: np.ndarray, mean: np.ndarray, basis: np.ndarray) -> np.ndarray:
    centered = np.asarray(x, dtype=np.float32) - np.asarray(mean, dtype=np.float32)[None, :]
    return l2_normalize(project_remove(centered, basis))


def random_orthonormal_basis(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(dim, k))
    return _orthonormalize(matrix)[:, :k]


def search_ip(
    queries: np.ndarray,
    corpus: np.ndarray,
    top_k: int,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    q = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
    c = np.ascontiguousarray(np.asarray(corpus, dtype=np.float32))
    top_k = min(int(top_k), len(c))
    if faiss is not None:
        index = faiss.IndexFlatIP(c.shape[1])
        index.add(c)
        scores, ids = index.search(q, top_k)
        return scores.astype(np.float32), ids.astype(np.int32)
    all_scores: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []
    for start in range(0, len(q), batch_size):
        block = q[start:start + batch_size] @ c.T
        if top_k == len(c):
            ids = np.argsort(-block, axis=1)
        else:
            part = np.argpartition(-block, top_k - 1, axis=1)[:, :top_k]
            part_scores = np.take_along_axis(block, part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            ids = np.take_along_axis(part, order, axis=1)
        scores = np.take_along_axis(block, ids, axis=1)
        all_scores.append(scores.astype(np.float32))
        all_ids.append(ids.astype(np.int32))
    return np.vstack(all_scores), np.vstack(all_ids)


def evaluate_ranking(
    indices: np.ndarray,
    gold: Sequence[Sequence[int]],
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    n = len(gold)
    hit1 = np.zeros(n, dtype=np.int8)
    hit5 = np.zeros(n, dtype=np.int8)
    hit10 = np.zeros(n, dtype=np.int8)
    ranks = np.full(n, indices.shape[1] + 1, dtype=np.int32)
    for i, row in enumerate(indices):
        gold_set = {int(x) for x in gold[i]}
        for rank0, doc_id in enumerate(row):
            if int(doc_id) in gold_set:
                ranks[i] = rank0 + 1
                break
        hit1[i] = int(ranks[i] <= 1)
        hit5[i] = int(ranks[i] <= 5)
        hit10[i] = int(ranks[i] <= 10)
    reciprocal = np.where(ranks <= indices.shape[1], 1.0 / ranks, 0.0)
    reciprocal10 = np.where(ranks <= 10, 1.0 / ranks, 0.0)
    ndcg10 = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0)
    metrics = {
        "R@1": float(hit1.mean()),
        "R@5": float(hit5.mean()),
        "R@10": float(hit10.mean()),
        "MRR": float(reciprocal.mean()),
        "MRR@10": float(reciprocal10.mean()),
        "nDCG@10": float(ndcg10.mean()),
        "mean_gold_rank": float(ranks.mean()),
        "median_gold_rank": float(np.median(ranks)),
    }
    per_query = {
        "hit1": hit1,
        "hit5": hit5,
        "hit10": hit10,
        "rank": ranks,
        "rr": reciprocal.astype(np.float32),
        "rr10": reciprocal10.astype(np.float32),
        "ndcg10": ndcg10.astype(np.float32),
    }
    return metrics, per_query


def retrieve_and_evaluate(
    queries: np.ndarray,
    corpus: np.ndarray,
    gold: Sequence[Sequence[int]],
    depth: int = 10,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    scores, ids = search_ip(queries, corpus, depth)
    metrics, per_query = evaluate_ranking(ids, gold)
    return metrics, per_query, scores, ids


def two_sided_from_samples(samples: np.ndarray) -> float:
    n = len(samples)
    left = (np.sum(samples <= 0.0) + 1.0) / (n + 1.0)
    right = (np.sum(samples >= 0.0) + 1.0) / (n + 1.0)
    return float(min(1.0, 2.0 * min(left, right)))


def query_bootstrap_diff(a: np.ndarray, b: np.ndarray, samples: int, seed: int) -> Dict[str, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    n = len(diff)
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        draws[i] = diff[idx].mean()
    return {
        "delta": float(diff.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_two_sided": two_sided_from_samples(draws),
    }


def cluster_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    cluster_ids: np.ndarray,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    clusters = np.unique(cluster_ids)
    members = {c: np.flatnonzero(cluster_ids == c) for c in clusters}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for i in range(samples):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        total = 0.0
        count = 0
        for cluster in sampled:
            idx = members[cluster]
            total += float(diff[idx].sum())
            count += int(len(idx))
        draws[i] = total / max(count, 1)
    return {
        "delta": float(diff.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_two_sided": two_sided_from_samples(draws),
        "n_clusters": int(len(clusters)),
    }


def cluster_signflip_p(
    a: np.ndarray,
    b: np.ndarray,
    cluster_ids: np.ndarray,
    permutations: int,
    seed: int,
) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    clusters = np.unique(cluster_ids)
    cluster_sums = np.asarray([diff[cluster_ids == c].sum() for c in clusters], dtype=np.float64)
    observed = abs(float(diff.mean()))
    rng = np.random.default_rng(seed)
    count = 0
    denominator = max(len(diff), 1)
    for _ in range(permutations):
        signs = rng.choice((-1.0, 1.0), size=len(cluster_sums))
        value = abs(float((cluster_sums * signs).sum() / denominator))
        count += int(value >= observed - 1e-15)
    return float((count + 1) / (permutations + 1))


def exact_binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    observed = math.comb(n, k) * (0.5 ** n)
    total = 0.0
    for i in range(n + 1):
        probability = math.comb(n, i) * (0.5 ** n)
        if probability <= observed + 1e-15:
            total += probability
    return float(min(1.0, total))


def mcnemar_exact(a_hit: np.ndarray, b_hit: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(a_hit, dtype=np.int8)
    b = np.asarray(b_hit, dtype=np.int8)
    a_only = int(np.sum((a == 1) & (b == 0)))
    b_only = int(np.sum((a == 0) & (b == 1)))
    discordant = a_only + b_only
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_exact": exact_binomial_two_sided(min(a_only, b_only), discordant),
    }


def vector_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-15:
        return float("nan")
    cosine = float(np.clip(abs(np.dot(a, b) / denom), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def projection_distance(a: np.ndarray, b: np.ndarray) -> float:
    pa = np.asarray(a, dtype=np.float64) @ np.asarray(a, dtype=np.float64).T
    pb = np.asarray(b, dtype=np.float64) @ np.asarray(b, dtype=np.float64).T
    return float(np.linalg.norm(pa - pb, ord="fro"))
