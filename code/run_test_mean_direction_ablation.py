from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import faiss
except Exception:
    faiss = None

try:
    import torch
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    raise RuntimeError(
        "This script requires torch and sentence-transformers."
    ) from exc


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def text_sha256(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def parse_named_strings(values: Sequence[str]) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=VALUE, got {value!r}")
        name, item = value.split("=", 1)
        output[name.strip()] = item.strip()
    return output


def parse_named_ints(values: Sequence[str]) -> Dict[str, int]:
    return {name: int(value) for name, value in parse_named_strings(values).items()}


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)



def first_existing(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    raise KeyError(f"None of these keys exists: {list(keys)}")


def parse_gold_item(item: Any) -> List[int]:
    if isinstance(item, (int, np.integer)):
        return [int(item)]
    if isinstance(item, str):
        value = item.strip()
        if value.lstrip("-").isdigit():
            return [int(value)]
    if isinstance(item, (list, tuple, np.ndarray)):
        out: List[int] = []
        for value in item:
            out.extend(parse_gold_item(value))
        return sorted(set(out))
    if isinstance(item, Mapping):
        for key in (
            "gold_indices", "positive_indices", "relevant_indices",
            "gold_index", "positive_index", "target_index", "target_idx",
            "index", "idx",
        ):
            if key in item:
                return parse_gold_item(item[key])
    raise TypeError(f"Unsupported gold format: {item!r}")


def extract_test_data(data: Mapping[str, Any]) -> Tuple[List[str], List[str], List[List[int]]]:
    queries = [
        str(x).strip()
        for x in first_existing(data, ("test_queries", "test_query_texts"))
    ]
    corpus = [
        str(x).strip()
        for x in first_existing(
            data,
            ("test_task_corpus", "test_corpus", "test_candidates", "task_corpus"),
        )
    ]
    gold = [
        parse_gold_item(x)
        for x in first_existing(data, ("test_gt", "test_gold", "med_gt"))
    ]
    if len(queries) != len(gold):
        raise ValueError(f"queries={len(queries)} but gold={len(gold)}")
    for query_id, gold_ids in enumerate(gold):
        if not gold_ids:
            raise ValueError(f"Query {query_id} has no gold candidate.")
        for doc_id in gold_ids:
            if doc_id < 0 or doc_id >= len(corpus):
                raise IndexError(
                    f"Gold index {doc_id} outside candidate pool size {len(corpus)}"
                )
    return queries, corpus, gold


def extract_cluster_ids(data: Mapping[str, Any], n_queries: int) -> Optional[np.ndarray]:
    """Extract dialogue IDs from test_items when available."""
    items = data.get("test_items")
    if not isinstance(items, list) or len(items) != n_queries:
        return None

    candidate_keys = (
        "dialog_id", "dialogue_id", "conversation_id", "conv_id",
        "transcript_id", "session_id",
    )
    cluster_ids: List[str] = []
    for item in items:
        if not isinstance(item, Mapping):
            return None
        value: Any = None
        for key in candidate_keys:
            if key in item:
                value = item[key]
                break
        if value is None:
            # Some builders keep the original pair under a nested key.
            for nested_key in ("pair", "item", "metadata", "meta"):
                nested = item.get(nested_key)
                if isinstance(nested, Mapping):
                    for key in candidate_keys:
                        if key in nested:
                            value = nested[key]
                            break
                if value is not None:
                    break
        if value is None:
            return None
        cluster_ids.append(str(value))
    return np.asarray(cluster_ids, dtype=object)



class EmbeddingCache:
    def __init__(self, root: str | Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def _path(
        self,
        model_name: str,
        model_path: str,
        label: str,
        texts: Sequence[str],
    ) -> Path:
        digest = hashlib.sha256()
        digest.update(model_name.encode("utf-8"))
        digest.update(str(Path(model_path).resolve()).encode("utf-8"))
        digest.update(label.encode("utf-8"))
        digest.update(text_sha256(texts).encode("ascii"))
        return self.root / (
            f"{safe_key(model_name)}__{safe_key(label)}__"
            f"{digest.hexdigest()[:20]}.npy"
        )

    def encode(
        self,
        model: SentenceTransformer,
        model_name: str,
        model_path: str,
        label: str,
        texts: Sequence[str],
        batch_size: int,
    ) -> np.ndarray:
        path = self._path(model_name, model_path, label, texts)
        if self.enabled and path.exists():
            arr = np.load(path)
            if arr.ndim == 2 and arr.shape[0] == len(texts):
                print(f"[cache hit] {label}: {path} {arr.shape}")
                return np.asarray(arr, dtype=np.float32)

        print(f"[encode] {label}: n={len(texts)}")
        arr = model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(texts):
            raise RuntimeError(f"Unexpected embedding shape for {label}: {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"NaN/Inf in embeddings: {label}")
        if self.enabled:
            np.save(path, arr)
            print(f"[cache write] {path}")
        return arr



def top_eigenvectors_second_moment(x: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    x64 = np.asarray(x, dtype=np.float64)
    gram = x64.T @ x64
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1][:k]
    basis = eigvecs[:, order]
    basis, _ = np.linalg.qr(basis)
    return basis.astype(np.float32), eigvals[order].astype(np.float64)


def top_centered_pca(x: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.asarray(x, dtype=np.float64).mean(axis=0)
    centered = np.asarray(x, dtype=np.float64) - mu[None, :]
    gram = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1][:k]
    basis = eigvecs[:, order]
    basis, _ = np.linalg.qr(basis)
    return mu.astype(np.float32), basis.astype(np.float32), eigvals[order].astype(np.float64)


def remove_basis(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    return x - (x @ basis) @ basis.T


def apply_method(
    method: str,
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    *,
    raw_basis: np.ndarray,
    source_mean: np.ndarray,
    mean_direction: np.ndarray,
    centered_pca_basis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if method == "Dense":
        q, c = query_embs, corpus_embs
    elif method == "ORCA-v1":
        q = remove_basis(query_embs, raw_basis[:, :1])
        c = remove_basis(corpus_embs, raw_basis[:, :1])
    elif method == "ORCA-v2":
        q = remove_basis(query_embs, raw_basis[:, 1:2])
        c = remove_basis(corpus_embs, raw_basis[:, 1:2])
    elif method == "ORCA-k2":
        q = remove_basis(query_embs, raw_basis[:, :2])
        c = remove_basis(corpus_embs, raw_basis[:, :2])
    elif method == "User-Mean-Direction":
        basis = mean_direction.reshape(-1, 1)
        q = remove_basis(query_embs, basis)
        c = remove_basis(corpus_embs, basis)
    elif method == "User-Mean-Centering":
        q = query_embs - source_mean[None, :]
        c = corpus_embs - source_mean[None, :]
    elif method == "User-Centered-PCA-k2":
        q0 = query_embs - source_mean[None, :]
        c0 = corpus_embs - source_mean[None, :]
        q = remove_basis(q0, centered_pca_basis[:, :2])
        c = remove_basis(c0, centered_pca_basis[:, :2])
    else:
        raise ValueError(f"Unknown method: {method}")

    return l2_normalize(q), l2_normalize(c)


def exact_search(
    queries: np.ndarray,
    candidates: np.ndarray,
    top_k: int,
    query_block_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    top_k = min(int(top_k), candidates.shape[0])
    queries = np.asarray(queries, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)

    if faiss is not None:
        index = faiss.IndexFlatIP(candidates.shape[1])
        index.add(candidates)
        scores, indices = index.search(queries, top_k)
        return scores, indices

    all_scores: List[np.ndarray] = []
    all_indices: List[np.ndarray] = []
    for start in range(0, queries.shape[0], query_block_size):
        end = min(start + query_block_size, queries.shape[0])
        scores = queries[start:end] @ candidates.T
        if top_k == candidates.shape[0]:
            idx = np.argsort(-scores, axis=1)
        else:
            part = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
            part_scores = np.take_along_axis(scores, part, axis=1)
            order = np.argsort(-part_scores, axis=1)
            idx = np.take_along_axis(part, order, axis=1)
        sorted_scores = np.take_along_axis(scores, idx, axis=1)
        all_scores.append(sorted_scores.astype(np.float32))
        all_indices.append(idx.astype(np.int32))
    return np.vstack(all_scores), np.vstack(all_indices)


def evaluate_ranking(indices: np.ndarray, gold: Sequence[Sequence[int]]) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    n = len(gold)
    hit1 = np.zeros(n, dtype=np.int8)
    hit5 = np.zeros(n, dtype=np.int8)
    hit10 = np.zeros(n, dtype=np.int8)
    ranks = np.full(n, indices.shape[1] + 1, dtype=np.int32)

    for i in range(n):
        gold_set = set(int(x) for x in gold[i])
        ranking = indices[i]
        for rank0, doc_id in enumerate(ranking):
            if int(doc_id) in gold_set:
                ranks[i] = rank0 + 1
                break
        hit1[i] = int(ranks[i] <= 1)
        hit5[i] = int(ranks[i] <= 5)
        hit10[i] = int(ranks[i] <= 10)

    reciprocal = np.where(ranks <= indices.shape[1], 1.0 / ranks, 0.0)
    reciprocal_at10 = np.where(ranks <= 10, 1.0 / ranks, 0.0)
    ndcg10 = np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0)

    metrics = {
        "R@1": float(hit1.mean()),
        "R@5": float(hit5.mean()),
        "R@10": float(hit10.mean()),
        "MRR@10": float(reciprocal_at10.mean()),
        "MRR_exact": float(reciprocal.mean()),
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
        "rr10": reciprocal_at10.astype(np.float32),
        "ndcg10": ndcg10.astype(np.float32),
    }
    return metrics, per_query


def two_sided_from_samples(samples: np.ndarray) -> float:
    n = len(samples)
    left = (np.sum(samples <= 0.0) + 1.0) / (n + 1.0)
    right = (np.sum(samples >= 0.0) + 1.0) / (n + 1.0)
    return float(min(1.0, 2.0 * min(left, right)))


def query_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
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
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    unique = np.unique(cluster_ids)
    members = {cluster: np.flatnonzero(cluster_ids == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)

    for i in range(samples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        numerator = 0.0
        denominator = 0
        for cluster in sampled:
            idx = members[cluster]
            numerator += float(diff[idx].sum())
            denominator += int(len(idx))
        draws[i] = numerator / max(denominator, 1)

    return {
        "delta": float(diff.mean()),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_two_sided": two_sided_from_samples(draws),
        "n_clusters": int(len(unique)),
    }


def exact_binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0
    observed_prob = math.comb(n, k) * (0.5 ** n)
    p = 0.0
    for i in range(n + 1):
        prob = math.comb(n, i) * (0.5 ** n)
        if prob <= observed_prob + 1e-15:
            p += prob
    return float(min(1.0, p))


def mcnemar_exact(a_hit: np.ndarray, b_hit: np.ndarray) -> Dict[str, Any]:
    a_hit = np.asarray(a_hit, dtype=np.int8)
    b_hit = np.asarray(b_hit, dtype=np.int8)
    a_only = int(np.sum((a_hit == 1) & (b_hit == 0)))
    b_only = int(np.sum((a_hit == 0) & (b_hit == 1)))
    discordant = a_only + b_only
    return {
        "a_only": a_only,
        "b_only": b_only,
        "discordant": discordant,
        "p_exact": exact_binomial_two_sided(min(a_only, b_only), discordant),
    }


def cluster_sign_flip_p(
    a: np.ndarray,
    b: np.ndarray,
    cluster_ids: np.ndarray,
    permutations: int,
    seed: int,
) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    unique = np.unique(cluster_ids)
    cluster_sums = np.asarray(
        [diff[cluster_ids == cluster].sum() for cluster in unique],
        dtype=np.float64,
    )
    observed = abs(float(cluster_sums.sum() / len(diff)))
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(permutations):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(unique))
        statistic = abs(float(np.sum(cluster_sums * signs) / len(diff)))
        extreme += int(statistic >= observed - 1e-15)
    return float((extreme + 1) / (permutations + 1))


def build_markdown(
    metric_rows: Sequence[Mapping[str, Any]],
    pairwise_rows: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    cluster_available: bool,
) -> str:
    lines = [
        "# ORCA test-set mean-direction ablation",
        "",
        "This is a fixed post-review test-set confirmation. It does not reselect k or redefine the main method.",
        "",
        "## Geometry",
        "",
        f"- |cos(v1, user mean)| = {geometry['abs_cos_v1_mean']:.8f}",
        f"- |cos(v2, user mean)| = {geometry['abs_cos_v2_mean']:.8f}",
        f"- top raw second-moment eigenvalues = {geometry['raw_eigenvalues']}",
        "",
        "## Test metrics",
        "",
        "| Model | Method | R@1 | R@5 | R@10 | MRR@10 | Exact MRR | nDCG@10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['model']} | {row['method']} | {row['R@1']:.4f} | "
            f"{row['R@5']:.4f} | {row['R@10']:.4f} | {row['MRR@10']:.4f} | "
            f"{row['MRR_exact']:.4f} | {row['nDCG@10']:.4f} |"
        )

    lines.extend([
        "",
        "## Paired R@5 comparisons",
        "",
        "| Model | Comparison A−B | Δ | Query 95% CI | Query p | McNemar p | Cluster 95% CI | Cluster p |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in pairwise_rows:
        cluster_ci = (
            f"[{row['cluster_ci_low']:+.4f}, {row['cluster_ci_high']:+.4f}]"
            if row.get("cluster_ci_low") is not None else "N/A"
        )
        cluster_p = (
            f"{row['cluster_permutation_p']:.6g}"
            if row.get("cluster_permutation_p") is not None else "N/A"
        )
        lines.append(
            f"| {row['model']} | {row['comparison']} | {row['delta']:+.4f} | "
            f"[{row['query_ci_low']:+.4f}, {row['query_ci_high']:+.4f}] | "
            f"{row['query_bootstrap_p']:.6g} | {row['mcnemar_p']:.6g} | "
            f"{cluster_ci} | {cluster_p} |"
        )

    lines.extend([
        "",
        "## Interpretation discipline",
        "",
        "- The decisive comparisons are ORCA-k2 vs User-Mean-Direction and ORCA-k2 vs ORCA-v1.",
        "- If their confidence intervals contain zero, the second direction has no established independent test-set contribution.",
        "- ORCA-v2 vs Dense tests whether the second direction is useful by itself.",
        "- Results are test-set confirmation analyses added after review and must not be described as preregistered model selection.",
        "- Dialogue-cluster statistics are available." if cluster_available else
          "- Dialogue IDs were not found in test_items; only query-level statistics were produced.",
        "",
    ])
    return "\n".join(lines)


def run_model(
    *,
    model_name: str,
    model_path: str,
    batch_size: int,
    device: str,
    source_texts: Sequence[str],
    test_queries: Sequence[str],
    test_corpus: Sequence[str],
    test_gold: Sequence[Sequence[int]],
    cluster_ids: Optional[np.ndarray],
    cache: EmbeddingCache,
    bootstrap_samples: int,
    cluster_permutations: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], Dict[str, np.ndarray]]:
    print("=" * 100)
    print(f"[model] {model_name}: {model_path}")
    print("=" * 100)

    model = SentenceTransformer(model_path, device=device)
    source_embs = cache.encode(
        model, model_name, model_path, "source", source_texts, batch_size
    )
    query_embs = cache.encode(
        model, model_name, model_path, "test_queries", test_queries, batch_size
    )
    corpus_embs = cache.encode(
        model, model_name, model_path, "test_corpus", test_corpus, batch_size
    )

    if source_embs.shape[1] != query_embs.shape[1] or source_embs.shape[1] != corpus_embs.shape[1]:
        raise ValueError("Embedding dimensions do not match.")

    raw_basis, raw_eigenvalues = top_eigenvectors_second_moment(source_embs, k=2)
    source_mean, centered_basis, centered_eigenvalues = top_centered_pca(source_embs, k=2)
    mean_norm = float(np.linalg.norm(source_mean))
    if mean_norm <= 1e-12:
        raise ValueError("Source mean is numerically zero; mean-direction ablation is undefined.")
    mean_direction = (source_mean / mean_norm).astype(np.float32)

    geometry = {
        "model": model_name,
        "source_n": int(len(source_texts)),
        "dimension": int(source_embs.shape[1]),
        "source_mean_norm": mean_norm,
        "abs_cos_v1_mean": float(abs(np.dot(raw_basis[:, 0], mean_direction))),
        "abs_cos_v2_mean": float(abs(np.dot(raw_basis[:, 1], mean_direction))),
        "raw_eigenvalues": [float(x) for x in raw_eigenvalues],
        "centered_eigenvalues": [float(x) for x in centered_eigenvalues],
    }

    methods = [
        "Dense",
        "ORCA-v1",
        "ORCA-v2",
        "ORCA-k2",
        "User-Mean-Direction",
        "User-Mean-Centering",
        "User-Centered-PCA-k2",
    ]

    metric_rows: List[Dict[str, Any]] = []
    per_method: Dict[str, Dict[str, np.ndarray]] = {}
    npz_payload: Dict[str, np.ndarray] = {}

    for method in methods:
        print(f"[retrieval] {model_name} / {method}")
        q, c = apply_method(
            method,
            query_embs,
            corpus_embs,
            raw_basis=raw_basis,
            source_mean=source_mean,
            mean_direction=mean_direction,
            centered_pca_basis=centered_basis,
        )
        start = time.perf_counter()
        _, indices = exact_search(q, c, top_k=len(test_corpus))
        elapsed = time.perf_counter() - start
        metrics, per_query = evaluate_ranking(indices, test_gold)
        per_method[method] = per_query

        row: Dict[str, Any] = {
            "model": model_name,
            "method": method,
            **metrics,
            "search_seconds": elapsed,
        }
        metric_rows.append(row)
        print(
            f"  R@5={metrics['R@5']:.4f} | R@1={metrics['R@1']:.4f} | "
            f"R@10={metrics['R@10']:.4f} | MRR@10={metrics['MRR@10']:.4f} | "
            f"ExactMRR={metrics['MRR_exact']:.4f}"
        )

        prefix = f"{safe_key(model_name)}__{safe_key(method)}"
        for name, array in per_query.items():
            npz_payload[f"{prefix}__{name}"] = array

    comparisons = [
        ("ORCA-k2", "User-Mean-Direction"),
        ("ORCA-k2", "ORCA-v1"),
        ("ORCA-v2", "Dense"),
        ("ORCA-v1", "Dense"),
        ("User-Mean-Direction", "Dense"),
        ("User-Mean-Centering", "Dense"),
        ("User-Centered-PCA-k2", "Dense"),
        ("ORCA-k2", "Dense"),
    ]

    pairwise_rows: List[Dict[str, Any]] = []
    for comp_id, (a_name, b_name) in enumerate(comparisons):
        a_hit = per_method[a_name]["hit5"]
        b_hit = per_method[b_name]["hit5"]
        query_stats = query_bootstrap_diff(
            a_hit, b_hit, bootstrap_samples, seed + 1000 + comp_id
        )
        mc = mcnemar_exact(a_hit, b_hit)

        row = {
            "model": model_name,
            "comparison": f"{a_name} - {b_name}",
            "method_a": a_name,
            "method_b": b_name,
            "delta": query_stats["delta"],
            "query_ci_low": query_stats["ci_low"],
            "query_ci_high": query_stats["ci_high"],
            "query_bootstrap_p": query_stats["p_two_sided"],
            "a_only_hit": mc["a_only"],
            "b_only_hit": mc["b_only"],
            "mcnemar_p": mc["p_exact"],
            "cluster_ci_low": None,
            "cluster_ci_high": None,
            "cluster_bootstrap_p": None,
            "cluster_permutation_p": None,
            "n_clusters": None,
        }

        if cluster_ids is not None:
            cluster_stats = cluster_bootstrap_diff(
                a_hit,
                b_hit,
                cluster_ids,
                bootstrap_samples,
                seed + 2000 + comp_id,
            )
            row.update({
                "cluster_ci_low": cluster_stats["ci_low"],
                "cluster_ci_high": cluster_stats["ci_high"],
                "cluster_bootstrap_p": cluster_stats["p_two_sided"],
                "n_clusters": cluster_stats["n_clusters"],
                "cluster_permutation_p": cluster_sign_flip_p(
                    a_hit,
                    b_hit,
                    cluster_ids,
                    cluster_permutations,
                    seed + 3000 + comp_id,
                ),
            })

        pairwise_rows.append(row)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metric_rows, pairwise_rows, geometry, npz_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen test-set mean-direction ablation for ORCA."
    )
    parser.add_argument(
        "--data",
        default="../data/cleaned/psydt_task_retrieval_final.json",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeat NAME=PATH, e.g. --model gte=../models/thenlper/gte-large-zh",
    )
    parser.add_argument(
        "--model-batch-size",
        action="append",
        default=[],
        help="Optional NAME=INTEGER; unspecified models use --batch-size.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-key", default="emotion_corpus")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--cluster-permutations", type=int, default=10000)
    parser.add_argument("--cache-dir", default="../cache/test_mean_ablation")
    parser.add_argument(
        "--output-prefix",
        default="../results/test_mean_ablation/test_mean_ablation",
    )
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data = load_json(args.data)
    test_queries, test_corpus, test_gold = extract_test_data(data)
    if args.source_key not in data:
        raise KeyError(
            f"Source key {args.source_key!r} not found. Available keys: {list(data)[:50]}"
        )
    source_texts = [str(x).strip() for x in data[args.source_key] if str(x).strip()]
    if len(source_texts) < 2:
        raise ValueError("Source corpus must contain at least two items.")

    cluster_ids = extract_cluster_ids(data, len(test_queries))
    if cluster_ids is None:
        print("[warning] No dialogue IDs found in test_items; cluster statistics disabled.")
    else:
        print(
            f"[cluster] n_queries={len(cluster_ids)}, "
            f"n_dialogues={len(np.unique(cluster_ids))}"
        )

    model_paths = parse_named_strings(args.model)
    model_batch_sizes = parse_named_ints(args.model_batch_size)
    cache = EmbeddingCache(args.cache_dir, enabled=not args.no_cache)

    all_metric_rows: List[Dict[str, Any]] = []
    all_pairwise_rows: List[Dict[str, Any]] = []
    all_geometry: Dict[str, Any] = {}
    all_npz: Dict[str, np.ndarray] = {}

    for model_index, (model_name, model_path) in enumerate(model_paths.items()):
        metric_rows, pairwise_rows, geometry, npz_payload = run_model(
            model_name=model_name,
            model_path=model_path,
            batch_size=model_batch_sizes.get(model_name, args.batch_size),
            device=args.device,
            source_texts=source_texts,
            test_queries=test_queries,
            test_corpus=test_corpus,
            test_gold=test_gold,
            cluster_ids=cluster_ids,
            cache=cache,
            bootstrap_samples=args.bootstrap_samples,
            cluster_permutations=args.cluster_permutations,
            seed=args.seed + model_index * 10000,
        )
        all_metric_rows.extend(metric_rows)
        all_pairwise_rows.extend(pairwise_rows)
        all_geometry[model_name] = geometry
        all_npz.update(npz_payload)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = f"{prefix}_metrics.csv"
    pairwise_path = f"{prefix}_pairwise.csv"
    details_path = f"{prefix}_details.json"
    per_query_path = f"{prefix}_per_query.npz"
    summary_path = f"{prefix}_summary.md"

    write_csv(all_metric_rows, metrics_path)
    write_csv(all_pairwise_rows, pairwise_path)
    np.savez_compressed(per_query_path, **all_npz)

    details = {
        "arguments": vars(args),
        "data": {
            "path": str(Path(args.data).resolve()),
            "source_key": args.source_key,
            "source_n": len(source_texts),
            "source_sha256": text_sha256(source_texts),
            "test_queries": len(test_queries),
            "test_candidates": len(test_corpus),
            "test_query_sha256": text_sha256(test_queries),
            "test_corpus_sha256": text_sha256(test_corpus),
            "cluster_statistics_available": cluster_ids is not None,
            "n_dialogues": int(len(np.unique(cluster_ids))) if cluster_ids is not None else None,
        },
        "geometry": all_geometry,
        "metrics": all_metric_rows,
        "pairwise": all_pairwise_rows,
        "protocol_note": (
            "Fixed post-review TEST confirmation. Source, benchmark, encoder, "
            "normalization order, and k=2 remain unchanged; no test-driven "
            "hyperparameter selection is performed."
        ),
    }
    save_json(details, details_path)


    first_geometry = next(iter(all_geometry.values()))
    markdown = build_markdown(
        all_metric_rows,
        all_pairwise_rows,
        first_geometry,
        cluster_ids is not None,
    )
    Path(summary_path).write_text(markdown, encoding="utf-8")

    print("\n[done]")
    print(f"  metrics   : {metrics_path}")
    print(f"  pairwise  : {pairwise_path}")
    print(f"  details   : {details_path}")
    print(f"  per-query : {per_query_path}")
    print(f"  summary   : {summary_path}")


if __name__ == "__main__":
    main()
