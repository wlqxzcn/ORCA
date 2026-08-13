from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch

from paper_experiment_utils import (
    EmbeddingCache,
    apply_center_pca_and_normalize,
    apply_remove_and_normalize,
    extract_split,
    fit_centered_pca_basis,
    fit_raw_svd_basis,
    get_source_texts,
    l2_normalize,
    load_json,
    load_model,
    mcnemar_exact,
    parse_named_ints,
    parse_named_strings,
    project_remove,
    query_bootstrap_diff,
    random_orthonormal_basis,
    retrieve_and_evaluate,
    save_json,
    set_seed,
    write_csv,
)


def run_model(
    model_name: str,
    model_path: str,
    batch_size: int,
    device: str,
    source_texts: List[str],
    split,
    cache: EmbeddingCache,
    k: int,
    random_trials: int,
    bootstrap_samples: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    model = load_model(model_path, device)
    source = cache.encode(model, model_name, model_path, "source", source_texts, batch_size)
    q_raw = cache.encode(model, model_name, model_path, f"{split.name}_queries", split.queries, batch_size)
    d_raw = cache.encode(model, model_name, model_path, f"{split.name}_corpus", split.corpus, batch_size)

    raw_basis, raw_values = fit_raw_svd_basis(source, k=max(k, 2))
    source_mean, source_center_basis, centered_values = fit_centered_pca_basis(source, k=k)
    candidate_mean, candidate_center_basis, candidate_values = fit_centered_pca_basis(d_raw, k=k)
    mean_direction = source_mean / max(float(np.linalg.norm(source_mean)), 1e-12)
    candidate_mean_direction = candidate_mean / max(float(np.linalg.norm(candidate_mean)), 1e-12)

    methods: Dict[str, tuple[np.ndarray, np.ndarray]] = {
        "Dense": (l2_normalize(q_raw), l2_normalize(d_raw)),
        f"ORCA-raw-SVD-k{k}": (
            apply_remove_and_normalize(q_raw, raw_basis[:, :k]),
            apply_remove_and_normalize(d_raw, raw_basis[:, :k]),
        ),
        "User-Mean-Direction": (
            apply_remove_and_normalize(q_raw, mean_direction[:, None]),
            apply_remove_and_normalize(d_raw, mean_direction[:, None]),
        ),
        "User-Mean-Centering": (
            l2_normalize(q_raw - source_mean[None, :]),
            l2_normalize(d_raw - source_mean[None, :]),
        ),
        f"User-Centered-PCA-k{k}": (
            apply_center_pca_and_normalize(q_raw, source_mean, source_center_basis[:, :k]),
            apply_center_pca_and_normalize(d_raw, source_mean, source_center_basis[:, :k]),
        ),
        "Candidate-Mean-Direction": (
            apply_remove_and_normalize(q_raw, candidate_mean_direction[:, None]),
            apply_remove_and_normalize(d_raw, candidate_mean_direction[:, None]),
        ),
        f"Candidate-Centered-PCA-k{k}": (
            # Match the paper's candidate-PCA baseline: fit on centered candidates,
            # then remove the fitted directions from the original vectors.
            l2_normalize(project_remove(q_raw, candidate_center_basis[:, :k])),
            l2_normalize(project_remove(d_raw, candidate_center_basis[:, :k])),
        ),
    }

    metric_rows: List[Dict[str, Any]] = []
    per_query: Dict[str, Dict[str, np.ndarray]] = {}
    for method, (q, d) in methods.items():
        metrics, pq, _, _ = retrieve_and_evaluate(q, d, split.gold, depth=10)
        metric_rows.append({"model": model_name, "split": split.name, "method": method, **metrics})
        per_query[method] = pq

    rng = np.random.default_rng(seed)
    random_metrics: List[Dict[str, float]] = []
    random_hits: List[np.ndarray] = []
    for trial in range(random_trials):
        basis = random_orthonormal_basis(q_raw.shape[1], k, rng)
        q = apply_remove_and_normalize(q_raw, basis)
        d = apply_remove_and_normalize(d_raw, basis)
        metrics, pq, _, _ = retrieve_and_evaluate(q, d, split.gold, depth=10)
        random_metrics.append(metrics)
        random_hits.append(pq["hit5"])
    random_row: Dict[str, Any] = {
        "model": model_name,
        "split": split.name,
        "method": f"Random-k{k}",
        "random_trials": random_trials,
    }
    for key in ("R@1", "R@5", "R@10", "MRR", "nDCG@10"):
        values = np.asarray([row[key] for row in random_metrics], dtype=float)
        random_row[key] = float(values.mean())
        random_row[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    metric_rows.append(random_row)

    comparisons = [
        (f"ORCA-raw-SVD-k{k}", "Dense"),
        (f"ORCA-raw-SVD-k{k}", "User-Mean-Direction"),
        (f"ORCA-raw-SVD-k{k}", "User-Mean-Centering"),
        (f"ORCA-raw-SVD-k{k}", f"User-Centered-PCA-k{k}"),
        (f"ORCA-raw-SVD-k{k}", "Candidate-Mean-Direction"),
        (f"ORCA-raw-SVD-k{k}", f"Candidate-Centered-PCA-k{k}"),
    ]
    pairwise_rows: List[Dict[str, Any]] = []
    for comp_id, (a, b) in enumerate(comparisons):
        stats = query_bootstrap_diff(
            per_query[a]["hit5"], per_query[b]["hit5"], bootstrap_samples, seed + 1000 + comp_id
        )
        mc = mcnemar_exact(per_query[a]["hit5"], per_query[b]["hit5"])
        pairwise_rows.append({
            "model": model_name,
            "split": split.name,
            "method_a": a,
            "method_b": b,
            "delta_R@5": stats["delta"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "bootstrap_p": stats["p_two_sided"],
            "a_only": mc["a_only"],
            "b_only": mc["b_only"],
            "mcnemar_p": mc["p_exact"],
        })

    geometry = {
        "model": model_name,
        "split": split.name,
        "source_n": len(source_texts),
        "dimension": int(source.shape[1]),
        "source_mean_norm": float(np.linalg.norm(source_mean)),
        "abs_cos_v1_user_mean": float(abs(np.dot(raw_basis[:, 0], mean_direction))),
        "abs_cos_v2_user_mean": float(abs(np.dot(raw_basis[:, 1], mean_direction))),
        "raw_top_eigenvalues": [float(x) for x in raw_values[:10]],
        "centered_top_eigenvalues": [float(x) for x in centered_values[:10]],
        "candidate_centered_top_eigenvalues": [float(x) for x in candidate_values[:10]],
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metric_rows, pairwise_rows, geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORCA mean-direction baselines (paper Table 10)")
    parser.add_argument("--data", default="../data/cleaned/psydt_task_retrieval_final.json")
    parser.add_argument("--model", action="append", required=True, help="Repeat NAME=PATH")
    parser.add_argument("--model-batch-size", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["dev", "test", "legacy_test"], default="dev")
    parser.add_argument("--source-key", default="emotion_corpus")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--random-trials", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="../cache/mean_direction")
    parser.add_argument("--output-prefix", default="../results/mean_direction/mean_direction")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = load_json(args.data)
    source_texts = get_source_texts(data, args.source_key)
    split = extract_split(data, args.split)
    model_paths = parse_named_strings(args.model)
    batch_sizes = parse_named_ints(args.model_batch_size)
    cache = EmbeddingCache(args.cache_dir, enabled=not args.no_cache)

    metrics: List[Dict[str, Any]] = []
    pairwise: List[Dict[str, Any]] = []
    geometry: Dict[str, Any] = {}
    for index, (name, path) in enumerate(model_paths.items()):
        m, p, g = run_model(
            name, path, batch_sizes.get(name, args.batch_size), args.device,
            source_texts, split, cache, args.k, args.random_trials,
            args.bootstrap_samples, args.seed + index * 10000,
        )
        metrics.extend(m)
        pairwise.extend(p)
        geometry[name] = g

    prefix = Path(args.output_prefix)
    write_csv(metrics, f"{prefix}_metrics.csv")
    write_csv(pairwise, f"{prefix}_pairwise.csv")
    save_json({"arguments": vars(args), "geometry": geometry, "metrics": metrics, "pairwise": pairwise}, f"{prefix}_details.json")
    print(f"Saved: {prefix}_metrics.csv")


if __name__ == "__main__":
    main()
