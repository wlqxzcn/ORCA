from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from paper_experiment_utils import (
    EmbeddingCache,
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
    query_bootstrap_diff,
    retrieve_and_evaluate,
    save_json,
    set_seed,
    write_csv,
)


def run_model(
    name: str,
    path: str,
    batch_size: int,
    device: str,
    source_texts: List[str],
    dev,
    test,
    k: int,
    bootstrap_samples: int,
    cache: EmbeddingCache,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    model = load_model(path, device)
    source = cache.encode(model, name, path, "source", source_texts, batch_size)
    dev_d = cache.encode(model, name, path, "dev_corpus", dev.corpus, batch_size)
    test_q = cache.encode(model, name, path, "test_queries", test.queries, batch_size)
    test_d = cache.encode(model, name, path, "test_corpus", test.corpus, batch_size)

    source_basis, source_values = fit_raw_svd_basis(source, k)
    dev_raw_basis, dev_raw_values = fit_raw_svd_basis(dev_d, k)
    test_raw_basis, test_raw_values = fit_raw_svd_basis(test_d, k)
    _, dev_center_basis, dev_center_values = fit_centered_pca_basis(dev_d, k)
    _, test_center_basis, test_center_values = fit_centered_pca_basis(test_d, k)

    methods = {
        "ORCA-user-source": source_basis,
        "Candidate-Raw-SVD-dev": dev_raw_basis,
        "Candidate-Raw-SVD-test": test_raw_basis,
        "Candidate-Centered-PCA-dev": dev_center_basis,
        "Candidate-Centered-PCA-test": test_center_basis,
    }

    metric_rows: List[Dict[str, Any]] = []
    per_query: Dict[str, Dict[str, np.ndarray]] = {}
    for method, basis in methods.items():
        q = apply_remove_and_normalize(test_q, basis)
        d = apply_remove_and_normalize(test_d, basis)
        metrics, pq, _, _ = retrieve_and_evaluate(q, d, test.gold, depth=10)
        metric_rows.append({"model": name, "method": method, **metrics})
        per_query[method] = pq

    pairwise_rows: List[Dict[str, Any]] = []
    for comp_id, baseline in enumerate(methods):
        if baseline == "ORCA-user-source":
            continue
        stats = query_bootstrap_diff(
            per_query["ORCA-user-source"]["hit5"],
            per_query[baseline]["hit5"],
            bootstrap_samples,
            seed + comp_id,
        )
        mc = mcnemar_exact(
            per_query["ORCA-user-source"]["hit5"], per_query[baseline]["hit5"]
        )
        pairwise_rows.append({
            "model": name,
            "method_a": "ORCA-user-source",
            "method_b": baseline,
            "fit_pool": "dev" if baseline.endswith("-dev") else "test",
            "baseline_type": "raw_svd" if "Raw-SVD" in baseline else "centered_pca",
            "orca_R@5": next(r["R@5"] for r in metric_rows if r["method"] == "ORCA-user-source"),
            "baseline_R@5": next(r["R@5"] for r in metric_rows if r["method"] == baseline),
            "delta_R@5": stats["delta"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "bootstrap_p": stats["p_two_sided"],
            "orca_only": mc["a_only"],
            "baseline_only": mc["b_only"],
            "mcnemar_p": mc["p_exact"],
        })

    geometry = {
        "source_top_eigenvalues": [float(x) for x in source_values[:10]],
        "dev_candidate_raw_top_eigenvalues": [float(x) for x in dev_raw_values[:10]],
        "test_candidate_raw_top_eigenvalues": [float(x) for x in test_raw_values[:10]],
        "dev_candidate_centered_top_eigenvalues": [float(x) for x in dev_center_values[:10]],
        "test_candidate_centered_top_eigenvalues": [float(x) for x in test_center_values[:10]],
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metric_rows, pairwise_rows, geometry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-pool paired significance (paper Table 12)")
    parser.add_argument("--data", default="../data/cleaned/psydt_task_retrieval_final.json")
    parser.add_argument("--model", action="append", required=True, help="Repeat NAME=PATH")
    parser.add_argument("--model-batch-size", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source-key", default="emotion_corpus")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="../cache/cross_pool")
    parser.add_argument("--output-prefix", default="../results/cross_pool/cross_pool")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = load_json(args.data)
    source = get_source_texts(data, args.source_key)
    dev = extract_split(data, "dev")
    test = extract_split(data, "test")
    paths = parse_named_strings(args.model)
    batch_sizes = parse_named_ints(args.model_batch_size)
    cache = EmbeddingCache(args.cache_dir, enabled=not args.no_cache)

    metrics: List[Dict[str, Any]] = []
    pairwise: List[Dict[str, Any]] = []
    geometry: Dict[str, Any] = {}
    for index, (name, path) in enumerate(paths.items()):
        m, p, g = run_model(
            name, path, batch_sizes.get(name, args.batch_size), args.device,
            source, dev, test, args.k, args.bootstrap_samples, cache,
            args.seed + index * 10000,
        )
        metrics.extend(m)
        pairwise.extend(p)
        geometry[name] = g

    prefix = Path(args.output_prefix)
    write_csv(metrics, f"{prefix}_metrics.csv")
    write_csv(pairwise, f"{prefix}_pairwise.csv")
    save_json({"arguments": vars(args), "metrics": metrics, "pairwise": pairwise, "geometry": geometry}, f"{prefix}_details.json")
    print(f"Saved: {prefix}_pairwise.csv")


if __name__ == "__main__":
    main()
