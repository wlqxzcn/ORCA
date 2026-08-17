from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from paper_experiment_utils import (
    EmbeddingCache,
    apply_remove_and_normalize,
    cluster_bootstrap_diff,
    cluster_signflip_p,
    extract_split,
    fit_centered_pca_basis,
    fit_raw_svd_basis,
    get_source_texts,
    load_json,
    load_model,
    mcnemar_exact,
    parse_named_ints,
    parse_named_strings,
    retrieve_and_evaluate,
    save_json,
    set_seed,
    write_csv,
)


def holm_adjust(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values[int(idx)]))
        running = max(running, value)
        adjusted[int(idx)] = running
    return [float(x) for x in adjusted]


def dialogue_counts(a_hit: np.ndarray, b_hit: np.ndarray, cluster_ids: np.ndarray) -> Dict[str, int]:
    diff = np.asarray(a_hit, dtype=np.int8) - np.asarray(b_hit, dtype=np.int8)
    unique = np.unique(cluster_ids)
    a_only_dialogues = b_only_dialogues = net_pos = net_neg = net_tie = 0
    for c in unique:
        d = diff[cluster_ids == c]
        a_only_dialogues += int(np.any(d > 0))
        b_only_dialogues += int(np.any(d < 0))
        s = int(d.sum())
        net_pos += int(s > 0)
        net_neg += int(s < 0)
        net_tie += int(s == 0)
    return {
        "n_dialogues": int(len(unique)),
        "orca_only_dialogues": int(a_only_dialogues),
        "baseline_only_dialogues": int(b_only_dialogues),
        "net_orca_better_dialogues": int(net_pos),
        "net_baseline_better_dialogues": int(net_neg),
        "net_tied_dialogues": int(net_tie),
    }


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
    cluster_permutations: int,
    cache: EmbeddingCache,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if test.cluster_ids is None:
        raise ValueError("No dialogue IDs found in test_items; cluster inference cannot run")

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
    raw_cluster_ps: List[float] = []
    baselines: List[str] = []
    for comp_id, baseline in enumerate(methods):
        if baseline == "ORCA-user-source":
            continue
        a = per_query["ORCA-user-source"]["hit5"]
        b = per_query[baseline]["hit5"]
        cs = cluster_bootstrap_diff(
            a, b, test.cluster_ids, bootstrap_samples, seed + 1000 + comp_id
        )
        sp = cluster_signflip_p(
            a, b, test.cluster_ids, cluster_permutations, seed + 2000 + comp_id
        )
        mc = mcnemar_exact(a, b)
        dc = dialogue_counts(a, b, test.cluster_ids)
        row = {
            "model": name,
            "method_a": "ORCA-user-source",
            "method_b": baseline,
            "fit_pool": "dev" if baseline.endswith("-dev") else "test",
            "baseline_type": "raw_svd" if "Raw-SVD" in baseline else "centered_pca",
            "orca_R@5": float(a.mean()),
            "baseline_R@5": float(b.mean()),
            "delta_R@5": float((a - b).mean()),
            "cluster_ci_low": cs["ci_low"],
            "cluster_ci_high": cs["ci_high"],
            "cluster_bootstrap_p": cs["p_two_sided"],
            "cluster_signflip_p": sp,
            "query_orca_only": mc["a_only"],
            "query_baseline_only": mc["b_only"],
            **dc,
        }
        pairwise_rows.append(row)
        raw_cluster_ps.append(sp)
        baselines.append(baseline)

    adjusted = holm_adjust(raw_cluster_ps)
    for row, adj in zip(pairwise_rows, adjusted):
        row["cluster_signflip_p_holm"] = adj

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
    p = argparse.ArgumentParser(description="Cluster-aware candidate-source comparison")
    p.add_argument("--data", required=True)
    p.add_argument("--model", action="append", required=True, help="Repeat NAME=PATH")
    p.add_argument("--model-batch-size", action="append", default=[])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--source-key", default="emotion_corpus")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--cluster-permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default="../cache/cross_pool_cluster")
    p.add_argument("--output-prefix", default="../results/cross_pool_cluster/cross_pool_cluster")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = load_json(args.data)
    source = get_source_texts(data, args.source_key)
    dev = extract_split(data, "dev")
    test = extract_split(data, "test")
    if test.cluster_ids is None:
        raise ValueError("test_items do not contain dialogue IDs")
    print(f"[cluster] queries={len(test.queries)} dialogues={len(np.unique(test.cluster_ids))}")

    paths = parse_named_strings(args.model)
    batch_sizes = parse_named_ints(args.model_batch_size)
    cache = EmbeddingCache(args.cache_dir, enabled=not args.no_cache)

    metrics: List[Dict[str, Any]] = []
    pairwise: List[Dict[str, Any]] = []
    geometry: Dict[str, Any] = {}
    for index, (name, path) in enumerate(paths.items()):
        m, p, g = run_model(
            name, path, batch_sizes.get(name, args.batch_size), args.device,
            source, dev, test, args.k, args.bootstrap_samples,
            args.cluster_permutations, cache, args.seed + index * 10000,
        )
        metrics.extend(m)
        pairwise.extend(p)
        geometry[name] = g

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(metrics, f"{prefix}_metrics.csv")
    write_csv(pairwise, f"{prefix}_pairwise_cluster.csv")
    save_json({
        "arguments": vars(args),
        "n_queries": len(test.queries),
        "n_dialogues": int(len(np.unique(test.cluster_ids))),
        "metrics": metrics,
        "pairwise": pairwise,
        "geometry": geometry,
    }, f"{prefix}_details.json")
    print(f"Saved: {prefix}_pairwise_cluster.csv")


if __name__ == "__main__":
    main()
