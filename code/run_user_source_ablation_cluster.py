from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from paper_experiment_utils import (
    EmbeddingCache,
    apply_remove_and_normalize,
    cluster_bootstrap_diff,
    cluster_signflip_p,
    extract_split,
    fit_raw_svd_basis,
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
from run_user_source_ablation_repeated import build_source_pools, sample_source


def two_sided_from_samples(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    n = len(samples)
    left = (np.sum(samples <= 0.0) + 1.0) / (n + 1.0)
    right = (np.sum(samples >= 0.0) + 1.0) / (n + 1.0)
    return float(min(1.0, 2.0 * min(left, right)))


def dialogue_counts(a_hit: np.ndarray, b_hit: np.ndarray, cluster_ids: np.ndarray) -> Dict[str, int]:
    a = np.asarray(a_hit, dtype=np.int8)
    b = np.asarray(b_hit, dtype=np.int8)
    diff = a - b
    unique = np.unique(cluster_ids)
    a_only_dialogues = 0
    b_only_dialogues = 0
    net_positive = 0
    net_negative = 0
    net_tie = 0
    for c in unique:
        d = diff[cluster_ids == c]
        a_only_dialogues += int(np.any(d > 0))
        b_only_dialogues += int(np.any(d < 0))
        s = int(d.sum())
        net_positive += int(s > 0)
        net_negative += int(s < 0)
        net_tie += int(s == 0)
    return {
        "n_dialogues": int(len(unique)),
        "a_only_dialogues": int(a_only_dialogues),
        "b_only_dialogues": int(b_only_dialogues),
        "net_positive_dialogues": int(net_positive),
        "net_negative_dialogues": int(net_negative),
        "net_tied_dialogues": int(net_tie),
    }


def hierarchical_cluster_bootstrap(
    main_hit: np.ndarray,
    alt_hits: Sequence[np.ndarray],
    cluster_ids: np.ndarray,
    samples: int,
    seed: int,
) -> Dict[str, float]:
    
    main_hit = np.asarray(main_hit, dtype=np.float64)
    alt_hits = [np.asarray(x, dtype=np.float64) for x in alt_hits]
    clusters = np.unique(cluster_ids)
    members = {c: np.flatnonzero(cluster_ids == c) for c in clusters}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)

    for i in range(samples):
        alt = alt_hits[int(rng.integers(0, len(alt_hits)))]
        diff = main_hit - alt
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        total = 0.0
        count = 0
        for c in sampled:
            idx = members[c]
            total += float(diff[idx].sum())
            count += int(len(idx))
        draws[i] = total / max(count, 1)

    replicate_deltas = np.asarray([float((main_hit - x).mean()) for x in alt_hits], dtype=np.float64)
    return {
        "delta_mean_over_repeats": float(replicate_deltas.mean()),
        "delta_std_over_repeats": float(replicate_deltas.std(ddof=1)) if len(replicate_deltas) > 1 else 0.0,
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "p_two_sided": two_sided_from_samples(draws),
        "n_dialogues": int(len(clusters)),
        "source_repeats": int(len(alt_hits)),
    }


def run_model(
    *,
    name: str,
    path: str,
    batch_size: int,
    device: str,
    pools: Mapping[str, List[str]],
    split,
    source_size: int,
    repeats: int,
    k: int,
    cache: EmbeddingCache,
    bootstrap_samples: int,
    cluster_permutations: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if split.cluster_ids is None:
        raise ValueError(
            "No dialogue IDs found in test_items. Rebuild psydt_task_retrieval_final.json "
            "with test_items containing dialog_id/dialogue_id before cluster inference."
        )

    model = load_model(path, device)
    q_raw = cache.encode(model, name, path, f"{split.name}_queries", split.queries, batch_size)
    d_raw = cache.encode(model, name, path, f"{split.name}_corpus", split.corpus, batch_size)
    rng = np.random.default_rng(seed)

    run_rows: List[Dict[str, Any]] = []
    hit_by_source: Dict[str, List[np.ndarray]] = {}

    for source_name, pool in pools.items():
        n_repeats = 1 if source_name == "main_affective_user" else repeats
        hit_by_source[source_name] = []
        for repeat in range(n_repeats):
            texts = sample_source(
                pool,
                source_size,
                rng,
                fixed=(source_name == "main_affective_user"),
            )
            source = cache.encode(
                model, name, path, f"source_{source_name}_r{repeat}", texts, batch_size
            )
            basis, values = fit_raw_svd_basis(source, k)
            q = apply_remove_and_normalize(q_raw, basis)
            d = apply_remove_and_normalize(d_raw, basis)
            metrics, pq, _, _ = retrieve_and_evaluate(q, d, split.gold, depth=10)
            hit5 = np.asarray(pq["hit5"], dtype=np.int8)
            hit_by_source[source_name].append(hit5)
            run_rows.append({
                "model": name,
                "split": split.name,
                "source": source_name,
                "source_pool_n": len(pool),
                "source_n": len(texts),
                "repeat": repeat,
                "k": k,
                **metrics,
                "selected_energy_ratio": float(values[:k].sum() / max(values.sum(), 1e-15)),
            })

    if "main_affective_user" not in hit_by_source:
        raise ValueError("--sources must include main_affective_user as the reference condition")
    main_hit = hit_by_source["main_affective_user"][0]

    per_repeat_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for source_id, (source_name, alt_hits) in enumerate(hit_by_source.items()):
        if source_name == "main_affective_user":
            continue

        alt_run_rows = [r for r in run_rows if r["source"] == source_name]
        alt_r5 = np.asarray([float(r["R@5"]) for r in alt_run_rows], dtype=np.float64)

        for repeat, alt_hit in enumerate(alt_hits):
            cs = cluster_bootstrap_diff(
                main_hit, alt_hit, split.cluster_ids,
                bootstrap_samples, seed + 100000 + source_id * 1000 + repeat,
            )
            sp = cluster_signflip_p(
                main_hit, alt_hit, split.cluster_ids,
                cluster_permutations, seed + 200000 + source_id * 1000 + repeat,
            )
            mc = mcnemar_exact(main_hit, alt_hit)
            dc = dialogue_counts(main_hit, alt_hit, split.cluster_ids)
            per_repeat_rows.append({
                "model": name,
                "reference_source": "main_affective_user",
                "comparison_source": source_name,
                "repeat": repeat,
                "main_R@5": float(main_hit.mean()),
                "comparison_R@5": float(alt_hit.mean()),
                "delta_R@5": float((main_hit - alt_hit).mean()),
                "cluster_ci_low": cs["ci_low"],
                "cluster_ci_high": cs["ci_high"],
                "cluster_bootstrap_p": cs["p_two_sided"],
                "cluster_signflip_p": sp,
                "n_dialogues": cs["n_clusters"],
                "main_only_queries": mc["a_only"],
                "comparison_only_queries": mc["b_only"],
                **dc,
            })

        hs = hierarchical_cluster_bootstrap(
            main_hit, alt_hits, split.cluster_ids,
            bootstrap_samples, seed + 300000 + source_id,
        )
        summary_rows.append({
            "model": name,
            "reference_source": "main_affective_user",
            "comparison_source": source_name,
            "main_R@5": float(main_hit.mean()),
            "comparison_R@5_mean": float(alt_r5.mean()),
            "comparison_R@5_std": float(alt_r5.std(ddof=1)) if len(alt_r5) > 1 else 0.0,
            "delta_R@5_mean": hs["delta_mean_over_repeats"],
            "delta_R@5_std": hs["delta_std_over_repeats"],
            "cluster_source_bootstrap_ci_low": hs["ci_low"],
            "cluster_source_bootstrap_ci_high": hs["ci_high"],
            "cluster_source_bootstrap_p": hs["p_two_sided"],
            "n_dialogues": hs["n_dialogues"],
            "source_repeats": hs["source_repeats"],
        })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_rows, per_repeat_rows, summary_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dialogue-cluster source-ablation inference")
    p.add_argument("--data", required=True)
    p.add_argument("--psydt-path", required=True)
    p.add_argument("--model", action="append", required=True, help="Repeat NAME=PATH")
    p.add_argument("--model-batch-size", action="append", default=[])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--split", choices=["dev", "test"], default="test")
    p.add_argument("--source-size", type=int, default=5000)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-samples", type=int, default=5000)
    p.add_argument("--cluster-permutations", type=int, default=10000)
    p.add_argument(
        "--sources",
        default="main_affective_user,all_train_user,non_affective_user,first_person_non_affective_user,train_counselor",
    )
    p.add_argument("--cache-dir", default="../cache/user_source_cluster")
    p.add_argument("--output-prefix", default="../results/source_cluster/source_cluster")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = load_json(args.data)
    pools_all = build_source_pools(data, args.psydt_path, args.seed)
    requested = [x.strip() for x in args.sources.split(",") if x.strip()]
    unknown = [x for x in requested if x not in pools_all]
    if unknown:
        raise ValueError(f"Unknown sources: {unknown}; available={sorted(pools_all)}")
    pools = {x: pools_all[x] for x in requested}
    split = extract_split(data, args.split)
    if split.cluster_ids is None:
        raise ValueError("Cluster IDs are unavailable in the requested split")
    print(f"[cluster] queries={len(split.queries)} dialogues={len(np.unique(split.cluster_ids))}")

    paths = parse_named_strings(args.model)
    batch_sizes = parse_named_ints(args.model_batch_size)
    cache = EmbeddingCache(args.cache_dir, enabled=not args.no_cache)

    all_runs: List[Dict[str, Any]] = []
    all_repeat: List[Dict[str, Any]] = []
    all_summary: List[Dict[str, Any]] = []
    for model_index, (name, path) in enumerate(paths.items()):
        runs, rep, summ = run_model(
            name=name,
            path=path,
            batch_size=batch_sizes.get(name, args.batch_size),
            device=args.device,
            pools=pools,
            split=split,
            source_size=args.source_size,
            repeats=args.repeats,
            k=args.k,
            cache=cache,
            bootstrap_samples=args.bootstrap_samples,
            cluster_permutations=args.cluster_permutations,
            seed=args.seed + model_index * 10000,
        )
        all_runs.extend(runs)
        all_repeat.extend(rep)
        all_summary.extend(summ)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    write_csv(all_runs, f"{prefix}_runs.csv")
    write_csv(all_repeat, f"{prefix}_per_repeat_cluster.csv")
    write_csv(all_summary, f"{prefix}_cluster_summary.csv")
    save_json({
        "arguments": vars(args),
        "n_queries": len(split.queries),
        "n_dialogues": int(len(np.unique(split.cluster_ids))),
        "source_pool_counts": {k: len(v) for k, v in pools.items()},
        "summary": all_summary,
    }, f"{prefix}_details.json")
    print(f"Saved: {prefix}_cluster_summary.csv")


if __name__ == "__main__":
    main()
