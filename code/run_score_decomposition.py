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
    fit_raw_svd_basis,
    get_source_texts,
    l2_normalize,
    load_json,
    load_model,
    parse_named_ints,
    parse_named_strings,
    project_remove,
    retrieve_and_evaluate,
    save_json,
    search_ip,
    set_seed,
    write_csv,
)


def permutation_group_p(a: np.ndarray, b: np.ndarray, permutations: int, seed: int) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    observed = abs(float(a.mean() - b.mean()))
    joined = np.concatenate([a, b])
    n_a = len(a)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        shuffled = rng.permutation(joined)
        value = abs(float(shuffled[:n_a].mean() - shuffled[n_a:].mean()))
        count += int(value >= observed - 1e-15)
    return float((count + 1) / (permutations + 1))


def run_model(
    name: str,
    path: str,
    batch_size: int,
    device: str,
    source_texts: List[str],
    split,
    k: int,
    group_k: int,
    ranking_depth: int,
    permutations: int,
    cache: EmbeddingCache,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    model = load_model(path, device)
    source = cache.encode(model, name, path, "source", source_texts, batch_size)
    q_raw = cache.encode(model, name, path, f"{split.name}_queries", split.queries, batch_size)
    d_raw = cache.encode(model, name, path, f"{split.name}_corpus", split.corpus, batch_size)

    basis, eigenvalues = fit_raw_svd_basis(source, k)
    q_dense = l2_normalize(q_raw)
    d_dense = l2_normalize(d_raw)
    q_orca = apply_remove_and_normalize(q_raw, basis)
    d_orca = apply_remove_and_normalize(d_raw, basis)

    depth = min(max(ranking_depth, 10), len(split.corpus))
    dense_metrics, dense_pq, dense_scores, dense_ids = retrieve_and_evaluate(q_dense, d_dense, split.gold, depth)
    orca_metrics, orca_pq, orca_scores, orca_ids = retrieve_and_evaluate(q_orca, d_orca, split.gold, depth)

    q_retained = project_remove(q_dense, basis)
    d_retained = project_remove(d_dense, basis)
    q_removed = q_dense @ basis
    d_removed = d_dense @ basis

    rows: List[Dict[str, Any]] = []
    for qid in range(len(split.queries)):
        gold_set = {int(x) for x in split.gold[qid]}
        gold_idx = int(split.gold[qid][0])
        hard_negative = None
        for candidate_id in dense_ids[qid]:
            if int(candidate_id) not in gold_set:
                hard_negative = int(candidate_id)
                break
        if hard_negative is None:
            raise RuntimeError(f"No hard negative for query {qid}")

        dense_hit = bool(dense_pq["hit5"][qid])
        orca_hit = bool(orca_pq["hit5"][qid])
        if not dense_hit and orca_hit:
            group = "rescued"
        elif dense_hit and not orca_hit:
            group = "harmed"
        else:
            group = "unchanged"

        rem_gold = float(q_removed[qid] @ d_removed[gold_idx])
        rem_neg = float(q_removed[qid] @ d_removed[hard_negative])
        keep_gold = float(q_retained[qid] @ d_retained[gold_idx])
        keep_neg = float(q_retained[qid] @ d_retained[hard_negative])
        dense_gold = float(q_dense[qid] @ d_dense[gold_idx])
        dense_neg = float(q_dense[qid] @ d_dense[hard_negative])
        orca_gold = float(q_orca[qid] @ d_orca[gold_idx])
        orca_neg = float(q_orca[qid] @ d_orca[hard_negative])

        rows.append({
            "model": name,
            "split": split.name,
            "query_index": qid,
            "group": group,
            "gold_index": gold_idx,
            "hard_negative_index": hard_negative,
            "dense_hit": dense_hit,
            "orca_hit": orca_hit,
            "dense_gold_rank": int(dense_pq["rank"][qid]),
            "orca_gold_rank": int(orca_pq["rank"][qid]),
            "gold_rank_change": int(dense_pq["rank"][qid] - orca_pq["rank"][qid]),
            "removed_gold_score": rem_gold,
            "removed_negative_score": rem_neg,
            "removed_wrong_advantage": rem_neg - rem_gold,
            "retained_gold_score": keep_gold,
            "retained_negative_score": keep_neg,
            "retained_gold_margin": keep_gold - keep_neg,
            "dense_gold_margin": dense_gold - dense_neg,
            "orca_gold_margin": orca_gold - orca_neg,
        })

    summary: List[Dict[str, Any]] = []
    for group in ("rescued", "harmed", "unchanged"):
        group_rows = [row for row in rows if row["group"] == group]
        if not group_rows:
            continue
        def values(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in group_rows], dtype=float)
        summary.append({
            "model": name,
            "split": split.name,
            "group": group,
            "n": len(group_rows),
            "removed_wrong_advantage_mean": float(values("removed_wrong_advantage").mean()),
            "removed_wrong_advantage_median": float(np.median(values("removed_wrong_advantage"))),
            "removed_wrong_advantage_positive_rate": float((values("removed_wrong_advantage") > 0).mean()),
            "retained_gold_margin_mean": float(values("retained_gold_margin").mean()),
            "dense_gold_margin_mean": float(values("dense_gold_margin").mean()),
            "orca_gold_margin_mean": float(values("orca_gold_margin").mean()),
            "gold_rank_change_mean": float(values("gold_rank_change").mean()),
        })

    rescued = np.asarray([row["removed_wrong_advantage"] for row in rows if row["group"] == "rescued"], dtype=float)
    harmed = np.asarray([row["removed_wrong_advantage"] for row in rows if row["group"] == "harmed"], dtype=float)
    group_test = None
    if len(rescued) and len(harmed):
        group_test = {
            "rescued_minus_harmed_mean": float(rescued.mean() - harmed.mean()),
            "permutation_p": permutation_group_p(rescued, harmed, permutations, seed),
            "permutations": permutations,
        }

    manifest = {
        "model": name,
        "split": split.name,
        "k": k,
        "group_k": group_k,
        "ranking_depth": depth,
        "dense": dense_metrics,
        "orca": orca_metrics,
        "basis": {
            "source_n": len(source_texts),
            "dimension": int(source.shape[1]),
            "top_eigenvalues": [float(x) for x in eigenvalues[:20]],
            "selected_energy_ratio": float(eigenvalues[:k].sum() / max(eigenvalues.sum(), 1e-15)),
        },
        "group_test": group_test,
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, summary, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORCA score decomposition (paper Table 11)")
    parser.add_argument("--data", default="../data/cleaned/psydt_task_retrieval_final.json")
    parser.add_argument("--model", action="append", required=True, help="Repeat NAME=PATH")
    parser.add_argument("--model-batch-size", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["dev", "test", "legacy_test"], default="test")
    parser.add_argument("--source-key", default="emotion_corpus")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--group-k", type=int, default=5)
    parser.add_argument("--ranking-depth", type=int, default=101)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="../cache/score_decomposition")
    parser.add_argument("--output-prefix", default="../results/score_decomposition/score_decomposition")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = load_json(args.data)
    source_texts = get_source_texts(data, args.source_key)
    split = extract_split(data, args.split)
    paths = parse_named_strings(args.model)
    batch_sizes = parse_named_ints(args.model_batch_size)
    cache = EmbeddingCache(args.cache_dir, enabled=not args.no_cache)

    all_rows: List[Dict[str, Any]] = []
    all_summary: List[Dict[str, Any]] = []
    manifests: Dict[str, Any] = {}
    for index, (name, path) in enumerate(paths.items()):
        rows, summary, manifest = run_model(
            name, path, batch_sizes.get(name, args.batch_size), args.device,
            source_texts, split, args.k, args.group_k, args.ranking_depth,
            args.permutations, cache, args.seed + index * 10000,
        )
        all_rows.extend(rows)
        all_summary.extend(summary)
        manifests[name] = manifest

    prefix = Path(args.output_prefix)
    write_csv(all_rows, f"{prefix}_per_query.csv")
    write_csv(all_summary, f"{prefix}_groups.csv")
    save_json({"arguments": vars(args), "models": manifests, "summary": all_summary}, f"{prefix}_manifest.json")
    print(f"Saved: {prefix}_groups.csv")


if __name__ == "__main__":
    main()
