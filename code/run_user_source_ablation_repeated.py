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
    load_json,
    load_model,
    parse_named_ints,
    parse_named_strings,
    retrieve_and_evaluate,
    save_json,
    set_seed,
    stable_dedup,
    write_csv,
)
from run_fixed_k_source_ablation_local import (
    collect_messages,
    is_affective_utterance,
    is_non_affective_first_person,
    load_psydt_dialogs,
    split_dialogs,
)


def build_source_pools(data: Dict[str, Any], psydt_path: str, seed: int) -> Dict[str, List[str]]:
    dialogs = load_psydt_dialogs(psydt_path)
    train, _, _ = split_dialogs(dialogs, seed=seed)
    train_user = collect_messages(train, role="user", max_len=150, min_len=5, keep_chinese_ratio=0.5)
    train_counselor = collect_messages(
        train,
        role="assistant",
        max_len=256,
        min_len=10,
        keep_chinese_ratio=0.5,
    )
    main_source = get_source_texts(data, "emotion_corpus")
    main_set = set(main_source)
    non_affective = [
        text for text in train_user
        if text not in main_set and not is_affective_utterance(text)
    ]
    first_person_non_affective = [text for text in non_affective if is_non_affective_first_person(text)]
    pools = {
        "main_affective_user": stable_dedup(main_source),
        "all_train_user": stable_dedup(train_user),
        "non_affective_user": stable_dedup(non_affective),
        "first_person_non_affective_user": stable_dedup(first_person_non_affective),
        "train_counselor": stable_dedup(train_counselor),
    }
    for name, texts in pools.items():
        if not texts:
            raise ValueError(f"Empty source pool: {name}")
    return pools


def sample_source(pool: List[str], size: int, rng: np.random.Generator, fixed: bool) -> List[str]:
    if fixed and len(pool) == size:
        return list(pool)
    if len(pool) < size:
        raise ValueError(f"Source pool has {len(pool)} items but source_size={size}; sampling with replacement is disabled")
    ids = rng.choice(len(pool), size=size, replace=False)
    return [pool[int(i)] for i in ids]


def run_model(
    name: str,
    path: str,
    batch_size: int,
    device: str,
    pools: Dict[str, List[str]],
    split,
    source_size: int,
    repeats: int,
    k: int,
    cache: EmbeddingCache,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model = load_model(path, device)
    q_raw = cache.encode(model, name, path, f"{split.name}_queries", split.queries, batch_size)
    d_raw = cache.encode(model, name, path, f"{split.name}_corpus", split.corpus, batch_size)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    for source_name, pool in pools.items():
        n_repeats = 1 if source_name == "main_affective_user" else repeats
        for repeat in range(n_repeats):
            texts = sample_source(
                pool, source_size, rng,
                fixed=(source_name == "main_affective_user"),
            )
            source = cache.encode(
                model, name, path, f"source_{source_name}_r{repeat}", texts, batch_size
            )
            basis, values = fit_raw_svd_basis(source, k)
            q = apply_remove_and_normalize(q_raw, basis)
            d = apply_remove_and_normalize(d_raw, basis)
            metrics, _, _, _ = retrieve_and_evaluate(q, d, split.gold, depth=10)
            rows.append({
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

    summary: List[Dict[str, Any]] = []
    for source_name in pools:
        group = [row for row in rows if row["source"] == source_name]
        result: Dict[str, Any] = {
            "model": name,
            "split": split.name,
            "source": source_name,
            "source_pool_n": len(pools[source_name]),
            "source_n": source_size,
            "repeats": len(group),
        }
        for metric in ("R@1", "R@5", "R@10", "MRR", "nDCG@10"):
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary.append(result)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated user-source ablation (paper Table 9)")
    parser.add_argument("--data", default="../data/cleaned/psydt_task_retrieval_final.json")
    parser.add_argument("--psydt-path", required=True, help="Original PsyDT multi-turn JSON")
    parser.add_argument("--model", action="append", required=True, help="Repeat NAME=PATH")
    parser.add_argument("--model-batch-size", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--source-size", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sources",
        default=(
            "main_affective_user,all_train_user,non_affective_user,"
            "first_person_non_affective_user,train_counselor"
        ),
        help=(
            "Comma-separated source names. Available: main_affective_user, "
            "all_train_user, non_affective_user, "
            "first_person_non_affective_user, train_counselor"
        ),
    )
    parser.add_argument("--cache-dir", default="../cache/user_source_ablation")
    parser.add_argument("--output-prefix", default="../results/user_source_ablation/user_source_ablation")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = load_json(args.data)
    pools = build_source_pools(data, args.psydt_path, args.seed)
    requested_sources = [x.strip() for x in args.sources.split(",") if x.strip()]
    unknown_sources = [x for x in requested_sources if x not in pools]
    if unknown_sources:
        raise ValueError(
            f"Unknown sources: {unknown_sources}; available={sorted(pools)}"
        )
    pools = {name: pools[name] for name in requested_sources}
    split = extract_split(data, args.split)
    paths = parse_named_strings(args.model)
    batch_sizes = parse_named_ints(args.model_batch_size)
    cache = EmbeddingCache(args.cache_dir, enabled=not args.no_cache)

    runs: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for index, (name, path) in enumerate(paths.items()):
        r, s = run_model(
            name, path, batch_sizes.get(name, args.batch_size), args.device,
            pools, split, args.source_size, args.repeats, args.k, cache,
            args.seed + index * 10000,
        )
        runs.extend(r)
        summary.extend(s)

    prefix = Path(args.output_prefix)
    write_csv(runs, f"{prefix}_runs.csv")
    write_csv(summary, f"{prefix}_summary.csv")
    save_json({"arguments": vars(args), "source_counts": {k: len(v) for k, v in pools.items()}, "summary": summary}, f"{prefix}_details.json")
    print(f"Saved: {prefix}_summary.csv")


if __name__ == "__main__":
    main()
