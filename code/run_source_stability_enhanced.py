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
    projection_distance,
    retrieve_and_evaluate,
    save_json,
    set_seed,
    vector_angle_degrees,
    write_csv,
)


def parse_sizes(value: str) -> List[int]:
    sizes = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not sizes or any(x < 2 for x in sizes):
        raise ValueError("All source sizes must be >=2")
    return sizes


def run_model(
    name: str,
    path: str,
    batch_size: int,
    device: str,
    source_texts: List[str],
    split,
    k: int,
    sizes: List[int],
    repeats: int,
    cache: EmbeddingCache,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    model = load_model(path, device)
    source = cache.encode(model, name, path, "source", source_texts, batch_size)
    q = cache.encode(model, name, path, f"{split.name}_queries", split.queries, batch_size)
    d = cache.encode(model, name, path, f"{split.name}_corpus", split.corpus, batch_size)

    full_basis, full_values = fit_raw_svd_basis(source, k)
    full_mean = source.mean(axis=0)
    full_mean /= max(float(np.linalg.norm(full_mean)), 1e-12)
    rng = np.random.default_rng(seed)

    run_rows: List[Dict[str, Any]] = []
    for size in sizes:
        if size > len(source):
            raise ValueError(f"size={size} exceeds source_n={len(source)}")
        n_repeats = 1 if size == len(source) else repeats
        for repeat in range(n_repeats):
            if size == len(source):
                indices = np.arange(len(source))
            else:
                indices = rng.choice(len(source), size=size, replace=False)
            sample = source[indices]
            basis, values = fit_raw_svd_basis(sample, k)
            q_proj = apply_remove_and_normalize(q, basis)
            d_proj = apply_remove_and_normalize(d, basis)
            metrics, _, _, _ = retrieve_and_evaluate(q_proj, d_proj, split.gold, depth=10)
            sample_mean = sample.mean(axis=0)
            sample_mean /= max(float(np.linalg.norm(sample_mean)), 1e-12)
            run_rows.append({
                "model": name,
                "split": split.name,
                "source_size": size,
                "repeat": repeat,
                **metrics,
                "v1_angle_deg": vector_angle_degrees(basis[:, 0], full_basis[:, 0]),
                "v2_angle_deg": vector_angle_degrees(basis[:, 1], full_basis[:, 1]) if k >= 2 else None,
                "projection_distance": projection_distance(basis, full_basis),
                "abs_cos_v1_sample_mean": float(abs(np.dot(basis[:, 0], sample_mean))),
                "abs_cos_v1_full_mean": float(abs(np.dot(basis[:, 0], full_mean))),
                "selected_energy_ratio": float(values[:k].sum() / max(values.sum(), 1e-15)),
            })

    summary_rows: List[Dict[str, Any]] = []
    for size in sizes:
        rows = [row for row in run_rows if row["source_size"] == size]
        summary: Dict[str, Any] = {
            "model": name,
            "split": split.name,
            "source_size": size,
            "repeats": len(rows),
        }
        for key in ("R@1", "R@5", "R@10", "MRR", "nDCG@10", "v1_angle_deg", "v2_angle_deg", "projection_distance", "abs_cos_v1_sample_mean"):
            values = np.asarray([float(row[key]) for row in rows if row[key] is not None], dtype=float)
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    details = {
        "model": name,
        "full_source_n": len(source),
        "dimension": int(source.shape[1]),
        "full_top_eigenvalues": [float(x) for x in full_values[:20]],
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return run_rows, summary_rows, details


def make_plots(summary_rows: List[Dict[str, Any]], output_prefix: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warning] matplotlib unavailable; CSV/JSON outputs were still written")
        return
    prefix = Path(output_prefix)
    for metric, ylabel in (("R@5_mean", "R@5"), ("v1_angle_deg_mean", "v1 angle (degrees)"), ("v2_angle_deg_mean", "v2 angle (degrees)")):
        plt.figure()
        for model in sorted({row["model"] for row in summary_rows}):
            rows = sorted([row for row in summary_rows if row["model"] == model], key=lambda r: r["source_size"])
            plt.errorbar(
                [row["source_size"] for row in rows],
                [row[metric] for row in rows],
                yerr=[row.get(metric.replace("_mean", "_std"), 0.0) for row in rows],
                marker="o",
                label=model,
            )
        plt.xlabel("Source size")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        path = prefix.parent / f"{prefix.name}_{metric}.png"
        plt.savefig(path, dpi=180)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Source-size stability (paper Table 13)")
    parser.add_argument("--data", default="../data/cleaned/psydt_task_retrieval_final.json")
    parser.add_argument("--model", action="append", required=True, help="Repeat NAME=PATH")
    parser.add_argument("--model-batch-size", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--source-key", default="emotion_corpus")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--sizes", default="250,500,1000,2000,4000,5000")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="../cache/source_stability")
    parser.add_argument("--output-prefix", default="../results/source_stability/source_stability")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = load_json(args.data)
    source = get_source_texts(data, args.source_key)
    split = extract_split(data, args.split)
    sizes = parse_sizes(args.sizes)
    paths = parse_named_strings(args.model)
    batch_sizes = parse_named_ints(args.model_batch_size)
    cache = EmbeddingCache(args.cache_dir, enabled=not args.no_cache)

    runs: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    for index, (name, path) in enumerate(paths.items()):
        r, s, d = run_model(
            name, path, batch_sizes.get(name, args.batch_size), args.device,
            source, split, args.k, sizes, args.repeats, cache,
            args.seed + index * 10000,
        )
        runs.extend(r)
        summary.extend(s)
        details[name] = d

    prefix = Path(args.output_prefix)
    write_csv(runs, f"{prefix}_runs.csv")
    write_csv(summary, f"{prefix}_summary.csv")
    save_json({"arguments": vars(args), "models": details, "summary": summary}, f"{prefix}_details.json")
    make_plots(summary, str(prefix))
    print(f"Saved: {prefix}_summary.csv")


if __name__ == "__main__":
    main()
