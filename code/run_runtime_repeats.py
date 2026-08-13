from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd


TIMING_COLS = [
    "orca_svd_sec",
    "orca_project_candidates_sec",
    "orca_project_queries_sec",
    "dense_search_sec",
    "orca_search_sec",
]

OPTIONAL_TIMING_COLS = [
    "dense_index_sec",
    "orca_index_sec",
    "offline_orca_extra_sec",
    "online_orca_query_project_sec",
    "query_encode_sec",
    "source_encode_sec",
    "candidate_encode_sec",
]

DISPLAY_NAMES = {
    "orca_svd_sec": "SVD",
    "orca_project_candidates_sec": "Cand. Proj.",
    "orca_project_queries_sec": "Query Proj.",
    "dense_search_sec": "Dense Search",
    "orca_search_sec": "ORCA Search",
    "dense_index_sec": "Dense Index",
    "orca_index_sec": "ORCA Index",
    "offline_orca_extra_sec": "Offline ORCA Extra",
    "online_orca_query_project_sec": "Online Query Proj.",
    "query_encode_sec": "Query Encode/Load",
    "source_encode_sec": "Source Encode/Load",
    "candidate_encode_sec": "Candidate Encode/Load",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--script", type=str, default="./orca_bigdata_final_safe.py")
    p.add_argument("--tag", type=str, required=True, help="Short name used in output filenames, e.g., gte or bge")
    p.add_argument("--test_pairs_file", type=str, required=True)
    p.add_argument("--train_responses_file", type=str, required=True)
    p.add_argument("--train_user_file", type=str, required=True)
    p.add_argument("--encoder_name", type=str, required=True)
    p.add_argument("--pool_sizes", nargs="+", default=["5000", "10000", "20000", "50000", "full"])
    p.add_argument("--orca_k", type=int, default=2)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max_queries", type=int, default=2000)
    p.add_argument("--source_limit", type=int, default=5000)
    p.add_argument("--source_mode", type=str, default="affective_user")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--cache_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--runs", type=int, default=5, help="Counted runs used for mean/std")
    p.add_argument("--warmup_runs", type=int, default=1, help="Warm-up runs discarded from summary")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--python", type=str, default=sys.executable)
    return p.parse_args()


def run_one(args: argparse.Namespace, run_id: int, warmup: bool) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"warmup{run_id}" if warmup else f"run{run_id}"
    out_scaling = out_dir / f"scaling_{args.tag}_{prefix}.csv"
    out_runtime = out_dir / f"runtime_{args.tag}_{prefix}.csv"
    out_metadata = out_dir / f"metadata_{args.tag}_{prefix}.json"

    cmd = [
        args.python,
        args.script,
        "--test_pairs_file", args.test_pairs_file,
        "--train_responses_file", args.train_responses_file,
        "--train_user_file", args.train_user_file,
        "--encoder_name", args.encoder_name,
        "--pool_sizes", *[str(x) for x in args.pool_sizes],
        "--orca_k", str(args.orca_k),
        "--topk", str(args.topk),
        "--max_queries", str(args.max_queries),
        "--source_limit", str(args.source_limit),
        "--source_mode", args.source_mode,
        "--batch_size", str(args.batch_size),
        "--cache_dir", args.cache_dir,
        "--out_scaling", str(out_scaling),
        "--out_runtime", str(out_runtime),
        "--out_metadata", str(out_metadata),
        "--seed", str(args.seed),
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    print("\n" + "=" * 80)
    print(f"[start] {args.tag} {prefix}")
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    print(f"[done] {args.tag} {prefix}: {out_runtime}")
    return out_runtime


def summarize(runtime_files: List[Path], out_dir: Path, tag: str) -> None:
    frames = []
    for i, path in enumerate(runtime_files, start=1):
        df = pd.read_csv(path)
        df["run"] = i
        df["runtime_file"] = str(path)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_path = out_dir / f"runtime_{tag}_all_counted_runs.csv"
    all_df.to_csv(all_path, index=False)

    available_cols = [c for c in TIMING_COLS + OPTIONAL_TIMING_COLS if c in all_df.columns]
    group_cols = ["encoder", "pool_label", "pool_n", "query_n", "source_n"]
    group_cols = [c for c in group_cols if c in all_df.columns]

    rows: List[Dict[str, object]] = []
    for keys, g in all_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, object] = dict(zip(group_cols, keys))
        row["n_runs"] = int(g["run"].nunique())
        for col in available_cols:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_std"] = float(g[col].std(ddof=1)) if row["n_runs"] > 1 else 0.0
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = out_dir / f"runtime_{tag}_mean_std_all_pools.csv"
    summary_df.to_csv(summary_path, index=False)

    full_df = summary_df[summary_df["pool_label"].astype(str).eq("full")].copy()
    if full_df.empty:
        raise RuntimeError("No pool_label == 'full' row found in runtime files.")
    full_path = out_dir / f"runtime_{tag}_mean_std_full.csv"
    full_df.to_csv(full_path, index=False)

    def fmt(mean: float, std: float) -> str:
        return f"{mean:.4f} $\\pm$ {std:.4f}s"

    r = full_df.iloc[0]
    cells = [str(tag)]
    for col in TIMING_COLS:
        cells.append(fmt(float(r[f"{col}_mean"]), float(r[f"{col}_std"])))
    latex_row = " & ".join(cells) + r" \\" + "\n"
    latex_path = out_dir / f"runtime_{tag}_latex_row.txt"
    latex_path.write_text(latex_row, encoding="utf-8")

    print("\n" + "=" * 80)
    print("[summary written]")
    print(all_path)
    print(summary_path)
    print(full_path)
    print(latex_path)
    print("\n[LaTeX row]")
    print(latex_row)


def main() -> None:
    args = parse_args()
    if args.runs <= 1:
        raise ValueError("Use --runs >= 2 to compute a standard deviation.")

    counted_files: List[Path] = []

    for w in range(1, args.warmup_runs + 1):
        run_one(args, w, warmup=True)

    for r in range(1, args.runs + 1):
        counted_files.append(run_one(args, r, warmup=False))

    summarize(counted_files, Path(args.out_dir), args.tag)


if __name__ == "__main__":
    main()
