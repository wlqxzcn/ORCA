from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from paper_experiment_utils import (
    EmbeddingCache,
    apply_remove_and_normalize,
    cluster_bootstrap_diff,
    cluster_signflip_p,
    fit_raw_svd_basis,
    l2_normalize,
    load_model,
    mcnemar_exact,
    parse_named_ints,
    parse_named_strings,
    retrieve_and_evaluate,
    save_json,
    set_seed,
    stable_dedup,
    write_csv,
)


@dataclass
class Turn:
    transcript_id: str
    mi_quality: str
    turn_id: int
    speaker: str
    text: str


@dataclass
class Pair:
    transcript_id: str
    query_text: str
    response_text: str


def find_column(df: pd.DataFrame, names: Sequence[str]) -> str:
    lowered = {str(col).lower(): str(col) for col in df.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise KeyError(f"None of columns {list(names)} found. Available: {list(df.columns)}")


def canonical_speaker(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"client", "patient", "user", "c"}:
        return "client"
    if text in {"therapist", "counsellor", "counselor", "assistant", "t"}:
        return "therapist"
    raise ValueError(f"Unknown interlocutor value: {value!r}")


def load_and_preprocess(csv_path: str) -> Tuple[List[Turn], List[Pair], Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    transcript_col = find_column(df, ("transcript_id", "conversation_id", "dialogue_id"))
    quality_col = find_column(df, ("mi_quality", "quality"))
    utterance_col = find_column(df, ("utterance_id", "turn_id", "id"))
    speaker_col = find_column(df, ("interlocutor", "speaker", "role"))
    text_col = find_column(df, ("utterance_text", "text", "content"))

    work = df[[transcript_col, quality_col, utterance_col, speaker_col, text_col]].copy()
    work.columns = ["transcript_id", "mi_quality", "utterance_id", "speaker", "text"]
    work["transcript_id"] = work["transcript_id"].astype(str)
    work["mi_quality"] = work["mi_quality"].astype(str).str.strip().str.lower()
    work["speaker"] = work["speaker"].map(canonical_speaker)
    work["text"] = work["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    work = work[work["text"].str.len() > 0].copy()
    work["utterance_order"] = pd.to_numeric(work["utterance_id"], errors="coerce")
    if work["utterance_order"].isna().any():
        # Stable row order fallback within each transcript.
        work["utterance_order"] = work.groupby("transcript_id").cumcount()

    raw_rows = len(work)
    dedup = work.drop_duplicates(
        subset=["transcript_id", "utterance_order", "speaker", "text"], keep="first"
    ).sort_values(["transcript_id", "utterance_order"], kind="stable")

    merged_turns: List[Turn] = []
    for transcript_id, group in dedup.groupby("transcript_id", sort=True):
        quality_values = group["mi_quality"].dropna().astype(str).tolist()
        quality = quality_values[0] if quality_values else "unknown"
        current_speaker: str | None = None
        buffer: List[str] = []
        turn_id = 0
        for row in group.itertuples(index=False):
            speaker = str(row.speaker)
            text = str(row.text).strip()
            if current_speaker is None:
                current_speaker = speaker
                buffer = [text]
            elif speaker == current_speaker:
                buffer.append(text)
            else:
                merged_turns.append(Turn(str(transcript_id), quality, turn_id, current_speaker, " ".join(buffer)))
                turn_id += 1
                current_speaker = speaker
                buffer = [text]
        if current_speaker is not None:
            merged_turns.append(Turn(str(transcript_id), quality, turn_id, current_speaker, " ".join(buffer)))

    pairs: List[Pair] = []
    by_transcript: Dict[str, List[Turn]] = {}
    for turn in merged_turns:
        by_transcript.setdefault(turn.transcript_id, []).append(turn)
    for transcript_id, turns in by_transcript.items():
        turns = sorted(turns, key=lambda item: item.turn_id)
        for current, nxt in zip(turns, turns[1:]):
            if current.speaker == "client" and nxt.speaker == "therapist":
                pairs.append(Pair(transcript_id, current.text, nxt.text))

    stats = {
        "raw_rows": int(raw_rows),
        "deduplicated_utterances": int(len(dedup)),
        "transcripts": int(dedup["transcript_id"].nunique()),
        "merged_turns": int(len(merged_turns)),
        "client_turns": int(sum(turn.speaker == "client" for turn in merged_turns)),
        "therapist_turns": int(sum(turn.speaker == "therapist" for turn in merged_turns)),
        "retrieval_pairs": int(len(pairs)),
        "quality_counts": dedup.groupby("transcript_id")["mi_quality"].first().value_counts().to_dict(),
    }
    if not pairs:
        raise ValueError("No client->therapist pairs were created")
    return merged_turns, pairs, stats


def make_stratified_folds(turns: List[Turn], n_folds: int, seed: int) -> Dict[str, int]:
    quality_by_transcript: Dict[str, str] = {}
    for turn in turns:
        quality_by_transcript[turn.transcript_id] = turn.mi_quality
    grouped: Dict[str, List[str]] = {}
    for transcript_id, quality in quality_by_transcript.items():
        grouped.setdefault(quality, []).append(transcript_id)
    rng = np.random.default_rng(seed)
    assignment: Dict[str, int] = {}
    for quality in sorted(grouped):
        ids = np.asarray(sorted(grouped[quality]), dtype=object)
        rng.shuffle(ids)
        for index, transcript_id in enumerate(ids.tolist()):
            assignment[str(transcript_id)] = index % n_folds
    return assignment


def aggregate_from_per_query(payload: Dict[str, np.ndarray]) -> Dict[str, float]:
    ranks = np.asarray(payload["rank"], dtype=np.int32)
    hit1 = np.asarray(payload["hit1"], dtype=np.float64)
    hit5 = np.asarray(payload["hit5"], dtype=np.float64)
    hit10 = np.asarray(payload["hit10"], dtype=np.float64)
    rr = np.asarray(payload["rr"], dtype=np.float64)
    ndcg10 = np.asarray(payload["ndcg10"], dtype=np.float64)
    return {
        "R@1": float(hit1.mean()),
        "R@5": float(hit5.mean()),
        "R@10": float(hit10.mean()),
        "MRR": float(rr.mean()),
        "nDCG@10": float(ndcg10.mean()),
        "mean_gold_rank": float(ranks.mean()),
    }


def concat_payloads(items: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    keys = items[0].keys()
    return {key: np.concatenate([item[key] for item in items]) for key in keys}


def run_model(
    name: str,
    path: str,
    batch_size: int,
    device: str,
    turns: List[Turn],
    pairs: List[Pair],
    folds: Dict[str, int],
    n_folds: int,
    k: int,
    cache: EmbeddingCache,
    bootstrap_samples: int,
    cluster_permutations: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    model = load_model(path, device)
    fold_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []
    per_method_folds: Dict[str, List[Dict[str, np.ndarray]]] = {}
    cluster_folds: List[np.ndarray] = []

    for fold in range(n_folds):
        test_transcripts = {tid for tid, fold_id in folds.items() if fold_id == fold}
        train_transcripts = set(folds) - test_transcripts
        source_texts = stable_dedup(
            turn.text for turn in turns
            if turn.transcript_id in train_transcripts and turn.speaker == "client"
        )
        fold_pairs = [pair for pair in pairs if pair.transcript_id in test_transcripts]
        if not source_texts or not fold_pairs:
            raise ValueError(f"Fold {fold} has empty source or test pairs")

        candidate_texts = stable_dedup(pair.response_text for pair in fold_pairs)
        candidate_index = {text: index for index, text in enumerate(candidate_texts)}
        query_texts = [pair.query_text for pair in fold_pairs]
        gold = [[candidate_index[pair.response_text]] for pair in fold_pairs]
        clusters = np.asarray([pair.transcript_id for pair in fold_pairs], dtype=object)
        cluster_folds.append(clusters)

        source = cache.encode(model, name, path, f"fold{fold}_source", source_texts, batch_size)
        q_raw = cache.encode(model, name, path, f"fold{fold}_queries", query_texts, batch_size)
        d_raw = cache.encode(model, name, path, f"fold{fold}_candidates", candidate_texts, batch_size)

        basis, eigenvalues = fit_raw_svd_basis(source, max(k, 2))
        mean = source.mean(axis=0)
        mean /= max(float(np.linalg.norm(mean)), 1e-12)
        candidate_basis, _ = fit_raw_svd_basis(d_raw, k)

        methods = {
            "Dense": (l2_normalize(q_raw), l2_normalize(d_raw)),
            "ORCA-v1": (apply_remove_and_normalize(q_raw, basis[:, [0]]), apply_remove_and_normalize(d_raw, basis[:, [0]])),
            "ORCA-v2": (apply_remove_and_normalize(q_raw, basis[:, [1]]), apply_remove_and_normalize(d_raw, basis[:, [1]])),
            f"ORCA-k{k}": (apply_remove_and_normalize(q_raw, basis[:, :k]), apply_remove_and_normalize(d_raw, basis[:, :k])),
            "User-Mean-Direction": (apply_remove_and_normalize(q_raw, mean[:, None]), apply_remove_and_normalize(d_raw, mean[:, None])),
            f"Candidate-Raw-SVD-k{k}": (apply_remove_and_normalize(q_raw, candidate_basis), apply_remove_and_normalize(d_raw, candidate_basis)),
        }

        for method, (q, d) in methods.items():
            metrics, payload, _, _ = retrieve_and_evaluate(q, d, gold, depth=len(candidate_texts))
            fold_rows.append({
                "model": name,
                "fold": fold,
                "method": method,
                "train_transcripts": len(train_transcripts),
                "test_transcripts": len(test_transcripts),
                "source_n": len(source_texts),
                "query_n": len(query_texts),
                "candidate_n": len(candidate_texts),
                **metrics,
            })
            per_method_folds.setdefault(method, []).append(payload)

        geometry_rows.append({
            "model": name,
            "fold": fold,
            "source_n": len(source_texts),
            "abs_cos_v1_user_mean": float(abs(np.dot(basis[:, 0], mean))),
            "abs_cos_v2_user_mean": float(abs(np.dot(basis[:, 1], mean))),
            "selected_energy_ratio": float(eigenvalues[:k].sum() / max(eigenvalues.sum(), 1e-15)),
        })

    clusters = np.concatenate(cluster_folds)
    aggregate_rows: List[Dict[str, Any]] = []
    aggregate_payloads: Dict[str, Dict[str, np.ndarray]] = {}
    for method, fold_payloads in per_method_folds.items():
        payload = concat_payloads(fold_payloads)
        aggregate_payloads[method] = payload
        aggregate_rows.append({"model": name, "method": method, **aggregate_from_per_query(payload)})

    orca_name = f"ORCA-k{k}"
    dense_payload = aggregate_payloads["Dense"]
    orca_payload = aggregate_payloads[orca_name]
    cluster_stats = cluster_bootstrap_diff(
        orca_payload["hit5"], dense_payload["hit5"], clusters,
        bootstrap_samples, seed + 100,
    )
    mc = mcnemar_exact(orca_payload["hit5"], dense_payload["hit5"])
    paired = {
        "model": name,
        "method_a": orca_name,
        "method_b": "Dense",
        "delta_R@5": cluster_stats["delta"],
        "cluster_ci_low": cluster_stats["ci_low"],
        "cluster_ci_high": cluster_stats["ci_high"],
        "cluster_bootstrap_p": cluster_stats["p_two_sided"],
        "cluster_permutation_p": cluster_signflip_p(
            orca_payload["hit5"], dense_payload["hit5"], clusters,
            cluster_permutations, seed + 200,
        ),
        "orca_only": mc["a_only"],
        "dense_only": mc["b_only"],
        "mcnemar_p": mc["p_exact"],
        "n_dialogues": int(len(np.unique(clusters))),
        "n_queries": int(len(clusters)),
    }

    details = {
        "model": name,
        "fold_assignment": folds,
        "geometry_mean": {
            "abs_cos_v1_user_mean": float(np.mean([row["abs_cos_v1_user_mean"] for row in geometry_rows])),
            "abs_cos_v2_user_mean": float(np.mean([row["abs_cos_v2_user_mean"] for row in geometry_rows])),
        },
        "paired": paired,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fold_rows, aggregate_rows, geometry_rows, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnnoMI five-fold ORCA validation (paper Tables 15-17)")
    parser.add_argument("--csv", required=True, help="Path to AnnoMI-full.csv")
    parser.add_argument("--model", action="append", required=True, help="Repeat NAME=PATH, e.g. mpnet=... and bge_en=...")
    parser.add_argument("--model-batch-size", action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--cluster-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="../cache/annomi_fivefold")
    parser.add_argument("--output-prefix", default="../results/annomi/annomi_fivefold")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    turns, pairs, preprocessing = load_and_preprocess(args.csv)
    folds = make_stratified_folds(turns, args.folds, args.seed)
    paths = parse_named_strings(args.model)
    batch_sizes = parse_named_ints(args.model_batch_size)
    cache = EmbeddingCache(args.cache_dir, enabled=not args.no_cache)

    fold_rows: List[Dict[str, Any]] = []
    aggregate_rows: List[Dict[str, Any]] = []
    geometry_rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    paired_rows: List[Dict[str, Any]] = []
    for index, (name, path) in enumerate(paths.items()):
        f, a, g, d = run_model(
            name, path, batch_sizes.get(name, args.batch_size), args.device,
            turns, pairs, folds, args.folds, args.k, cache,
            args.bootstrap_samples, args.cluster_permutations,
            args.seed + index * 10000,
        )
        fold_rows.extend(f)
        aggregate_rows.extend(a)
        geometry_rows.extend(g)
        details[name] = d
        paired_rows.append(d["paired"])

    prefix = Path(args.output_prefix)
    write_csv(fold_rows, f"{prefix}_fold_metrics.csv")
    write_csv(aggregate_rows, f"{prefix}_aggregate_metrics.csv")
    write_csv(paired_rows, f"{prefix}_paired_significance.csv")
    write_csv(geometry_rows, f"{prefix}_geometry.csv")
    save_json({
        "arguments": vars(args),
        "preprocessing": preprocessing,
        "fold_assignment": folds,
        "models": details,
        "aggregate_metrics": aggregate_rows,
    }, f"{prefix}_details.json")
    print(json.dumps(preprocessing, ensure_ascii=False, indent=2))
    print(f"Saved: {prefix}_aggregate_metrics.csv")


if __name__ == "__main__":
    main()
