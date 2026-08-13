from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


QUERY_KEYS = {
    "dev": ["dev_queries", "valid_queries", "validation_queries"],
    "test": ["test_queries"],
}
CORPUS_KEYS = {
    "dev": ["dev_task_corpus", "dev_corpus", "valid_task_corpus", "validation_task_corpus"],
    "test": ["test_task_corpus", "test_corpus"],
}
GT_KEYS = {
    "dev": ["dev_gt", "dev_gold", "dev_labels", "valid_gt", "validation_gt"],
    "test": ["test_gt", "test_gold", "test_labels"],
}
SOURCE_KEYS = [
    "emotion_corpus",
    "affective_corpus",
    "source_corpus",
    "train_user_source",
    "train_user_messages",
]


from orca_source_lexicons import (
    FIRST_PERSON_WORDS,
    EMOTION_WORDS,
    PHYSICAL_MEDICAL_WORDS,
    TOPIC_ENTITY_WORDS,
)



def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sha1_of_texts(texts: Sequence[str]) -> str:
    h = hashlib.sha1()
    for x in texts:
        h.update(str(x).encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(Path(path).parent)
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def first_existing(d: Dict[str, Any], keys: Sequence[str], required_name: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(
        f"Cannot find {required_name}. Tried {keys}. Existing keys: {list(d.keys())[:80]}"
    )


def normalize_gt(gt: Sequence[Any], corpus: Sequence[str]) -> List[int]:
    text_to_first_idx: Dict[str, int] = {}
    for i, t in enumerate(corpus):
        if t not in text_to_first_idx:
            text_to_first_idx[t] = i

    def resolve_one(x: Any) -> int:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, str):
            xs = x.strip()
            if xs.isdigit() or (xs.startswith("-") and xs[1:].isdigit()):
                return int(xs)
            if xs not in text_to_first_idx:
                raise ValueError(f"Gold text not found in corpus: {xs[:120]!r}")
            return text_to_first_idx[xs]
        raise TypeError(f"Unsupported gold type: {type(x)}")

    out: List[int] = []
    for g in gt:
        if isinstance(g, (list, tuple)):
            if not g:
                raise ValueError("Empty gold list.")
            idx = resolve_one(g[0])
        else:
            idx = resolve_one(g)
        if idx < 0 or idx >= len(corpus):
            raise ValueError(f"Gold index out of range: {idx}; corpus size={len(corpus)}")
        out.append(idx)
    return out


def load_split_and_source(benchmark_json: str | Path, split: str) -> Tuple[List[str], List[str], List[int], List[str], Dict[str, Any]]:
    obj = read_json(benchmark_json)
    if not isinstance(obj, dict):
        raise ValueError("Benchmark JSON must be a dictionary.")

    queries = [normalize_text(x) for x in first_existing(obj, QUERY_KEYS[split], f"{split} queries")]
    corpus = [normalize_text(x) for x in first_existing(obj, CORPUS_KEYS[split], f"{split} corpus")]
    gt_raw = list(first_existing(obj, GT_KEYS[split], f"{split} gt"))
    gt = normalize_gt(gt_raw, corpus)

    source = None
    source_key = None
    for k in SOURCE_KEYS:
        if k in obj:
            source = [normalize_text(x) for x in obj[k]]
            source_key = k
            break
    if source is None:
        raise KeyError(f"Cannot find ORCA source corpus. Tried {SOURCE_KEYS}")

    if len(queries) != len(gt):
        raise ValueError(f"{split}: query/gold length mismatch: {len(queries)} vs {len(gt)}")

    meta = {
        "benchmark_json": str(benchmark_json),
        "split": split,
        "query_n": len(queries),
        "corpus_n": len(corpus),
        "source_key": source_key,
        "source_n": len(source),
        "source_hash": sha1_of_texts(source),
        "query_hash": sha1_of_texts(queries),
        "corpus_hash": sha1_of_texts(corpus),
    }
    return queries, corpus, gt, source, meta


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype("float32")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


class Encoder:
    def __init__(self, model_name_or_path: str, batch_size: int = 64, device: Optional[str] = None):
        self.model_name_or_path = model_name_or_path
        self.batch_size = batch_size
        self.device = device
        self.kind = ""
        self.model = None
        self.tokenizer = None
        self._load()

    def _load(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            self.model = SentenceTransformer(self.model_name_or_path, **kwargs)
            self.kind = "sentence_transformers"
            return
        except Exception as e:
            print(f"[warn] SentenceTransformer load failed; fallback to transformers: {e}", file=sys.stderr)

        import torch
        from transformers import AutoModel, AutoTokenizer
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_name_or_path, local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.kind = "transformers_mean_pool"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [normalize_text(t) for t in texts]
        if self.kind == "sentence_transformers":
            arr = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return arr.astype("float32")

        import torch
        arrs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                tok = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                tok = {k: v.to(self.device) for k, v in tok.items()}
                out = self.model(**tok)
                last = out.last_hidden_state
                mask = tok["attention_mask"].unsqueeze(-1).float()
                pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                arrs.append(pooled.detach().cpu().numpy().astype("float32"))
        return np.vstack(arrs).astype("float32")


def cached_encode(
    encoder: Encoder,
    texts: Sequence[str],
    cache_dir: str | Path,
    name: str,
    normalize: bool = True,
    no_cache: bool = False,
) -> np.ndarray:
    ensure_dir(cache_dir)
    text_hash = sha1_of_texts(texts)
    model_hash = hashlib.sha1(encoder.model_name_or_path.encode("utf-8")).hexdigest()[:10]
    cache_path = Path(cache_dir) / f"{name}.{model_hash}.{len(texts)}.{text_hash}.npy"
    if (not no_cache) and cache_path.exists():
        arr = np.load(cache_path)
    else:
        print(f"[encode] {name}: n={len(texts)}")
        arr = encoder.encode(texts)
        if not no_cache:
            np.save(cache_path, arr)
    if normalize:
        arr = l2_normalize(arr.astype("float32"))
    return arr.astype("float32")


def fit_orca_basis(source_emb: np.ndarray, k: int) -> np.ndarray:
    _, _, vt = np.linalg.svd(source_emb.astype("float64"), full_matrices=False)
    return vt[:k].T.astype("float32")


def project_remove(x: np.ndarray, basis: np.ndarray, renorm: bool = True) -> np.ndarray:
    y = x - (x @ basis) @ basis.T
    y = y.astype("float32")
    return l2_normalize(y) if renorm else y


def topk_from_scores(scores: np.ndarray, topk: int) -> np.ndarray:
    if topk >= scores.shape[1]:
        return np.argsort(-scores, axis=1)[:, :topk]
    part = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1)
    return np.take_along_axis(part, order, axis=1)


def ranks_from_scores(scores: np.ndarray, gt: Sequence[int]) -> List[int]:
    ranks: List[int] = []
    for i, g in enumerate(gt):
        g = int(g)
        rank = int(np.sum(scores[i] > scores[i, g])) + 1
        ranks.append(rank)
    return ranks


def contains_count(text: str, words: Iterable[str]) -> int:
    return sum(1 for w in words if w and w in text)


def make_diag(text: str) -> Dict[str, int]:
    return {
        "first_person_hits": contains_count(text, FIRST_PERSON_WORDS),
        "emotion_hits": contains_count(text, EMOTION_WORDS),
        "physical_medical_hits": contains_count(text, PHYSICAL_MEDICAL_WORDS),
        "topic_entity_hits": contains_count(text, TOPIC_ENTITY_WORDS),
        "char_len": len(text),
    }


def shorten(text: str, max_chars: int) -> str:
    s = normalize_text(text)
    if len(s) <= max_chars:
        return s
    return s[:max_chars - 1] + "…"


def latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def category_for_case(dense_rank: int, orca_rank: int, dense_top1: int, orca_top1: int, gold: int) -> str:
    if dense_top1 != gold and orca_top1 == gold:
        return "orca_top1_gold_dense_not"
    if dense_rank > 5 and orca_rank <= 5:
        return "orca_top5_dense_miss"
    if dense_rank > 10 and orca_rank <= 10:
        return "orca_top10_dense_miss"
    if dense_rank - orca_rank >= 20 and orca_rank <= 10:
        return "large_rank_gain"
    if dense_rank > orca_rank:
        return "rank_improved"
    return "other"


def select_cases(rows: List[Dict[str, Any]], max_cases: int, seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    priority = [
        "orca_top1_gold_dense_not",
        "orca_top5_dense_miss",
        "orca_top10_dense_miss",
        "large_rank_gain",
        "rank_improved",
    ]
    chosen: List[Dict[str, Any]] = []
    used_qids = set()
    # Prefer cases with larger rank improvement inside each category.
    for cat in priority:
        candidates = [r for r in rows if r["case_category"] == cat and r["query_id"] not in used_qids]
        candidates = sorted(
            candidates,
            key=lambda r: (
                -int(r["rank_gain"]),
                int(r["orca_gold_rank"]),
                int(r["dense_gold_rank"]),
            ),
        )

        for r in candidates[:3]:
            if len(chosen) >= max_cases:
                break
            chosen.append(r)
            used_qids.add(r["query_id"])
        if len(chosen) >= max_cases:
            break

    if len(chosen) < max_cases:
        remaining = [r for r in rows if r["query_id"] not in used_qids and r["case_category"] != "other"]
        remaining = sorted(remaining, key=lambda r: (-int(r["rank_gain"]), int(r["orca_gold_rank"])))
        for r in remaining:
            if len(chosen) >= max_cases:
                break
            chosen.append(r)
            used_qids.add(r["query_id"])

    return chosen[:max_cases]


def save_markdown(path: str | Path, rows: List[Dict[str, Any]], max_chars: int) -> None:
    lines = []
    lines.append("# Dense vs ORCA Qualitative Cases")
    lines.append("")
    lines.append("Use this file for internal inspection. Paraphrase/anonymize examples before inserting into the paper.")
    lines.append("")
    lines.append("| ID | Category | Dense rank | ORCA rank | Query | Dense top-1 | ORCA top-1 | Pseudo-gold |")
    lines.append("|---:|---|---:|---:|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['query_id']} | {r['case_category']} | {r['dense_gold_rank']} | {r['orca_gold_rank']} | "
            f"{shorten(r['query'], max_chars)} | {shorten(r['dense_top1'], max_chars)} | "
            f"{shorten(r['orca_top1'], max_chars)} | {shorten(r['pseudo_gold'], max_chars)} |"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def save_latex_template(path: str | Path, rows: List[Dict[str, Any]], max_chars: int) -> None:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{p{0.20\textwidth}p{0.22\textwidth}p{0.22\textwidth}p{0.22\textwidth}p{0.08\textwidth}}")
    lines.append(r"\toprule")
    lines.append(r"User query & Dense top-1 & ORCA top-1 & Pseudo-gold response & Rank change \\")
    lines.append(r"\midrule")
    for r in rows:
        # For camera-ready, replace these raw snippets with paraphrased English summaries.
        q = latex_escape(shorten(r["query"], max_chars))
        d = latex_escape(shorten(r["dense_top1"], max_chars))
        o = latex_escape(shorten(r["orca_top1"], max_chars))
        g = latex_escape(shorten(r["pseudo_gold"], max_chars))
        change = f"{r['dense_gold_rank']} $\\rightarrow$ {r['orca_gold_rank']}"
        lines.append(f"{q} & {d} & {o} & {g} & {change} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\caption{Representative Dense versus ORCA retrieval cases. Raw examples should be paraphrased and anonymized before inclusion in the paper.}")
    lines.append(r"\label{tab:qual-cases}")
    lines.append(r"\end{table*}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_json", required=True, help="Path to psydt_task_retrieval_final.json")
    ap.add_argument("--encoder_name", required=True, help="Local encoder path, e.g., ../models/thenlper/gte-large-zh or ../models/BAAI/bge-large-zh-v1___5")
    ap.add_argument("--split", default="test", choices=["dev", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache_dir", default="../cache/qual_cases")
    ap.add_argument("--out_dir", default="../results/qual_cases")
    ap.add_argument("--orca_k", type=int, default=2)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_cases", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_chars", type=int, default=80)
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)

    queries, corpus, gt, source, meta = load_split_and_source(args.benchmark_json, args.split)
    print("[benchmark]", json.dumps(meta, ensure_ascii=False, indent=2))

    enc = Encoder(args.encoder_name, batch_size=args.batch_size, device=args.device)
    model_short = Path(args.encoder_name).name.replace("/", "_").replace(".", "_")
    if model_short in ["", "."]:
        model_short = hashlib.sha1(args.encoder_name.encode("utf-8")).hexdigest()[:8]

    q_emb = cached_encode(enc, queries, args.cache_dir, f"{model_short}.{args.split}.q", no_cache=args.no_cache)
    d_emb = cached_encode(enc, corpus, args.cache_dir, f"{model_short}.{args.split}.d", no_cache=args.no_cache)
    s_emb = cached_encode(enc, source, args.cache_dir, f"{model_short}.source", no_cache=args.no_cache)

    basis = fit_orca_basis(s_emb, args.orca_k)
    q_orca = project_remove(q_emb, basis)
    d_orca = project_remove(d_emb, basis)

    dense_scores = q_emb @ d_emb.T
    orca_scores = q_orca @ d_orca.T

    dense_top = topk_from_scores(dense_scores, max(args.topk, 10))
    orca_top = topk_from_scores(orca_scores, max(args.topk, 10))
    dense_ranks = ranks_from_scores(dense_scores, gt)
    orca_ranks = ranks_from_scores(orca_scores, gt)

    rows: List[Dict[str, Any]] = []
    category_counts: Dict[str, int] = {}

    for i, g in enumerate(gt):
        g = int(g)
        dense_top1_idx = int(dense_top[i, 0])
        orca_top1_idx = int(orca_top[i, 0])
        dense_rank = int(dense_ranks[i])
        orca_rank = int(orca_ranks[i])
        cat = category_for_case(dense_rank, orca_rank, dense_top1_idx, orca_top1_idx, g)
        category_counts[cat] = category_counts.get(cat, 0) + 1

        q_diag = make_diag(queries[i])
        row = {
            "query_id": i,
            "case_category": cat,
            "dense_gold_rank": dense_rank,
            "orca_gold_rank": orca_rank,
            "rank_gain": dense_rank - orca_rank,
            "dense_top1_index": dense_top1_idx,
            "orca_top1_index": orca_top1_idx,
            "gold_index": g,
            "dense_top1_is_gold": int(dense_top1_idx == g),
            "orca_top1_is_gold": int(orca_top1_idx == g),
            "dense_hit_at_5": int(dense_rank <= 5),
            "orca_hit_at_5": int(orca_rank <= 5),
            "dense_hit_at_10": int(dense_rank <= 10),
            "orca_hit_at_10": int(orca_rank <= 10),
            "query": queries[i],
            "dense_top1": corpus[dense_top1_idx],
            "orca_top1": corpus[orca_top1_idx],
            "pseudo_gold": corpus[g],
            "dense_top5_indices": " ".join(str(int(x)) for x in dense_top[i, :5]),
            "orca_top5_indices": " ".join(str(int(x)) for x in orca_top[i, :5]),
            "query_first_person_hits": q_diag["first_person_hits"],
            "query_emotion_hits": q_diag["emotion_hits"],
            "query_physical_medical_hits": q_diag["physical_medical_hits"],
            "query_topic_entity_hits": q_diag["topic_entity_hits"],
            "query_char_len": q_diag["char_len"],
            "auto_note": (
                f"Gold rank changes from Dense {dense_rank} to ORCA {orca_rank}; "
                f"category={cat}."
            ),
        }
        rows.append(row)

    selected = select_cases(rows, max_cases=args.max_cases, seed=args.seed)

    raw_csv = Path(args.out_dir) / f"qualitative_cases_raw_{model_short}_{args.split}.csv"
    selected_csv = Path(args.out_dir) / f"qualitative_cases_selected_{model_short}_{args.split}.csv"
    md_path = Path(args.out_dir) / f"qualitative_cases_for_paper_{model_short}_{args.split}.md"
    tex_path = Path(args.out_dir) / f"qualitative_cases_latex_template_{model_short}_{args.split}.tex"
    summary_path = Path(args.out_dir) / f"qualitative_cases_summary_{model_short}_{args.split}.json"

    # Sort raw rows by usefulness.
    rows_sorted = sorted(rows, key=lambda r: (r["case_category"] == "other", -int(r["rank_gain"]), int(r["orca_gold_rank"])))
    write_csv(raw_csv, rows_sorted)
    write_csv(selected_csv, selected)
    save_markdown(md_path, selected, max_chars=args.max_chars)
    save_latex_template(tex_path, selected, max_chars=args.max_chars)

    summary = {
        "args": vars(args),
        "benchmark": meta,
        "model_short": model_short,
        "case_category_counts": category_counts,
        "selected_case_ids": [r["query_id"] for r in selected],
        "outputs": {
            "raw_csv": str(raw_csv),
            "selected_csv": str(selected_csv),
            "markdown": str(md_path),
            "latex_template": str(tex_path),
        },
        "dense_R@5_from_ranks": float(np.mean([r <= 5 for r in dense_ranks])),
        "orca_R@5_from_ranks": float(np.mean([r <= 5 for r in orca_ranks])),
        "dense_R@10_from_ranks": float(np.mean([r <= 10 for r in dense_ranks])),
        "orca_R@10_from_ranks": float(np.mean([r <= 10 for r in orca_ranks])),
    }
    write_json(summary_path, summary)

    print("[done]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
