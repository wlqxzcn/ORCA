from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


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


def write_csv(path: str | Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    ensure_dir(Path(path).parent)
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def batched(iterable: Sequence[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(iterable), batch_size):
        yield list(iterable[i : i + batch_size])


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
SOURCE_KEYS = ["emotion_corpus", "affective_corpus", "source_corpus", "train_user_source", "train_user_messages"]


def first_existing(d: Dict[str, Any], keys: Sequence[str], required_name: str) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"Cannot find {required_name}. Tried keys: {keys}. Existing top-level keys: {list(d.keys())[:50]}")


def normalize_gt(gt: Sequence[Any], corpus: Sequence[str]) -> List[int]:
    def resolve_one(x: Any, text_to_first_idx: Dict[str, int]) -> int:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, str):
            xs = x.strip()
            # allow signed integer strings just in case
            if xs.isdigit() or (xs.startswith("-") and xs[1:].isdigit()):
                return int(xs)
            if xs not in text_to_first_idx:
                raise ValueError(f"Gold text not found in corpus: {xs[:80]!r}")
            return text_to_first_idx[xs]
        raise TypeError(f"Unsupported atomic gold type: {type(x)}; value={repr(x)[:120]}")

    out: List[int] = []
    text_to_first_idx: Dict[str, int] = {}
    for i, t in enumerate(corpus):
        if t not in text_to_first_idx:
            text_to_first_idx[t] = i

    multi_gold_count = 0
    for g in gt:
        if isinstance(g, (list, tuple)):
            if len(g) == 0:
                raise ValueError("Encountered empty gold list.")
            if len(g) > 1:
                multi_gold_count += 1
            idx = resolve_one(g[0], text_to_first_idx)
        else:
            idx = resolve_one(g, text_to_first_idx)

        if idx < 0 or idx >= len(corpus):
            raise ValueError(f"Gold index out of range: {idx}, corpus size={len(corpus)}")
        out.append(idx)

    if multi_gold_count:
        print(
            f"[warn] {multi_gold_count} queries contain multiple gold labels; "
            "using the first label for single-pseudo-gold evaluation."
        )
    return out


@dataclass
class SplitData:
    queries: List[str]
    corpus: List[str]
    gt: List[int]


@dataclass
class Benchmark:
    dev: SplitData
    test: SplitData
    source: List[str]
    meta: Dict[str, Any]


def load_benchmark(path: str | Path) -> Benchmark:
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise ValueError("Benchmark JSON must be a dictionary.")

    dev_queries = list(first_existing(obj, QUERY_KEYS["dev"], "dev queries"))
    test_queries = list(first_existing(obj, QUERY_KEYS["test"], "test queries"))
    dev_corpus = list(first_existing(obj, CORPUS_KEYS["dev"], "dev corpus"))
    test_corpus = list(first_existing(obj, CORPUS_KEYS["test"], "test corpus"))
    dev_gt_raw = list(first_existing(obj, GT_KEYS["dev"], "dev gold labels"))
    test_gt_raw = list(first_existing(obj, GT_KEYS["test"], "test gold labels"))

    source = None
    for k in SOURCE_KEYS:
        if k in obj:
            source = list(obj[k])
            break
    if source is None:
        raise KeyError(f"Cannot find ORCA source corpus. Tried keys: {SOURCE_KEYS}")

    dev_gt = normalize_gt(dev_gt_raw, dev_corpus)
    test_gt = normalize_gt(test_gt_raw, test_corpus)

    if not (len(dev_queries) == len(dev_gt)):
        raise ValueError(f"dev query/gold length mismatch: {len(dev_queries)} vs {len(dev_gt)}")
    if not (len(test_queries) == len(test_gt)):
        raise ValueError(f"test query/gold length mismatch: {len(test_queries)} vs {len(test_gt)}")

    meta = {
        "benchmark_json": str(path),
        "dev_queries": len(dev_queries),
        "test_queries": len(test_queries),
        "dev_corpus": len(dev_corpus),
        "test_corpus": len(test_corpus),
        "source": len(source),
        "source_hash": sha1_of_texts(source),
        "dev_query_hash": sha1_of_texts(dev_queries),
        "test_query_hash": sha1_of_texts(test_queries),
        "dev_corpus_hash": sha1_of_texts(dev_corpus),
        "test_corpus_hash": sha1_of_texts(test_corpus),
    }

    return Benchmark(
        dev=SplitData(dev_queries, dev_corpus, dev_gt),
        test=SplitData(test_queries, test_corpus, test_gt),
        source=source,
        meta=meta,
    )


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
            print(f"[warn] SentenceTransformer loading failed, fallback to transformers mean pooling: {e}", file=sys.stderr)

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
        if self.kind == "sentence_transformers":
            arr = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
            return arr.astype("float32")

        import torch
        all_vecs = []
        with torch.no_grad():
            for batch in batched(list(texts), self.batch_size):
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
                all_vecs.append(pooled.detach().cpu().numpy().astype("float32"))
        return np.vstack(all_vecs)


def cached_encode(
    encoder: Encoder,
    texts: Sequence[str],
    cache_dir: str | Path,
    name: str,
    normalize: bool = True,
) -> np.ndarray:
    ensure_dir(cache_dir)
    text_hash = sha1_of_texts(texts)
    model_hash = hashlib.sha1(encoder.model_name_or_path.encode("utf-8")).hexdigest()[:10]
    cache_path = Path(cache_dir) / f"{name}.{model_hash}.{len(texts)}.{text_hash}.npy"
    if cache_path.exists():
        arr = np.load(cache_path)
    else:
        print(f"[encode] {name}: n={len(texts)}")
        arr = encoder.encode(texts)
        np.save(cache_path, arr)
    if normalize:
        arr = l2_normalize(arr.astype("float32"))
    return arr.astype("float32")



def topk_inner_product(q: np.ndarray, d: np.ndarray, k: int) -> np.ndarray:
    scores = q @ d.T
    if k >= d.shape[0]:
        idx = np.argsort(-scores, axis=1)
    else:
        part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(scores, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
    return idx[:, :k]


def evaluate(q: np.ndarray, d: np.ndarray, gt: Sequence[int], topk: int = 10) -> Dict[str, float]:
    max_k = max(1, topk, 10)
    ranked = topk_inner_product(q, d, max_k)
    n = len(gt)
    r1 = r5 = r10 = 0
    mrr = 0.0
    ndcg10 = 0.0
    for i, g in enumerate(gt):
        row = ranked[i]
        pos = np.where(row == int(g))[0]
        if len(pos) > 0:
            rank = int(pos[0]) + 1
            if rank <= 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
            if rank <= 10:
                r10 += 1
            mrr += 1.0 / rank
            if rank <= 10:
                ndcg10 += 1.0 / math.log2(rank + 1)
    return {
        "R@1": r1 / n,
        "R@5": r5 / n,
        "R@10": r10 / n,
        "MRR": mrr / n,
        "nDCG@10": ndcg10 / n,
    }


def paired_hits(q: np.ndarray, d: np.ndarray, gt: Sequence[int], k: int) -> np.ndarray:
    ranked = topk_inner_product(q, d, k)
    hits = np.zeros(len(gt), dtype=bool)
    for i, g in enumerate(gt):
        hits[i] = int(g) in set(map(int, ranked[i]))
    return hits


def fit_orca_basis(source_emb: np.ndarray, k: int) -> np.ndarray:
    _, _, vt = np.linalg.svd(source_emb.astype("float64"), full_matrices=False)
    return vt[:k].T.astype("float32")


def project_remove(x: np.ndarray, basis: np.ndarray, renorm: bool = True) -> np.ndarray:
    y = x - (x @ basis) @ basis.T
    y = y.astype("float32")
    return l2_normalize(y) if renorm else y


def fit_pca_basis(x: np.ndarray, k: int, centered: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    xx = x.astype("float64")
    mean = None
    if centered:
        mean = xx.mean(axis=0, keepdims=True)
        xx = xx - mean
    _, _, vt = np.linalg.svd(xx, full_matrices=False)
    return vt[:k].T.astype("float32"), (mean.astype("float32") if mean is not None else None)


def pca_remove(x: np.ndarray, basis: np.ndarray, mean: Optional[np.ndarray], renorm: bool = True) -> np.ndarray:
    if mean is not None:
        xx = x - mean
        y = xx - (xx @ basis) @ basis.T
    else:
        y = x - (x @ basis) @ basis.T
    y = y.astype("float32")
    return l2_normalize(y) if renorm else y


def fit_whitening(x: np.ndarray, eps: float = 1e-5) -> Tuple[np.ndarray, np.ndarray]:
    xx = x.astype("float64")
    mean = xx.mean(axis=0, keepdims=True)
    xc = xx - mean
    cov = (xc.T @ xc) / max(1, xc.shape[0] - 1)
    u, s, _ = np.linalg.svd(cov, full_matrices=False)
    W = u @ np.diag(1.0 / np.sqrt(s + eps)) @ u.T
    return mean.astype("float32"), W.astype("float32")


def apply_whitening(x: np.ndarray, mean: np.ndarray, W: np.ndarray) -> np.ndarray:
    y = (x - mean) @ W
    return l2_normalize(y.astype("float32"))


def random_orthonormal_basis(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(dim, k)).astype("float64")
    q, _ = np.linalg.qr(a)
    return q[:, :k].astype("float32")


def run_k_sweep(
    source_emb: np.ndarray,
    dev_q: np.ndarray,
    dev_d: np.ndarray,
    dev_gt: Sequence[int],
    ks: Sequence[int],
) -> List[Dict[str, Any]]:
    rows = []
    for k in ks:
        basis = fit_orca_basis(source_emb, k)
        q_proj = project_remove(dev_q, basis)
        d_proj = project_remove(dev_d, basis)
        m = evaluate(q_proj, d_proj, dev_gt, topk=10)
        rows.append({"k": k, **m})
        print(f"[k-sweep] k={k} R@5={m['R@5']:.4f}")
    return rows


def run_test_baselines(
    source_emb: np.ndarray,
    test_q: np.ndarray,
    test_d: np.ndarray,
    test_gt: Sequence[int],
    k: int,
    random_trials: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    dense = evaluate(test_q, test_d, test_gt, topk=10)
    rows.append({"method": "Dense", **dense})

    basis = fit_orca_basis(source_emb, k)
    orca_q = project_remove(test_q, basis)
    orca_d = project_remove(test_d, basis)
    orca = evaluate(orca_q, orca_d, test_gt, topk=10)
    rows.append({"method": f"ORCA(k={k})", **orca})


    pca_basis, pca_mean = fit_pca_basis(test_d, k=k, centered=False)
    pca_q = pca_remove(test_q, pca_basis, pca_mean)
    pca_d = pca_remove(test_d, pca_basis, pca_mean)
    pca = evaluate(pca_q, pca_d, test_gt, topk=10)
    rows.append({"method": f"PCA-removal(k={k},raw)", **pca})


    w_mean, W = fit_whitening(test_d)
    w_q = apply_whitening(test_q, w_mean, W)
    w_d = apply_whitening(test_d, w_mean, W)
    whitening = evaluate(w_q, w_d, test_gt, topk=10)
    rows.append({"method": "Whitening", **whitening})


    trial_metrics = []
    rng = np.random.default_rng(seed)
    for t in range(random_trials):
        rb = random_orthonormal_basis(test_q.shape[1], k, rng)
        rq = project_remove(test_q, rb)
        rd = project_remove(test_d, rb)
        rm = evaluate(rq, rd, test_gt, topk=10)
        trial_metrics.append(rm)
        rows.append({"method": f"Random-removal(k={k},trial={t+1})", **rm})

    r5_values = np.array([m["R@5"] for m in trial_metrics], dtype=float)
    random_summary = {
        "method": f"Random-removal(k={k},mean±std,n={random_trials})",
        "R@1": float(np.mean([m["R@1"] for m in trial_metrics])),
        "R@5": float(r5_values.mean()),
        "R@5_std": float(r5_values.std(ddof=1)) if random_trials > 1 else 0.0,
        "R@10": float(np.mean([m["R@10"] for m in trial_metrics])),
        "MRR": float(np.mean([m["MRR"] for m in trial_metrics])),
        "nDCG@10": float(np.mean([m["nDCG@10"] for m in trial_metrics])),
    }

    artifacts = {
        "orca_basis": basis,
        "orca_q": orca_q,
        "orca_d": orca_d,
        "dense_q": test_q,
        "dense_d": test_d,
        "whitening_q": w_q,
        "whitening_d": w_d,
        "random_summary": random_summary,
    }
    return rows + [random_summary], artifacts


def make_human_eval_csv(
    out_path: str | Path,
    queries: Sequence[str],
    corpus: Sequence[str],
    gt: Sequence[int],
    dense_q: np.ndarray,
    dense_d: np.ndarray,
    orca_q: np.ndarray,
    orca_d: np.ndarray,
    sample_n: int,
    topn: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    n = len(queries)
    sample_ids = list(range(n))
    rng.shuffle(sample_ids)
    sample_ids = sample_ids[: min(sample_n, n)]

    dense_top = topk_inner_product(dense_q, dense_d, topn)
    orca_top = topk_inner_product(orca_q, orca_d, topn)

    rows = []
    for qid in sample_ids:
        if rng.random() < 0.5:
            mapping = {"Dense": "System A", "ORCA": "System B"}
        else:
            mapping = {"Dense": "System B", "ORCA": "System A"}
        for method, top in [("Dense", dense_top), ("ORCA", orca_top)]:
            seen_text = set()
            for rank, cand_idx in enumerate(top[qid], start=1):
                cand_idx = int(cand_idx)
                txt = corpus[cand_idx]
                if txt in seen_text:
                    continue
                seen_text.add(txt)
                rows.append(
                    {
                        "query_id": qid,
                        "query": queries[qid],
                        "system_blind": mapping[method],
                        "rank": rank,
                        "candidate_index": cand_idx,
                        "candidate_response": txt,
                        "is_pseudo_gold": int(cand_idx == int(gt[qid])),
                        "relevance_1_5": "",
                        "supportiveness_1_5": "",
                        "safety_1_5": "",
                        "notes": "",
                        "true_method_remove_before_annotation": method,
                    }
                )
    write_csv(out_path, rows)
    print(f"[human-eval] wrote {len(rows)} rows to {out_path}")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark_json", type=str, required=True, help="Path to psydt_task_retrieval_final.json")
    ap.add_argument("--encoder_name", type=str, required=True, help="Local encoder path or model name")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--cache_dir", type=str, default="../cache/orca_supplementary")
    ap.add_argument("--out_dir", type=str, default="../results/orca_supplementary")
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64])
    ap.add_argument("--selected_k", type=int, default=None, help="If omitted, select best k by dev R@5")
    ap.add_argument("--random_trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--make_human_eval", action="store_true")
    ap.add_argument("--human_sample_n", type=int, default=50)
    ap.add_argument("--human_topn", type=int, default=3)
    args = ap.parse_args()

    set_seed(args.seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)

    bench = load_benchmark(args.benchmark_json)
    print("[benchmark]", json.dumps(bench.meta, ensure_ascii=False, indent=2))

    enc = Encoder(args.encoder_name, batch_size=args.batch_size, device=args.device)
    model_short = Path(args.encoder_name).name.replace("/", "_").replace(".", "_")
    if model_short in ["", "."]:
        model_short = hashlib.sha1(args.encoder_name.encode("utf-8")).hexdigest()[:8]


    source_emb = cached_encode(enc, bench.source, args.cache_dir, f"{model_short}.source", normalize=True)
    dev_q = cached_encode(enc, bench.dev.queries, args.cache_dir, f"{model_short}.dev_q", normalize=True)
    dev_d = cached_encode(enc, bench.dev.corpus, args.cache_dir, f"{model_short}.dev_d", normalize=True)
    test_q = cached_encode(enc, bench.test.queries, args.cache_dir, f"{model_short}.test_q", normalize=True)
    test_d = cached_encode(enc, bench.test.corpus, args.cache_dir, f"{model_short}.test_d", normalize=True)


    k_rows = run_k_sweep(source_emb, dev_q, dev_d, bench.dev.gt, args.ks)
    k_path = Path(args.out_dir) / f"k_sweep_{model_short}.csv"
    write_csv(k_path, k_rows)

    if args.selected_k is None:
        best = max(k_rows, key=lambda r: (r["R@5"], r["R@1"], -int(r["k"])))
        selected_k = int(best["k"])
    else:
        selected_k = int(args.selected_k)
    print(f"[selected_k] {selected_k}")


    rows, artifacts = run_test_baselines(
        source_emb=source_emb,
        test_q=test_q,
        test_d=test_d,
        test_gt=bench.test.gt,
        k=selected_k,
        random_trials=args.random_trials,
        seed=args.seed,
    )
    baseline_path = Path(args.out_dir) / f"baselines_whitening_random_{model_short}.csv"
    write_csv(baseline_path, rows)


    human_path = None
    if args.make_human_eval:
        human_path = Path(args.out_dir) / f"human_eval_sample_{model_short}.csv"
        make_human_eval_csv(
            out_path=human_path,
            queries=bench.test.queries,
            corpus=bench.test.corpus,
            gt=bench.test.gt,
            dense_q=artifacts["dense_q"],
            dense_d=artifacts["dense_d"],
            orca_q=artifacts["orca_q"],
            orca_d=artifacts["orca_d"],
            sample_n=args.human_sample_n,
            topn=args.human_topn,
            seed=args.seed,
        )

    meta = {
        "benchmark": bench.meta,
        "encoder": args.encoder_name,
        "encoder_kind": enc.kind,
        "ks": args.ks,
        "selected_k": selected_k,
        "random_trials": args.random_trials,
        "seed": args.seed,
        "outputs": {
            "k_sweep": str(k_path),
            "baselines_whitening_random": str(baseline_path),
            "human_eval_sample": str(human_path) if human_path else None,
        },
        "random_summary": artifacts["random_summary"],
    }
    meta_path = Path(args.out_dir) / f"metadata_{model_short}.json"
    write_json(meta_path, meta)
    print("[done] outputs:")
    print(json.dumps(meta["outputs"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

