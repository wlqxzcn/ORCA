import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

def now_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir_for_file(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_name(name: str) -> str:
    name = str(name)
    for ch in ["/", "\\", ":", " ", "\t", "\n"]:
        name = name.replace(ch, "_")
    return name


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def print_metrics(prefix: str, m: Dict[str, float]):
    keys = ["R@1", "R@5", "R@10", "MRR", "nDCG@10", "latency_ms_per_query"]
    parts = []
    for k in keys:
        if k in m:
            parts.append(f"{k}={m[k]:.4f}")
    print(f"{prefix}: " + " | ".join(parts))



def looks_like_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (path / "modules.json").exists() or (path / "config.json").exists()


def find_bge_model(models_root: Path) -> Optional[Path]:
    baai_dir = models_root / "BAAI"

    if looks_like_model_dir(baai_dir):
        return baai_dir

    if not baai_dir.exists():
        return None

    preferred = [
        "bge-large-zh-v1.5",
        "bge-large-zh-v1___5",
        "bge-base-zh-v1.5",
        "bge-base-zh-v1___5",
        "bge-small-zh-v1.5",
        "bge-small-zh-v1___5",
    ]

    for name in preferred:
        p = baai_dir / name
        if looks_like_model_dir(p):
            return p

    candidates = []
    for p in baai_dir.rglob("*"):
        if p.is_dir() and looks_like_model_dir(p):
            candidates.append(p)

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda p: (
            0 if "bge" in p.name.lower() else 1,
            0 if "large" in p.name.lower() else 1,
            len(str(p)),
        ),
    )
    return candidates[0]


def resolve_model_paths(
    models_root: str,
    gte_path: Optional[str] = None,
    bge_path: Optional[str] = None,
) -> Dict[str, Path]:
    root = Path(models_root).resolve()
    out = {}

    if gte_path:
        p = Path(gte_path).resolve()
    else:
        p = (root / "thenlper" / "gte-large-zh").resolve()

    if looks_like_model_dir(p):
        out["gte"] = p
    else:
        print(f"[Warning] GTE model not found: {p}")

    if bge_path:
        p = Path(bge_path).resolve()
    else:
        p = find_bge_model(root)

    if p is not None and looks_like_model_dir(p):
        out["bge"] = p.resolve()
    else:
        print(f"[Warning] BGE model not found under: {root / 'BAAI'}")

    if not out:
        raise FileNotFoundError("No valid local model found.")

    return out


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    cache_path: Optional[str],
) -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            print(f"[Cache] loaded {cache_path} {arr.shape}")
            return arr.astype("float32")
        print(f"[Cache warning] shape mismatch for {cache_path}, recomputing.")

    print(f"[Encode] {len(texts)} texts")
    arr = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, arr)
        print(f"[Cache] saved {cache_path} {arr.shape}")

    return arr


def evaluate(indices: np.ndarray, gt: List[List[int]]) -> Dict[str, float]:
    n = len(gt)
    hit1 = hit5 = hit10 = 0
    mrr = 0.0
    ndcg10 = 0.0

    for i in range(n):
        gold = set(int(x) for x in gt[i])
        ranking = [int(x) for x in indices[i].tolist()]

        if any(x in gold for x in ranking[:1]):
            hit1 += 1
        if any(x in gold for x in ranking[:5]):
            hit5 += 1
        if any(x in gold for x in ranking[:10]):
            hit10 += 1

        found_rank = None
        for r, doc_id in enumerate(ranking, start=1):
            if doc_id in gold:
                found_rank = r
                break

        if found_rank is not None:
            mrr += 1.0 / found_rank
            if found_rank <= 10:
                ndcg10 += 1.0 / math.log2(found_rank + 1)

    return {
        "R@1": hit1 / n,
        "R@5": hit5 / n,
        "R@10": hit10 / n,
        "MRR": mrr / n,
        "nDCG@10": ndcg10 / n,
    }


def relative_gain(m: Dict[str, float], base: Dict[str, float]) -> Dict[str, float]:
    out = dict(m)
    for key in ["R@1", "R@5", "R@10", "MRR", "nDCG@10"]:
        if key in m and key in base and base[key] > 0:
            out[f"relative_gain_{key}"] = (m[key] - base[key]) / base[key]
    return out


def search_dense(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    top_k: int,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    q = query_embs.astype("float32")
    c = corpus_embs.astype("float32")

    if normalize:
        q = q.copy()
        c = c.copy()
        faiss.normalize_L2(q)
        faiss.normalize_L2(c)

    index = faiss.IndexFlatIP(c.shape[1])
    index.add(c)

    start = time.perf_counter()
    scores, indices = index.search(q, top_k)
    latency = (time.perf_counter() - start) * 1000.0 / max(len(q), 1)

    return scores, indices, latency


def run_dense(
    corpus_embs: np.ndarray,
    query_embs: np.ndarray,
    gt: List[List[int]],
    top_k: int,
    depth: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    retrieve_k = max(top_k, depth)
    scores, indices, latency = search_dense(query_embs, corpus_embs, retrieve_k, normalize=True)
    m = evaluate(indices[:, :top_k], gt)
    m["latency_ms_per_query"] = latency
    return m, indices, scores



def make_tokenizer(mode: str):
    if mode in {"auto", "jieba"}:
        try:
            import jieba  # type: ignore

            def tok_jieba(text: str) -> List[str]:
                return [w.strip() for w in jieba.lcut(str(text)) if w.strip()]

            print("[BM25] tokenizer=jieba")
            return tok_jieba
        except Exception:
            if mode == "jieba":
                raise
            print("[BM25] jieba unavailable, fallback to char tokenizer.")

    def tok_char(text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", str(text))

    print("[BM25] tokenizer=char")
    return tok_char


class BM25Index:
    def __init__(self, docs: List[str], tokenizer, k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)
        self.N = len(docs)
        self.doc_lens = np.zeros(self.N, dtype=np.float32)
        self.avgdl = 0.0
        self.idf: Dict[str, float] = {}
        self.inverted: Dict[str, Dict[int, int]] = {}
        self._build()

    def _build(self):
        print(f"[BM25] building index for {self.N} docs")
        df = Counter()
        inverted: Dict[str, Dict[int, int]] = defaultdict(dict)
        total_len = 0

        for doc_id, text in enumerate(self.docs):
            tokens = self.tokenizer(text)
            tf = Counter(tokens)
            dl = sum(tf.values())
            self.doc_lens[doc_id] = dl
            total_len += dl

            for term, count in tf.items():
                df[term] += 1
                inverted[term][doc_id] = int(count)

        self.avgdl = float(total_len / max(self.N, 1))
        self.inverted = dict(inverted)

        for term, dfi in df.items():
            self.idf[term] = math.log(1.0 + (self.N - dfi + 0.5) / (dfi + 0.5))

        print(f"[BM25] vocab={len(self.idf)}, avgdl={self.avgdl:.2f}")

    def score_query(self, query: str) -> np.ndarray:
        tokens = self.tokenizer(query)
        qtf = Counter(tokens)
        scores = np.zeros(self.N, dtype=np.float32)

        for term in qtf.keys():
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)

            for doc_id, tf in postings.items():
                dl = self.doc_lens[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-6))
                scores[doc_id] += idf * (tf * (self.k1 + 1.0)) / max(denom, 1e-6)

        return scores

    def search(self, queries: List[str], top_k: int) -> Tuple[np.ndarray, np.ndarray, float]:
        all_scores = np.zeros((len(queries), top_k), dtype=np.float32)
        all_indices = np.zeros((len(queries), top_k), dtype=np.int64)

        start = time.perf_counter()
        for i, query in enumerate(queries):
            scores = self.score_query(query)
            if top_k >= self.N:
                idx = np.argsort(-scores)[:top_k]
            else:
                part = np.argpartition(-scores, kth=top_k - 1)[:top_k]
                idx = part[np.argsort(-scores[part])]
            all_indices[i] = idx
            all_scores[i] = scores[idx]

        latency = (time.perf_counter() - start) * 1000.0 / max(len(queries), 1)
        return all_scores, all_indices, latency


def run_bm25(
    docs: List[str],
    queries: List[str],
    gt: List[List[int]],
    tokenizer,
    top_k: int,
    depth: int,
    k1: float,
    b: float,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    retrieve_k = max(top_k, depth)
    bm25 = BM25Index(docs, tokenizer, k1=k1, b=b)
    scores, indices, latency = bm25.search(queries, retrieve_k)
    m = evaluate(indices[:, :top_k], gt)
    m["latency_ms_per_query"] = latency
    return m, indices, scores


def fit_svd_basis(
    emotion_embs: np.ndarray,
    max_k: int,
    normalize_for_svd: bool = False,
    center_for_svd: bool = False,
) -> np.ndarray:
    E = torch.from_numpy(emotion_embs.astype("float32"))

    if normalize_for_svd:
        E = F.normalize(E, p=2, dim=1)

    if center_for_svd:
        E = E - E.mean(dim=0, keepdim=True)

    E = E.double()
    max_k = min(max_k, E.shape[0], E.shape[1])

    print(
        f"[ORCA SVD] E={tuple(E.shape)}, max_k={max_k}, "
        f"normalize_for_svd={normalize_for_svd}, center_for_svd={center_for_svd}"
    )

    with torch.no_grad():
        _, _, Vh = torch.linalg.svd(E, full_matrices=False)
        B = Vh[:max_k].T.contiguous()
        B, _ = torch.linalg.qr(B, mode="reduced")

    return B.float().cpu().numpy().astype("float32")


def project_remove(x: np.ndarray, B: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    B = B.astype("float32")
    return (x - (x @ B) @ B.T).astype("float32")


def run_orca(
    corpus_embs: np.ndarray,
    query_embs: np.ndarray,
    gt: List[List[int]],
    B: np.ndarray,
    top_k: int,
    depth: int,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    c = project_remove(corpus_embs, B)
    q = project_remove(query_embs, B)
    return run_dense(c, q, gt, top_k=top_k, depth=depth)


def select_orca_k(
    B_max: np.ndarray,
    k_list: List[int],
    dev_corpus_embs: np.ndarray,
    dev_query_embs: np.ndarray,
    dev_gt: List[List[int]],
    top_k: int,
    depth: int,
    selection_metric: str,
) -> Tuple[int, Dict[str, Dict[str, float]]]:
    best_k = None
    best_score = -1.0
    sweep = {}

    print("\n[ORCA] dev k selection")
    for k in k_list:
        B = B_max[:, :k]
        m, _, _ = run_orca(dev_corpus_embs, dev_query_embs, dev_gt, B, top_k=top_k, depth=depth)
        sweep[str(k)] = m
        print_metrics(f"[ORCA][k={k}][dev]", m)

        score = float(m[selection_metric])
        if score > best_score:
            best_k = int(k)
            best_score = score

    print(f"[ORCA] selected k={best_k}, {selection_metric}={best_score:.4f}")
    return int(best_k), sweep



def rrf_fusion(
    ranking_a: np.ndarray,
    ranking_b: np.ndarray,
    top_k: int,
    depth: int,
    rrf_k: int = 60,
    weight_a: float = 1.0,
    weight_b: float = 1.0,
) -> np.ndarray:
    n_queries = ranking_a.shape[0]
    out = np.zeros((n_queries, top_k), dtype=np.int64)

    depth_a = min(depth, ranking_a.shape[1])
    depth_b = min(depth, ranking_b.shape[1])

    for i in range(n_queries):
        scores: Dict[int, float] = defaultdict(float)

        for r in range(depth_a):
            doc_id = int(ranking_a[i, r])
            scores[doc_id] += weight_a / (rrf_k + r + 1)

        for r in range(depth_b):
            doc_id = int(ranking_b[i, r])
            scores[doc_id] += weight_b / (rrf_k + r + 1)

        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        out[i] = [doc_id for doc_id, _ in ordered[:top_k]]

    return out


def select_rrf_params(
    name: str,
    dev_rank_a: np.ndarray,
    dev_rank_b: np.ndarray,
    dev_gt: List[List[int]],
    top_k: int,
    depth: int,
    rrf_k_list: List[int],
    weight_a_list: List[float],
    weight_b_list: List[float],
    selection_metric: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    print(f"\n[{name}] dev RRF parameter selection")

    best_cfg = None
    best_score = -1.0
    results = {}

    for rrf_k in rrf_k_list:
        for wa in weight_a_list:
            for wb in weight_b_list:
                fused = rrf_fusion(
                    dev_rank_a,
                    dev_rank_b,
                    top_k=top_k,
                    depth=depth,
                    rrf_k=rrf_k,
                    weight_a=wa,
                    weight_b=wb,
                )
                m = evaluate(fused, dev_gt)
                key = f"rrf_k={rrf_k},wa={wa},wb={wb}"
                results[key] = m

                score = float(m[selection_metric])
                if score > best_score:
                    best_score = score
                    best_cfg = {
                        "rrf_k": int(rrf_k),
                        "weight_a": float(wa),
                        "weight_b": float(wb),
                        selection_metric: best_score,
                    }

    assert best_cfg is not None
    print(
        f"[{name}] best rrf_k={best_cfg['rrf_k']}, "
        f"weight_a={best_cfg['weight_a']}, weight_b={best_cfg['weight_b']}, "
        f"{selection_metric}={best_cfg[selection_metric]:.4f}"
    )
    return best_cfg, results


def run_rrf_test(
    rank_a: np.ndarray,
    rank_b: np.ndarray,
    gt: List[List[int]],
    cfg: Dict[str, Any],
    top_k: int,
    depth: int,
) -> Tuple[Dict[str, float], np.ndarray]:
    fused = rrf_fusion(
        rank_a,
        rank_b,
        top_k=top_k,
        depth=depth,
        rrf_k=int(cfg["rrf_k"]),
        weight_a=float(cfg["weight_a"]),
        weight_b=float(cfg["weight_b"]),
    )
    return evaluate(fused, gt), fused



def md_row(name: str, m: Dict[str, float]) -> str:
    return (
        f"| {name} | {m['R@1']:.4f} | {m['R@5']:.4f} | {m['R@10']:.4f} | "
        f"{m['MRR']:.4f} | {m['nDCG@10']:.4f} |"
    )


def make_markdown(results: Dict[str, Any]) -> str:
    lines = []
    lines.append("# ORCA + BM25 RRF Fusion Summary\n")
    lines.append(f"Generated at: {results.get('created_at', '')}\n")

    for model_key, r in results["models"].items():
        lines.append(f"## {model_key}\n")
        lines.append(f"- Model path: `{r['model_path']}`")
        lines.append(f"- ORCA selected k: **{r['orca']['selected_k']}**")
        lines.append(f"- BM25+Dense RRF cfg: `{r['bm25_dense_rrf']['selected_cfg']}`")
        lines.append(f"- BM25+ORCA RRF cfg: `{r['bm25_orca_rrf']['selected_cfg']}`")
        lines.append("")

        lines.append("### Test results\n")
        lines.append("| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 |")
        lines.append("|---|---:|---:|---:|---:|---:|")

        rows = [
            ("BM25", r["bm25"]["test"]),
            ("Dense", r["dense"]["test"]),
            ("ORCA", r["orca"]["test"]),
            ("BM25 + Dense RRF", r["bm25_dense_rrf"]["test"]),
            ("BM25 + ORCA RRF", r["bm25_orca_rrf"]["test"]),
        ]

        for name, m in rows:
            lines.append(md_row(name, m))

        lines.append("")
        dense_r5 = r["dense"]["test"]["R@5"]
        lines.append("### Relative R@5 gain over Dense\n")
        lines.append("| Method | R@5 | Relative gain |")
        lines.append("|---|---:|---:|")
        for name, m in rows:
            rel = 0.0 if name == "Dense" else (m["R@5"] - dense_r5) / dense_r5
            lines.append(f"| {name} | {m['R@5']:.4f} | {rel * 100:.2f}% |")
        lines.append("")

        lines.append("### ORCA dev k sweep\n")
        lines.append("| k | R@1 | R@5 | R@10 | MRR | nDCG@10 |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for k, m in r["orca"]["dev_k_sweep"].items():
            lines.append(
                f"| {k} | {m['R@1']:.4f} | {m['R@5']:.4f} | "
                f"{m['R@10']:.4f} | {m['MRR']:.4f} | {m['nDCG@10']:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)



def run_for_one_model(
    model_key: str,
    model_path: Path,
    data: Dict[str, Any],
    bm25_dev: Dict[str, Any],
    bm25_test: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print("\n" + "=" * 100)
    print(f"[Model] {model_key}: {model_path}")
    print("=" * 100)

    batch_size = args.bge_batch_size if model_key == "bge" else args.gte_batch_size

    model = SentenceTransformer(str(model_path), device=args.device)

    cache_root = Path(args.cache_dir) / safe_name(model_key)
    cache_root.mkdir(parents=True, exist_ok=True)

    def cache(name: str) -> Optional[str]:
        if args.no_cache:
            return None
        return str(cache_root / f"{name}.npy")

    emotion_embs = encode_texts(model, data["emotion_corpus"], batch_size, cache("emotion_corpus"))
    dev_corpus_embs = encode_texts(model, data["dev_task_corpus"], batch_size, cache("dev_task_corpus"))
    dev_query_embs = encode_texts(model, data["dev_queries"], batch_size, cache("dev_queries"))
    test_corpus_embs = encode_texts(model, data["test_task_corpus"], batch_size, cache("test_task_corpus"))
    test_query_embs = encode_texts(model, data["test_queries"], batch_size, cache("test_queries"))

    top_k = args.top_k
    depth = args.fusion_depth

    print("\n[Dense]")
    dense_dev, dense_dev_idx, _ = run_dense(
        dev_corpus_embs, dev_query_embs, data["dev_gt"], top_k=top_k, depth=depth
    )
    dense_test, dense_test_idx, _ = run_dense(
        test_corpus_embs, test_query_embs, data["test_gt"], top_k=top_k, depth=depth
    )
    print_metrics("[Dense][dev]", dense_dev)
    print_metrics("[Dense][test]", dense_test)

    print("\n[ORCA]")
    k_list = parse_int_list(args.k_list)
    B_max = fit_svd_basis(
        emotion_embs=emotion_embs,
        max_k=max(k_list),
        normalize_for_svd=args.normalize_for_svd,
        center_for_svd=args.center_for_svd,
    )
    selected_k, dev_k_sweep = select_orca_k(
        B_max,
        k_list=k_list,
        dev_corpus_embs=dev_corpus_embs,
        dev_query_embs=dev_query_embs,
        dev_gt=data["dev_gt"],
        top_k=top_k,
        depth=depth,
        selection_metric=args.selection_metric,
    )
    B = B_max[:, :selected_k]

    orca_dev, orca_dev_idx, _ = run_orca(
        dev_corpus_embs, dev_query_embs, data["dev_gt"], B, top_k=top_k, depth=depth
    )
    orca_test, orca_test_idx, _ = run_orca(
        test_corpus_embs, test_query_embs, data["test_gt"], B, top_k=top_k, depth=depth
    )
    print_metrics("[ORCA][dev]", orca_dev)
    print_metrics("[ORCA][test]", orca_test)

    rrf_k_list = parse_int_list(args.rrf_k_list)
    dense_weights = parse_float_list(args.dense_weight_list)
    orca_weights = parse_float_list(args.orca_weight_list)
    bm25_weights = parse_float_list(args.bm25_weight_list)

    bd_cfg, bd_dev_grid = select_rrf_params(
        name="BM25+Dense RRF",
        dev_rank_a=dense_dev_idx,
        dev_rank_b=bm25_dev["indices"],
        dev_gt=data["dev_gt"],
        top_k=top_k,
        depth=depth,
        rrf_k_list=rrf_k_list,
        weight_a_list=dense_weights,
        weight_b_list=bm25_weights,
        selection_metric=args.selection_metric,
    )
    bm25_dense_test, bm25_dense_test_idx = run_rrf_test(
        dense_test_idx,
        bm25_test["indices"],
        data["test_gt"],
        cfg=bd_cfg,
        top_k=top_k,
        depth=depth,
    )
    print_metrics("[BM25+Dense RRF][test]", bm25_dense_test)

    bo_cfg, bo_dev_grid = select_rrf_params(
        name="BM25+ORCA RRF",
        dev_rank_a=orca_dev_idx,
        dev_rank_b=bm25_dev["indices"],
        dev_gt=data["dev_gt"],
        top_k=top_k,
        depth=depth,
        rrf_k_list=rrf_k_list,
        weight_a_list=orca_weights,
        weight_b_list=bm25_weights,
        selection_metric=args.selection_metric,
    )
    bm25_orca_test, bm25_orca_test_idx = run_rrf_test(
        orca_test_idx,
        bm25_test["indices"],
        data["test_gt"],
        cfg=bo_cfg,
        top_k=top_k,
        depth=depth,
    )
    print_metrics("[BM25+ORCA RRF][test]", bm25_orca_test)

    result = {
        "model_key": model_key,
        "model_path": str(model_path),
        "embedding_dim": int(emotion_embs.shape[1]),
        "bm25": {
            "dev": bm25_dev["metrics"],
            "test": bm25_test["metrics"],
        },
        "dense": {
            "dev": dense_dev,
            "test": dense_test,
        },
        "orca": {
            "selected_k": int(selected_k),
            "dev": relative_gain(orca_dev, dense_dev),
            "test": relative_gain(orca_test, dense_test),
            "dev_k_sweep": dev_k_sweep,
        },
        "bm25_dense_rrf": {
            "selected_cfg": bd_cfg,
            "dev_grid": bd_dev_grid,
            "test": relative_gain(bm25_dense_test, dense_test),
        },
        "bm25_orca_rrf": {
            "selected_cfg": bo_cfg,
            "dev_grid": bo_dev_grid,
            "test": relative_gain(bm25_orca_test, dense_test),
        },
    }

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, default="../data/cleaned/psydt_task_retrieval_final.json")
    parser.add_argument("--models_root", type=str, default="../models")
    parser.add_argument("--gte_path", type=str, default=None)
    parser.add_argument("--bge_path", type=str, default=None)
    parser.add_argument("--models", type=str, default="gte,bge")

    parser.add_argument("--output_json", type=str, default="../results/orca_rrf_fusion.json")
    parser.add_argument("--output_md", type=str, default="../results/orca_rrf_fusion_summary.md")
    parser.add_argument("--cache_dir", type=str, default="../cache/embeddings_final")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gte_batch_size", type=int, default=16)
    parser.add_argument("--bge_batch_size", type=int, default=32)
    parser.add_argument("--no_cache", action="store_true")

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--fusion_depth", type=int, default=1000)
    parser.add_argument("--selection_metric", type=str, default="R@5")

    parser.add_argument("--k_list", type=str, default="2,4,8,16,32,64")
    parser.add_argument("--normalize_for_svd", action="store_true")
    parser.add_argument("--center_for_svd", action="store_true")

    parser.add_argument("--bm25_tokenizer", type=str, default="auto", choices=["auto", "jieba", "char"])
    parser.add_argument("--bm25_k1", type=float, default=1.5)
    parser.add_argument("--bm25_b", type=float, default=0.75)

    # RRF tuning grid
    parser.add_argument("--rrf_k_list", type=str, default="10,30,60,100")
    parser.add_argument("--dense_weight_list", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--orca_weight_list", type=str, default="0.5,1.0,1.5,2.0")
    parser.add_argument("--bm25_weight_list", type=str, default="0.5,1.0,1.5,2.0")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 100)
    print("[ORCA + BM25 RRF Fusion Experiments]")
    print(f"Start time: {now_time()}")
    print("=" * 100)

    data = load_json(args.data)
    required = [
        "emotion_corpus",
        "dev_task_corpus", "dev_queries", "dev_gt",
        "test_task_corpus", "test_queries", "test_gt",
    ]
    missing = [x for x in required if x not in data]
    if missing:
        raise KeyError(f"Dataset missing fields: {missing}")

    print("[Dataset]")
    print(f"Path: {args.data}")
    if "meta" in data and "counts" in data["meta"]:
        for k, v in data["meta"]["counts"].items():
            print(f"  {k}: {v}")

    tokenizer = make_tokenizer(args.bm25_tokenizer)

    print("\n" + "=" * 100)
    print("[BM25 dev]")
    print("=" * 100)
    bm25_dev_metrics, bm25_dev_idx, bm25_dev_scores = run_bm25(
        docs=data["dev_task_corpus"],
        queries=data["dev_queries"],
        gt=data["dev_gt"],
        tokenizer=tokenizer,
        top_k=args.top_k,
        depth=args.fusion_depth,
        k1=args.bm25_k1,
        b=args.bm25_b,
    )
    print_metrics("[BM25][dev]", bm25_dev_metrics)

    print("\n" + "=" * 100)
    print("[BM25 test]")
    print("=" * 100)
    bm25_test_metrics, bm25_test_idx, bm25_test_scores = run_bm25(
        docs=data["test_task_corpus"],
        queries=data["test_queries"],
        gt=data["test_gt"],
        tokenizer=tokenizer,
        top_k=args.top_k,
        depth=args.fusion_depth,
        k1=args.bm25_k1,
        b=args.bm25_b,
    )
    print_metrics("[BM25][test]", bm25_test_metrics)

    bm25_dev = {
        "metrics": bm25_dev_metrics,
        "indices": bm25_dev_idx,
        "scores": bm25_dev_scores,
    }
    bm25_test = {
        "metrics": bm25_test_metrics,
        "indices": bm25_test_idx,
        "scores": bm25_test_scores,
    }

    all_model_paths = resolve_model_paths(
        models_root=args.models_root,
        gte_path=args.gte_path,
        bge_path=args.bge_path,
    )

    requested = [x.strip() for x in args.models.split(",") if x.strip()]
    model_paths = {}
    for key in requested:
        if key not in all_model_paths:
            raise KeyError(f"Requested model '{key}' not found. Available: {list(all_model_paths.keys())}")
        model_paths[key] = all_model_paths[key]

    print("\n[Resolved models]")
    for k, p in model_paths.items():
        print(f"  {k}: {p}")

    results = {
        "created_at": now_time(),
        "data_path": args.data,
        "dataset_meta": data.get("meta", {}),
        "global_config": {
            "models": requested,
            "top_k": args.top_k,
            "fusion_depth": args.fusion_depth,
            "selection_metric": args.selection_metric,
            "k_list": parse_int_list(args.k_list),
            "rrf_k_list": parse_int_list(args.rrf_k_list),
            "dense_weight_list": parse_float_list(args.dense_weight_list),
            "orca_weight_list": parse_float_list(args.orca_weight_list),
            "bm25_weight_list": parse_float_list(args.bm25_weight_list),
            "bm25_tokenizer": args.bm25_tokenizer,
            "bm25_k1": args.bm25_k1,
            "bm25_b": args.bm25_b,
            "seed": args.seed,
        },
        "bm25_model_independent": {
            "dev": bm25_dev_metrics,
            "test": bm25_test_metrics,
        },
        "models": {},
    }

    for model_key, model_path in model_paths.items():
        model_result = run_for_one_model(
            model_key=model_key,
            model_path=model_path,
            data=data,
            bm25_dev=bm25_dev,
            bm25_test=bm25_test,
            args=args,
        )
        results["models"][model_key] = model_result
        save_json(results, args.output_json)
        print(f"[Intermediate saved] {args.output_json}")

    md = make_markdown(results)
    ensure_dir_for_file(args.output_md)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)

    save_json(results, args.output_json)

    print("\n" + "=" * 100)
    print("[Done]")
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_md}")
    print("=" * 100)


if __name__ == "__main__":
    main()
