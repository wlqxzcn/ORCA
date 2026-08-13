import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


def ensure_dir_for_file(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def now_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
        "bge-large-zh",
        "bge-base-zh",
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
    gte_path: Optional[str],
    bge_path: Optional[str],
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
        print(f"[Warning] gte model not found: {p}")

    if bge_path:
        p = Path(bge_path).resolve()
    else:
        p = find_bge_model(root)

    if p is not None and looks_like_model_dir(p):
        out["bge"] = p.resolve()
    else:
        print(f"[Warning] BGE model not found under: {root / 'BAAI'}")

    if not out:
        raise FileNotFoundError(
            f"No valid model directory found under {root}. "
            f"Please specify --gte_path or --bge_path."
        )

    return out


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    cache_path: Optional[str],
    normalize_embeddings: bool = False,
) -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            print(f"[Cache] loaded {cache_path} {arr.shape}")
            return arr.astype("float32")
        print(
            f"[Cache warning] shape mismatch for {cache_path}: "
            f"cache={arr.shape[0]}, expected={len(texts)}. Recompute."
        )

    print(f"[Encode] {len(texts)} texts")
    arr = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    ).astype("float32")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, arr)
        print(f"[Cache] saved {cache_path} {arr.shape}")

    return arr


def search_ip(
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
    latency_ms = (time.perf_counter() - start) * 1000.0 / max(len(q), 1)
    return scores, indices, latency_ms


def evaluate(indices: np.ndarray, gt: List[List[int]]) -> Dict[str, float]:
    n = len(gt)
    if n == 0:
        raise ValueError("Empty ground truth.")

    hit1 = hit5 = hit10 = 0
    mrr = 0.0
    ndcg10 = 0.0

    for i in range(n):
        gold = set(int(x) for x in gt[i])
        rank_list = [int(x) for x in indices[i].tolist()]

        if any(x in gold for x in rank_list[:1]):
            hit1 += 1
        if any(x in gold for x in rank_list[:5]):
            hit5 += 1
        if any(x in gold for x in rank_list[:10]):
            hit10 += 1

        found_rank = None
        for r, doc_id in enumerate(rank_list, start=1):
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


def hits_at_k(indices: np.ndarray, gt: List[List[int]], k: int) -> np.ndarray:
    hits = np.zeros(len(gt), dtype=np.bool_)
    for i in range(len(gt)):
        gold = set(int(x) for x in gt[i])
        pred = [int(x) for x in indices[i, :k].tolist()]
        hits[i] = any(x in gold for x in pred)
    return hits


def average_jaccard(indices_a: np.ndarray, indices_b: np.ndarray, k: int = 5) -> float:
    vals = []
    for i in range(indices_a.shape[0]):
        a = set(int(x) for x in indices_a[i, :k].tolist())
        b = set(int(x) for x in indices_b[i, :k].tolist())
        if len(a | b) == 0:
            vals.append(1.0)
        else:
            vals.append(len(a & b) / len(a | b))
    return float(np.mean(vals))


def print_metrics(prefix: str, metrics: Dict[str, float]):
    keys = ["R@1", "R@5", "R@10", "MRR", "nDCG@10", "latency_ms_per_query"]
    parts = []
    for k in keys:
        if k in metrics:
            parts.append(f"{k}={metrics[k]:.4f}")
    print(f"{prefix}: " + " | ".join(parts))



def fit_svd_full_basis(
    emotion_embs: np.ndarray,
    normalize_for_svd: bool = False,
    center_for_svd: bool = False,
    use_float64_svd: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    E = torch.from_numpy(emotion_embs.astype("float32"))

    if normalize_for_svd:
        E = F.normalize(E, p=2, dim=1)

    if center_for_svd:
        E = E - E.mean(dim=0, keepdim=True)

    if use_float64_svd:
        E = E.double()

    print(
        f"[SVD] E={tuple(E.shape)}, normalize_for_svd={normalize_for_svd}, "
        f"center_for_svd={center_for_svd}"
    )

    with torch.no_grad():
        _, S, Vh = torch.linalg.svd(E, full_matrices=False)
        B = Vh.T.contiguous()
        B, _ = torch.linalg.qr(B, mode="reduced")

    return B.float().cpu().numpy().astype("float32"), S.float().cpu().numpy().astype("float32")


def basis_at_k(B_full: np.ndarray, k: int) -> np.ndarray:
    k = min(int(k), B_full.shape[1])
    return B_full[:, :k].astype("float32")


def project_remove(x: np.ndarray, B: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    B = B.astype("float32")
    return (x - (x @ B) @ B.T).astype("float32")


def project_keep(x: np.ndarray, B: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    B = B.astype("float32")
    return ((x @ B) @ B.T).astype("float32")


def run_retrieval(
    corpus_embs: np.ndarray,
    query_embs: np.ndarray,
    gt: List[List[int]],
    top_k: int,
    method: str,
    B: Optional[np.ndarray] = None,
    mode: str = "dense",
) -> Tuple[Dict[str, float], np.ndarray]:
    if mode == "dense":
        c = corpus_embs
        q = query_embs
    elif mode == "symmetric":
        if B is None:
            raise ValueError("B is required for projected retrieval.")
        c = project_remove(corpus_embs, B)
        q = project_remove(query_embs, B)
    elif mode == "doc_only":
        if B is None:
            raise ValueError("B is required for projected retrieval.")
        c = project_remove(corpus_embs, B)
        q = query_embs
    elif mode == "query_only":
        if B is None:
            raise ValueError("B is required for projected retrieval.")
        c = corpus_embs
        q = project_remove(query_embs, B)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _, indices, latency = search_ip(q, c, top_k=top_k, normalize=True)
    metrics = evaluate(indices, gt)
    metrics["latency_ms_per_query"] = latency
    return metrics, indices


def orthogonality_error(x: np.ndarray, B: np.ndarray) -> float:
    task = project_remove(x, B)
    emo = project_keep(x, B)
    t = torch.from_numpy(task)
    e = torch.from_numpy(emo)
    t = F.normalize(t, p=2, dim=1)
    e = F.normalize(e, p=2, dim=1)
    return torch.abs(torch.sum(t * e, dim=1)).mean().item()


def projection_errors(B: np.ndarray) -> Dict[str, float]:
    B_t = torch.from_numpy(B.astype("float64"))
    D = B_t.shape[0]
    I = torch.eye(D, dtype=torch.float64)
    P = I - B_t @ B_t.T
    return {
        "idempotence_error_fro": torch.linalg.norm(P @ P - P, ord="fro").item(),
        "symmetry_error_fro": torch.linalg.norm(P - P.T, ord="fro").item(),
    }


def energy_threshold_to_k(S: np.ndarray, tau: float) -> int:
    s2 = S.astype("float64") ** 2
    cum = np.cumsum(s2) / max(float(np.sum(s2)), 1e-12)
    return int(np.searchsorted(cum, tau, side="left") + 1)



def exact_binomial_two_sided(k: int, n: int) -> float:
    if n <= 0:
        return 1.0

    k = min(k, n - k)
    log2 = math.log(2.0)
    probs = []
    for i in range(k + 1):
        logp = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * log2
        probs.append(math.exp(logp))
    return float(min(1.0, 2.0 * sum(probs)))


def mcnemar_test(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    name_a: str = "Dense",
    name_b: str = "ORCA",
) -> Dict[str, Any]:
    hits_a = hits_a.astype(bool)
    hits_b = hits_b.astype(bool)

    both_correct = int(np.sum(hits_a & hits_b))
    both_wrong = int(np.sum(~hits_a & ~hits_b))
    a_only = int(np.sum(hits_a & ~hits_b))
    b_only = int(np.sum(~hits_a & hits_b))
    discordant = a_only + b_only

    if discordant == 0:
        chi2 = 0.0
        p_approx = 1.0
        p_exact = 1.0
    else:
        chi2 = (max(abs(a_only - b_only) - 1, 0) ** 2) / discordant
        # chi-square df=1 survival function = erfc(sqrt(x/2))
        p_approx = math.erfc(math.sqrt(chi2 / 2.0))
        p_exact = exact_binomial_two_sided(min(a_only, b_only), discordant)

    return {
        "name_a": name_a,
        "name_b": name_b,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        f"{name_a}_only": a_only,
        f"{name_b}_only": b_only,
        "discordant": discordant,
        "mcnemar_chi2_cc": chi2,
        "p_value_chi2_approx": p_approx,
        "p_value_exact_binomial": p_exact,
    }


def bootstrap_paired_difference(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    n_bootstrap: int = 5000,
    seed: int = 42,
    ci: float = 0.95,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    hits_a = hits_a.astype(np.float32)
    hits_b = hits_b.astype(np.float32)
    n = len(hits_a)

    base_diff = float(np.mean(hits_b - hits_a))
    diffs = np.empty(n_bootstrap, dtype=np.float32)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(hits_b[idx] - hits_a[idx]))

    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(diffs, alpha))
    hi = float(np.quantile(diffs, 1.0 - alpha))

    return {
        "diff_B_minus_A": base_diff,
        "ci": ci,
        "ci_low": lo,
        "ci_high": hi,
        "n_bootstrap": int(n_bootstrap),
    }


def significance_suite(
    dense_indices: np.ndarray,
    orca_indices: np.ndarray,
    gt: List[List[int]],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    out = {}
    for k in [1, 5, 10]:
        hd = hits_at_k(dense_indices, gt, k)
        hn = hits_at_k(orca_indices, gt, k)
        out[f"R@{k}"] = {
            "mcnemar": mcnemar_test(hd, hn, name_a="Dense", name_b="ORCA"),
            "bootstrap": bootstrap_paired_difference(
                hd, hn, n_bootstrap=n_bootstrap, seed=seed + k, ci=0.95
            ),
        }
    return out


def select_k_on_dev(
    B_full: np.ndarray,
    k_list: List[int],
    dev_corpus_embs: np.ndarray,
    dev_query_embs: np.ndarray,
    dev_gt: List[List[int]],
    top_k: int,
    selection_metric: str,
) -> Tuple[int, Dict[str, Dict[str, float]]]:
    results = {}
    best_k = None
    best_score = -1.0

    print("\n[Dev k sweep]")
    for k in k_list:
        B = basis_at_k(B_full, k)
        metrics, _ = run_retrieval(
            dev_corpus_embs,
            dev_query_embs,
            dev_gt,
            top_k=top_k,
            method="ORCA",
            B=B,
            mode="symmetric",
        )
        results[str(k)] = metrics
        print_metrics(f"[k={k}]", metrics)

        score = float(metrics[selection_metric])
        if score > best_score:
            best_score = score
            best_k = k

    print(f"[Selected k] {best_k} by {selection_metric}={best_score:.4f}")
    return int(best_k), results


def run_energy_threshold_scan(
    B_full: np.ndarray,
    S: np.ndarray,
    thresholds: List[float],
    dev_corpus_embs: np.ndarray,
    dev_query_embs: np.ndarray,
    dev_gt: List[List[int]],
    test_corpus_embs: np.ndarray,
    test_query_embs: np.ndarray,
    test_gt: List[List[int]],
    top_k: int,
) -> Dict[str, Any]:
    print("\n[Energy threshold scan]")
    out = {}

    for tau in thresholds:
        k = energy_threshold_to_k(S, tau)
        k = min(k, B_full.shape[1])
        B = basis_at_k(B_full, k)

        dev_metrics, _ = run_retrieval(
            dev_corpus_embs, dev_query_embs, dev_gt,
            top_k=top_k, method="ORCA", B=B, mode="symmetric"
        )
        test_metrics, _ = run_retrieval(
            test_corpus_embs, test_query_embs, test_gt,
            top_k=top_k, method="ORCA", B=B, mode="symmetric"
        )

        out[str(tau)] = {
            "k": int(k),
            "dev": dev_metrics,
            "test": test_metrics,
        }

        print(
            f"[tau={tau:.2f}] k={k} | "
            f"dev R@5={dev_metrics['R@5']:.4f} | "
            f"test R@5={test_metrics['R@5']:.4f}"
        )

    return out


def make_noisy_queries(
    queries: List[str],
    noise_pool: List[str],
    level: int,
    seed: int,
) -> List[str]:
    if level <= 0:
        return list(queries)

    if not noise_pool:
        raise ValueError("noise_pool is empty; cannot inject noise.")

    rng = random.Random(seed + 1009 * level)
    out = []

    for i, q in enumerate(queries):
        noises = [rng.choice(noise_pool) for _ in range(level)]
        noisy = q + "。" + "。".join(noises)
        out.append(noisy)

    return out


def run_noise_robustness(
    model: SentenceTransformer,
    model_key: str,
    batch_size: int,
    cache_dir: Path,
    queries: List[str],
    corpus_embs: np.ndarray,
    corpus_gt: List[List[int]],
    B: np.ndarray,
    clean_query_embs: np.ndarray,
    clean_dense_indices: np.ndarray,
    clean_orca_indices: np.ndarray,
    noise_pool: List[str],
    noise_levels: List[int],
    top_k: int,
    seed: int,
    no_cache: bool,
) -> Dict[str, Any]:
    print("\n[Noise robustness]")
    out = {}

    for level in noise_levels:
        if level == 0:
            noisy_queries = list(queries)
            noisy_embs = clean_query_embs
        else:
            noisy_queries = make_noisy_queries(
                queries=queries,
                noise_pool=noise_pool,
                level=level,
                seed=seed,
            )
            cache_path = None
            if not no_cache:
                cache_path = str(cache_dir / f"noise_level_{level}.npy")
            noisy_embs = encode_texts(
                model=model,
                texts=noisy_queries,
                batch_size=batch_size,
                cache_path=cache_path,
                normalize_embeddings=False,
            )

        dense_metrics, dense_indices = run_retrieval(
            corpus_embs,
            noisy_embs,
            corpus_gt,
            top_k=top_k,
            method="Dense",
            B=None,
            mode="dense",
        )

        orca_metrics, orca_indices = run_retrieval(
            corpus_embs,
            noisy_embs,
            corpus_gt,
            top_k=top_k,
            method="ORCA",
            B=B,
            mode="symmetric",
        )

        dense_jaccard = average_jaccard(clean_dense_indices, dense_indices, k=5)
        orca_jaccard = average_jaccard(clean_orca_indices, orca_indices, k=5)

        out[str(level)] = {
            "dense": dense_metrics,
            "orca": orca_metrics,
            "dense_top5_jaccard_vs_clean": dense_jaccard,
            "orca_top5_jaccard_vs_clean": orca_jaccard,
            "relative_improvement_R@5": (
                None if dense_metrics["R@5"] <= 0
                else (orca_metrics["R@5"] - dense_metrics["R@5"]) / dense_metrics["R@5"]
            ),
        }

        print(
            f"[noise={level}] Dense R@5={dense_metrics['R@5']:.4f} | "
            f"ORCA R@5={orca_metrics['R@5']:.4f} | "
            f"rel={out[str(level)]['relative_improvement_R@5'] * 100 if out[str(level)]['relative_improvement_R@5'] is not None else 0:.2f}% | "
            f"Jaccard Dense={dense_jaccard:.4f}, ORCA={orca_jaccard:.4f}"
        )

    return out


def run_svd_normalization_ablation(
    emotion_embs: np.ndarray,
    dev_corpus_embs: np.ndarray,
    dev_query_embs: np.ndarray,
    dev_gt: List[List[int]],
    test_corpus_embs: np.ndarray,
    test_query_embs: np.ndarray,
    test_gt: List[List[int]],
    k_list: List[int],
    top_k: int,
    selection_metric: str,
) -> Dict[str, Any]:
    print("\n[SVD normalization ablation]")
    out = {}

    modes = {
        "raw_svd": {
            "normalize_for_svd": False,
            "center_for_svd": False,
        },
        "normalized_svd": {
            "normalize_for_svd": True,
            "center_for_svd": False,
        },
    }

    for name, cfg in modes.items():
        print(f"\n[SVD mode] {name}")
        B_full, S = fit_svd_full_basis(
            emotion_embs,
            normalize_for_svd=cfg["normalize_for_svd"],
            center_for_svd=cfg["center_for_svd"],
            use_float64_svd=True,
        )

        selected_k, dev_sweep = select_k_on_dev(
            B_full=B_full,
            k_list=k_list,
            dev_corpus_embs=dev_corpus_embs,
            dev_query_embs=dev_query_embs,
            dev_gt=dev_gt,
            top_k=top_k,
            selection_metric=selection_metric,
        )

        B = basis_at_k(B_full, selected_k)
        test_metrics, _ = run_retrieval(
            test_corpus_embs,
            test_query_embs,
            test_gt,
            top_k=top_k,
            method="ORCA",
            B=B,
            mode="symmetric",
        )

        out[name] = {
            "config": cfg,
            "selected_k": int(selected_k),
            "dev_sweep": dev_sweep,
            "test": test_metrics,
        }

        print_metrics(f"[{name}][test][k={selected_k}]", test_metrics)

    return out


def run_emotion_track_examples(
    query_texts: List[str],
    query_embs: np.ndarray,
    mem_texts: List[str],
    mem_embs: np.ndarray,
    B: np.ndarray,
    n_examples: int,
    top_k: int,
    seed: int,
) -> List[Dict[str, Any]]:

    if not mem_texts or mem_embs is None or len(mem_texts) == 0:
        return []

    rng = random.Random(seed)
    n = len(query_texts)
    idxs = list(range(n))
    rng.shuffle(idxs)
    idxs = idxs[:min(n_examples, n)]

    q_emo = project_keep(query_embs, B)
    m_emo = project_keep(mem_embs, B)

    q = q_emo.copy().astype("float32")
    m = m_emo.copy().astype("float32")
    faiss.normalize_L2(q)
    faiss.normalize_L2(m)

    examples = []
    for idx in idxs:
        sims = m @ q[idx]
        order = np.argsort(-sims)[:top_k]
        examples.append({
            "query_id": int(idx),
            "query": query_texts[idx],
            "emotion_memory_topk": [
                {
                    "rank": int(r + 1),
                    "mem_id": int(j),
                    "score": float(sims[j]),
                    "text": mem_texts[j],
                }
                for r, j in enumerate(order)
            ],
        })

    return examples



def md_metrics_row(name: str, m: Dict[str, float]) -> str:
    return (
        f"| {name} | {m['R@1']:.4f} | {m['R@5']:.4f} | "
        f"{m['R@10']:.4f} | {m['MRR']:.4f} | {m['nDCG@10']:.4f} |"
    )


def make_markdown(results: Dict[str, Any]) -> str:
    lines = []
    lines.append("# ORCA Supplementary Experiments Summary\n")
    lines.append(f"Generated at: {results.get('created_at', '')}\n")

    for model_key, r in results["models"].items():
        lines.append(f"## {model_key}\n")
        lines.append(f"- Model path: `{r['model_path']}`")
        lines.append(f"- Selected k: **{r['selected_k']}**")
        lines.append(f"- Dense test R@5: **{r['base']['dense_test']['R@5']:.4f}**")
        lines.append(f"- ORCA test R@5: **{r['base']['orca_test']['R@5']:.4f}**")
        rel = r["base"]["relative_improvement_R@5"]
        lines.append(f"- Relative R@5 improvement: **{rel * 100:.2f}%**")
        lines.append("")

        lines.append("### Main test metrics\n")
        lines.append("| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append(md_metrics_row("Dense", r["base"]["dense_test"]))
        lines.append(md_metrics_row("ORCA Symmetric", r["base"]["orca_test"]))
        lines.append(md_metrics_row("ORCA Doc-only", r["base"]["doc_only_test"]))
        lines.append(md_metrics_row("ORCA Query-only", r["base"]["query_only_test"]))
        lines.append("")

        lines.append("### Significance: Dense vs ORCA\n")
        lines.append("| Metric | Dense-only | ORCA-only | Exact p | Bootstrap diff | 95% CI |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for metric_name, sig in r["significance"].items():
            mc = sig["mcnemar"]
            bs = sig["bootstrap"]
            lines.append(
                f"| {metric_name} | {mc['Dense_only']} | {mc['ORCA_only']} | "
                f"{mc['p_value_exact_binomial']:.6g} | {bs['diff_B_minus_A']:.4f} | "
                f"[{bs['ci_low']:.4f}, {bs['ci_high']:.4f}] |"
            )
        lines.append("")

        lines.append("### Dev k sweep\n")
        lines.append("| k | R@1 | R@5 | R@10 | MRR | nDCG@10 |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for k_str, m in r["dev_k_sweep"].items():
            lines.append(
                f"| {k_str} | {m['R@1']:.4f} | {m['R@5']:.4f} | "
                f"{m['R@10']:.4f} | {m['MRR']:.4f} | {m['nDCG@10']:.4f} |"
            )
        lines.append("")

        if "energy_threshold_scan" in r:
            lines.append("### Energy threshold scan\n")
            lines.append("| τ | k | Dev R@5 | Test R@5 | Test R@10 |")
            lines.append("|---:|---:|---:|---:|---:|")
            for tau, item in r["energy_threshold_scan"].items():
                lines.append(
                    f"| {tau} | {item['k']} | {item['dev']['R@5']:.4f} | "
                    f"{item['test']['R@5']:.4f} | {item['test']['R@10']:.4f} |"
                )
            lines.append("")

        if "noise_robustness" in r:
            lines.append("### Noise robustness\n")
            lines.append("| Noise level | Dense R@5 | ORCA R@5 | Relative gain | Dense Jaccard | ORCA Jaccard |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for level, item in r["noise_robustness"].items():
                rel = item["relative_improvement_R@5"]
                rel_txt = "" if rel is None else f"{rel * 100:.2f}%"
                lines.append(
                    f"| {level} | {item['dense']['R@5']:.4f} | {item['orca']['R@5']:.4f} | "
                    f"{rel_txt} | {item['dense_top5_jaccard_vs_clean']:.4f} | "
                    f"{item['orca_top5_jaccard_vs_clean']:.4f} |"
                )
            lines.append("")

        if "svd_normalization_ablation" in r:
            lines.append("### SVD normalization ablation\n")
            lines.append("| SVD mode | Selected k | Test R@5 | Test R@10 | MRR |")
            lines.append("|---|---:|---:|---:|---:|")
            for name, item in r["svd_normalization_ablation"].items():
                m = item["test"]
                lines.append(
                    f"| {name} | {item['selected_k']} | {m['R@5']:.4f} | "
                    f"{m['R@10']:.4f} | {m['MRR']:.4f} |"
                )
            lines.append("")

        if "emotion_track_examples" in r:
            lines.append("### Emotion track examples\n")
            for ex in r["emotion_track_examples"][:5]:
                lines.append(f"- Query: {ex['query']}")
                for mem in ex["emotion_memory_topk"]:
                    lines.append(
                        f"  - Top {mem['rank']}: {mem['text']} "
                        f"(score={mem['score']:.4f})"
                    )
            lines.append("")

    return "\n".join(lines)


def run_for_one_model(
    model_key: str,
    model_path: Path,
    data: Dict[str, Any],
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

    emotion_embs = encode_texts(
        model, data["emotion_corpus"], batch_size, cache("emotion_corpus")
    )
    dev_corpus_embs = encode_texts(
        model, data["dev_task_corpus"], batch_size, cache("dev_task_corpus")
    )
    dev_query_embs = encode_texts(
        model, data["dev_queries"], batch_size, cache("dev_queries")
    )
    test_corpus_embs = encode_texts(
        model, data["test_task_corpus"], batch_size, cache("test_task_corpus")
    )
    test_query_embs = encode_texts(
        model, data["test_queries"], batch_size, cache("test_queries")
    )

    mem_embs = None
    if "mem_corpus" in data and len(data["mem_corpus"]) > 0:
        mem_embs = encode_texts(
            model, data["mem_corpus"], batch_size, cache("mem_corpus")
        )

    k_list = parse_int_list(args.k_list)
    noise_levels = parse_int_list(args.noise_levels)
    thresholds = parse_float_list(args.energy_thresholds)

    B_full, S = fit_svd_full_basis(
        emotion_embs,
        normalize_for_svd=args.normalize_for_svd,
        center_for_svd=args.center_for_svd,
        use_float64_svd=True,
    )

    print("\n[Base retrieval]")
    dense_dev, dense_dev_indices = run_retrieval(
        dev_corpus_embs, dev_query_embs, data["dev_gt"],
        top_k=args.top_k, method="Dense", mode="dense"
    )
    print_metrics("[Dense dev]", dense_dev)

    dense_test, dense_test_indices = run_retrieval(
        test_corpus_embs, test_query_embs, data["test_gt"],
        top_k=args.top_k, method="Dense", mode="dense"
    )
    print_metrics("[Dense test]", dense_test)

    selected_k, dev_k_sweep = select_k_on_dev(
        B_full=B_full,
        k_list=k_list,
        dev_corpus_embs=dev_corpus_embs,
        dev_query_embs=dev_query_embs,
        dev_gt=data["dev_gt"],
        top_k=args.top_k,
        selection_metric=args.selection_metric,
    )

    B = basis_at_k(B_full, selected_k)


    orca_test, orca_test_indices = run_retrieval(
        test_corpus_embs, test_query_embs, data["test_gt"],
        top_k=args.top_k, method="ORCA", B=B, mode="symmetric"
    )
    print_metrics("[ORCA symmetric test]", orca_test)

    doc_only_test, doc_only_indices = run_retrieval(
        test_corpus_embs, test_query_embs, data["test_gt"],
        top_k=args.top_k, method="ORCA", B=B, mode="doc_only"
    )
    print_metrics("[ORCA doc-only test]", doc_only_test)

    query_only_test, query_only_indices = run_retrieval(
        test_corpus_embs, test_query_embs, data["test_gt"],
        top_k=args.top_k, method="ORCA", B=B, mode="query_only"
    )
    print_metrics("[ORCA query-only test]", query_only_test)


    diag = {
        "orthogonality_cosine_error_test_queries": orthogonality_error(test_query_embs, B),
    }
    diag.update(projection_errors(B))

    print("\n[Significance]")
    significance = significance_suite(
        dense_indices=dense_test_indices,
        orca_indices=orca_test_indices,
        gt=data["test_gt"],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    for metric_name, sig in significance.items():
        mc = sig["mcnemar"]
        bs = sig["bootstrap"]
        print(
            f"[{metric_name}] Dense-only={mc['Dense_only']}, ORCA-only={mc['ORCA_only']}, "
            f"p_exact={mc['p_value_exact_binomial']:.6g}, "
            f"diff={bs['diff_B_minus_A']:.4f}, "
            f"CI=[{bs['ci_low']:.4f}, {bs['ci_high']:.4f}]"
        )

    rel = None if dense_test["R@5"] <= 0 else (orca_test["R@5"] - dense_test["R@5"]) / dense_test["R@5"]

    result = {
        "model_key": model_key,
        "model_path": str(model_path),
        "embedding_dim": int(emotion_embs.shape[1]),
        "config": {
            "k_list": k_list,
            "selected_k": selected_k,
            "top_k": args.top_k,
            "selection_metric": args.selection_metric,
            "normalize_for_svd": args.normalize_for_svd,
            "center_for_svd": args.center_for_svd,
        },
        "base": {
            "dense_dev": dense_dev,
            "dense_test": dense_test,
            "orca_test": orca_test,
            "doc_only_test": doc_only_test,
            "query_only_test": query_only_test,
            "relative_improvement_R@5": rel,
            "diagnostics": diag,
        },
        "selected_k": int(selected_k),
        "dev_k_sweep": dev_k_sweep,
        "significance": significance,
    }

    if not args.skip_energy_threshold:
        result["energy_threshold_scan"] = run_energy_threshold_scan(
            B_full=B_full,
            S=S,
            thresholds=thresholds,
            dev_corpus_embs=dev_corpus_embs,
            dev_query_embs=dev_query_embs,
            dev_gt=data["dev_gt"],
            test_corpus_embs=test_corpus_embs,
            test_query_embs=test_query_embs,
            test_gt=data["test_gt"],
            top_k=args.top_k,
        )

    if not args.skip_noise:
        noise_cache = cache_root / "noise_queries"
        noise_cache.mkdir(parents=True, exist_ok=True)
        result["noise_robustness"] = run_noise_robustness(
            model=model,
            model_key=model_key,
            batch_size=batch_size,
            cache_dir=noise_cache,
            queries=data["test_queries"],
            corpus_embs=test_corpus_embs,
            corpus_gt=data["test_gt"],
            B=B,
            clean_query_embs=test_query_embs,
            clean_dense_indices=dense_test_indices,
            clean_orca_indices=orca_test_indices,
            noise_pool=data["emotion_corpus"],
            noise_levels=noise_levels,
            top_k=args.top_k,
            seed=args.seed,
            no_cache=args.no_cache,
        )

    if not args.skip_svd_norm_ablation:
        result["svd_normalization_ablation"] = run_svd_normalization_ablation(
            emotion_embs=emotion_embs,
            dev_corpus_embs=dev_corpus_embs,
            dev_query_embs=dev_query_embs,
            dev_gt=data["dev_gt"],
            test_corpus_embs=test_corpus_embs,
            test_query_embs=test_query_embs,
            test_gt=data["test_gt"],
            k_list=k_list,
            top_k=args.top_k,
            selection_metric=args.selection_metric,
        )

    if not args.skip_emotion_examples and mem_embs is not None:
        result["emotion_track_examples"] = run_emotion_track_examples(
            query_texts=data["test_queries"],
            query_embs=test_query_embs,
            mem_texts=data["mem_corpus"],
            mem_embs=mem_embs,
            B=B,
            n_examples=args.emotion_examples,
            top_k=args.emotion_top_k,
            seed=args.seed,
        )

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

    parser.add_argument("--output_json", type=str, default="../results/orca_supplementary_experiments.json")
    parser.add_argument("--output_md", type=str, default="../results/orca_supplementary_experiments_summary.md")
    parser.add_argument("--cache_dir", type=str, default="../cache/embeddings_final")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gte_batch_size", type=int, default=16)
    parser.add_argument("--bge_batch_size", type=int, default=32)

    parser.add_argument("--k_list", type=str, default="2,4,8,16,32,64")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--selection_metric", type=str, default="R@5")

    parser.add_argument("--normalize_for_svd", action="store_true")
    parser.add_argument("--center_for_svd", action="store_true")

    parser.add_argument("--energy_thresholds", type=str, default="0.50,0.70,0.80,0.90,0.95,0.99")
    parser.add_argument("--noise_levels", type=str, default="0,1,3,5")
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--emotion_examples", type=int, default=10)
    parser.add_argument("--emotion_top_k", type=int, default=3)

    parser.add_argument("--skip_energy_threshold", action="store_true")
    parser.add_argument("--skip_noise", action="store_true")
    parser.add_argument("--skip_svd_norm_ablation", action="store_true")
    parser.add_argument("--skip_emotion_examples", action="store_true")
    parser.add_argument("--no_cache", action="store_true")

    args = parser.parse_args()

    print("=" * 100)
    print("[ORCA supplementary experiments]")
    print(f"Start time: {now_time()}")
    print("=" * 100)

    data = load_json(args.data)
    required = [
        "emotion_corpus",
        "dev_task_corpus", "dev_queries", "dev_gt",
        "test_task_corpus", "test_queries", "test_gt",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Dataset is missing required fields: {missing}")

    print("[Dataset]")
    print(f"Path: {args.data}")
    if "meta" in data and "counts" in data["meta"]:
        for k, v in data["meta"]["counts"].items():
            print(f"  {k}: {v}")

    all_paths = resolve_model_paths(
        models_root=args.models_root,
        gte_path=args.gte_path,
        bge_path=args.bge_path,
    )

    requested = [x.strip() for x in args.models.split(",") if x.strip()]
    model_paths = {}
    for key in requested:
        if key not in all_paths:
            raise KeyError(f"Requested model '{key}' not found. Available: {list(all_paths.keys())}")
        model_paths[key] = all_paths[key]

    print("\n[Resolved models]")
    for k, p in model_paths.items():
        print(f"  {k}: {p}")

    results = {
        "created_at": now_time(),
        "data_path": args.data,
        "dataset_meta": data.get("meta", {}),
        "global_config": {
            "models_root": args.models_root,
            "models": requested,
            "device": args.device,
            "k_list": parse_int_list(args.k_list),
            "top_k": args.top_k,
            "selection_metric": args.selection_metric,
            "normalize_for_svd": args.normalize_for_svd,
            "center_for_svd": args.center_for_svd,
            "energy_thresholds": parse_float_list(args.energy_thresholds),
            "noise_levels": parse_int_list(args.noise_levels),
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
        },
        "models": {},
    }

    for model_key, model_path in model_paths.items():
        model_result = run_for_one_model(model_key, model_path, data, args)
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
