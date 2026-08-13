import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer



def ensure_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_name(name: str) -> str:
    name = str(name)
    for ch in ["/", "\\", ":", " ", "\t", "\n"]:
        name = name.replace(ch, "_")
    return name


def parse_k_list(k_list: str) -> List[int]:
    out = []
    for x in k_list.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def now_time() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")



def looks_like_sentence_transformer(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (path / "modules.json").exists() or (path / "config.json").exists()


def find_bge_model(models_root: Path) -> Optional[Path]:
    baai_dir = models_root / "BAAI"

    if looks_like_sentence_transformer(baai_dir):
        return baai_dir

    if not baai_dir.exists():
        return None

    preferred_names = [
        "bge-large-zh-v1.5",
        "bge-base-zh-v1.5",
        "bge-small-zh-v1.5",
        "bge-large-zh",
        "bge-base-zh",
    ]

    for name in preferred_names:
        p = baai_dir / name
        if looks_like_sentence_transformer(p):
            return p

    candidates = []
    for p in baai_dir.rglob("*"):
        if p.is_dir() and looks_like_sentence_transformer(p):
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
    paths = {}

    if gte_path:
        p = Path(gte_path).resolve()
    else:
        p = (root / "thenlper" / "gte-large-zh").resolve()

    if looks_like_sentence_transformer(p):
        paths["gte"] = p
    else:
        print(f"[Warning] gte path not found or not a model dir: {p}")

    if bge_path:
        p = Path(bge_path).resolve()
    else:
        p = find_bge_model(root)

    if p is not None and looks_like_sentence_transformer(p):
        paths["bge"] = p.resolve()
    else:
        print(f"[Warning] BGE path not found or not a model dir under: {root / 'BAAI'}")

    if not paths:
        raise FileNotFoundError(
            f"No valid local model found under {root}. "
            f"Please check --models_root, --gte_path, or --bge_path."
        )

    return paths


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
            print(f"[Cache] loaded: {cache_path} {arr.shape}")
            return arr.astype("float32")
        else:
            print(
                f"[Cache warning] shape mismatch for {cache_path}: "
                f"cache={arr.shape[0]}, expected={len(texts)}. Recomputing."
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
        print(f"[Cache] saved: {cache_path} {arr.shape}")

    return arr


def search_ip(
    query_embs: np.ndarray,
    corpus_embs: np.ndarray,
    top_k: int = 10,
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
        raise ValueError("Empty gt.")

    hit1 = 0
    hit5 = 0
    hit10 = 0
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


def print_metrics(prefix: str, metrics: Dict[str, float]):
    keys = ["R@1", "R@5", "R@10", "MRR", "nDCG@10", "latency_ms_per_query"]
    items = []
    for k in keys:
        if k in metrics:
            items.append(f"{k}={metrics[k]:.4f}")
    print(f"{prefix}: " + " | ".join(items))


def fit_svd_basis(
    emotion_embs: np.ndarray,
    max_k: int,
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

    max_k = min(max_k, E.shape[0], E.shape[1])

    print(
        f"[SVD] E={tuple(E.shape)}, max_k={max_k}, "
        f"normalize_for_svd={normalize_for_svd}, center_for_svd={center_for_svd}"
    )

    with torch.no_grad():
        _, S, Vh = torch.linalg.svd(E, full_matrices=False)
        B = Vh[:max_k].T.contiguous()
        B, _ = torch.linalg.qr(B, mode="reduced")

    return B.float().cpu().numpy().astype("float32"), S.float().cpu().numpy().astype("float32")


def project_remove(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    B = basis.astype("float32")
    return (x - (x @ B) @ B.T).astype("float32")


def project_keep(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    B = basis.astype("float32")
    return ((x @ B) @ B.T).astype("float32")


def get_pca_basis(
    corpus_embs: np.ndarray,
    max_k: int,
    normalize_for_svd: bool = False,
    center_for_svd: bool = False,
) -> np.ndarray:
    X = torch.from_numpy(corpus_embs.astype("float32"))

    if normalize_for_svd:
        X = F.normalize(X, p=2, dim=1)

    if center_for_svd:
        X = X - X.mean(dim=0, keepdim=True)

    X = X.double()
    max_k = min(max_k, X.shape[0], X.shape[1])

    with torch.no_grad():
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        B = Vh[:max_k].T.contiguous()
        B, _ = torch.linalg.qr(B, mode="reduced")

    return B.float().cpu().numpy().astype("float32")


def orthogonality_error(x: np.ndarray, basis: np.ndarray) -> float:
    task = project_remove(x, basis)
    emo = project_keep(x, basis)

    t = torch.from_numpy(task)
    e = torch.from_numpy(emo)

    t = F.normalize(t, p=2, dim=1)
    e = F.normalize(e, p=2, dim=1)

    return torch.abs(torch.sum(t * e, dim=1)).mean().item()


def projection_errors(basis: np.ndarray) -> Dict[str, float]:
    B = torch.from_numpy(basis.astype("float64"))
    D = B.shape[0]
    I = torch.eye(D, dtype=torch.float64)
    P = I - B @ B.T

    idem = torch.linalg.norm(P @ P - P, ord="fro").item()
    sym = torch.linalg.norm(P - P.T, ord="fro").item()

    return {
        "idempotence_error_fro": idem,
        "symmetry_error_fro": sym,
    }


def run_dense(
    corpus_embs: np.ndarray,
    query_embs: np.ndarray,
    gt: List[List[int]],
    top_k: int,
    split_name: str,
) -> Dict[str, float]:
    _, indices, latency = search_ip(query_embs, corpus_embs, top_k=top_k, normalize=True)
    m = evaluate(indices, gt)
    m["latency_ms_per_query"] = latency
    print_metrics(f"[Dense][{split_name}]", m)
    return m


def run_projected(
    corpus_embs: np.ndarray,
    query_embs: np.ndarray,
    gt: List[List[int]],
    basis: np.ndarray,
    mode: str,
    top_k: int,
    split_name: str,
    method_name: str = "ORCA",
) -> Dict[str, float]:
    if mode == "symmetric":
        c = project_remove(corpus_embs, basis)
        q = project_remove(query_embs, basis)
    elif mode == "doc_only":
        c = project_remove(corpus_embs, basis)
        q = query_embs
    elif mode == "query_only":
        c = corpus_embs
        q = project_remove(query_embs, basis)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    _, indices, latency = search_ip(q, c, top_k=top_k, normalize=True)
    m = evaluate(indices, gt)
    m["latency_ms_per_query"] = latency

    if method_name == "ORCA":
        m["orthogonality_cosine_error"] = orthogonality_error(query_embs, basis)
        m.update(projection_errors(basis))

    print_metrics(f"[{method_name} {mode}][{split_name}][k={basis.shape[1]}]", m)
    return m


def select_best_k(k_results: Dict[str, Dict[str, float]], metric: str = "R@5") -> int:
    best_k = None
    best_score = -1.0

    for k_str, m in k_results.items():
        k = int(k_str)
        score = float(m[metric])
        if score > best_score:
            best_k = k
            best_score = score

    print(f"[Select k] best_k={best_k}, {metric}={best_score:.4f}")
    return int(best_k)


def make_markdown_summary(all_results: Dict[str, Any]) -> str:
    lines = []
    lines.append("# ORCA Local Model Experiment Summary\n")
    lines.append(f"Generated at: {now_time()}\n")

    for model_key, r in all_results["models"].items():
        lines.append(f"## {model_key}\n")
        lines.append(f"- Local path: `{r['model_path']}`")
        lines.append(f"- Selected k: **{r['selected_k']}**")
        lines.append(f"- Dense test R@5: **{r['dense']['test']['R@5']:.4f}**")
        lines.append(f"- ORCA test R@5: **{r['test']['orca_symmetric']['R@5']:.4f}**")
        rel = r["summary"]["relative_improvement_R@5"]
        if rel is not None:
            lines.append(f"- Relative R@5 improvement: **{rel * 100:.2f}%**")
        lines.append("")

        lines.append("| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 |")
        lines.append("|---|---:|---:|---:|---:|---:|")

        rows = [
            ("Dense", r["dense"]["test"]),
            ("ORCA Symmetric", r["test"]["orca_symmetric"]),
            ("ORCA Doc-only", r["test"]["orca_doc_only"]),
            ("ORCA Query-only", r["test"]["orca_query_only"]),
            ("PCA Symmetric", r["test"]["pca_symmetric"]),
        ]

        for name, m in rows:
            lines.append(
                f"| {name} | {m['R@1']:.4f} | {m['R@5']:.4f} | "
                f"{m['R@10']:.4f} | {m['MRR']:.4f} | {m['nDCG@10']:.4f} |"
            )

        lines.append("")
        lines.append("### Dev k sweep\n")
        lines.append("| k | R@1 | R@5 | R@10 | MRR | nDCG@10 |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for k_str, m in r["dev_orca_symmetric_by_k"].items():
            lines.append(
                f"| {k_str} | {m['R@1']:.4f} | {m['R@5']:.4f} | "
                f"{m['R@10']:.4f} | {m['MRR']:.4f} | {m['nDCG@10']:.4f} |"
            )

        lines.append("")

    return "\n".join(lines)


def run_one_model(
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

    model_cache_dir = Path(args.cache_dir) / safe_name(model_key)
    model_cache_dir.mkdir(parents=True, exist_ok=True)

    def cache(name: str) -> Optional[str]:
        if args.no_cache:
            return None
        return str(model_cache_dir / f"{name}.npy")

    emotion_embs = encode_texts(
        model,
        data["emotion_corpus"],
        batch_size=batch_size,
        cache_path=cache("emotion_corpus"),
        normalize_embeddings=False,
    )

    dev_corpus_embs = encode_texts(
        model,
        data["dev_task_corpus"],
        batch_size=batch_size,
        cache_path=cache("dev_task_corpus"),
        normalize_embeddings=False,
    )

    dev_query_embs = encode_texts(
        model,
        data["dev_queries"],
        batch_size=batch_size,
        cache_path=cache("dev_queries"),
        normalize_embeddings=False,
    )

    test_corpus_embs = encode_texts(
        model,
        data["test_task_corpus"],
        batch_size=batch_size,
        cache_path=cache("test_task_corpus"),
        normalize_embeddings=False,
    )

    test_query_embs = encode_texts(
        model,
        data["test_queries"],
        batch_size=batch_size,
        cache_path=cache("test_queries"),
        normalize_embeddings=False,
    )

    k_list = parse_k_list(args.k_list)
    max_k = max(k_list)

    basis_max, singular_values = fit_svd_basis(
        emotion_embs=emotion_embs,
        max_k=max_k,
        normalize_for_svd=args.normalize_for_svd,
        center_for_svd=args.center_for_svd,
        use_float64_svd=True,
    )

    result = {
        "model_key": model_key,
        "model_path": str(model_path),
        "embedding_dim": int(emotion_embs.shape[1]),
        "config": {
            "k_list": k_list,
            "top_k": args.top_k,
            "normalize_for_svd": args.normalize_for_svd,
            "center_for_svd": args.center_for_svd,
            "selection_metric": args.selection_metric,
            "batch_size": batch_size,
        },
        "dense": {},
        "dev_orca_symmetric_by_k": {},
        "test": {},
    }

    print("\n" + "-" * 80)
    print("[Dense baseline]")
    print("-" * 80)
    result["dense"]["dev"] = run_dense(
        dev_corpus_embs,
        dev_query_embs,
        data["dev_gt"],
        top_k=args.top_k,
        split_name="dev",
    )
    result["dense"]["test"] = run_dense(
        test_corpus_embs,
        test_query_embs,
        data["test_gt"],
        top_k=args.top_k,
        split_name="test",
    )

    print("\n" + "-" * 80)
    print("[Dev k sweep: ORCA symmetric]")
    print("-" * 80)
    for k in k_list:
        B = basis_max[:, :k]
        m = run_projected(
            dev_corpus_embs,
            dev_query_embs,
            data["dev_gt"],
            basis=B,
            mode="symmetric",
            top_k=args.top_k,
            split_name="dev",
            method_name="ORCA",
        )
        result["dev_orca_symmetric_by_k"][str(k)] = m

    best_k = select_best_k(result["dev_orca_symmetric_by_k"], metric=args.selection_metric)
    result["selected_k"] = best_k

    B_best = basis_max[:, :best_k]

    print("\n" + "-" * 80)
    print("[Final test: selected k]")
    print("-" * 80)
    result["test"]["orca_symmetric"] = run_projected(
        test_corpus_embs,
        test_query_embs,
        data["test_gt"],
        basis=B_best,
        mode="symmetric",
        top_k=args.top_k,
        split_name="test",
        method_name="ORCA",
    )

    result["test"]["orca_doc_only"] = run_projected(
        test_corpus_embs,
        test_query_embs,
        data["test_gt"],
        basis=B_best,
        mode="doc_only",
        top_k=args.top_k,
        split_name="test",
        method_name="ORCA",
    )

    result["test"]["orca_query_only"] = run_projected(
        test_corpus_embs,
        test_query_embs,
        data["test_gt"],
        basis=B_best,
        mode="query_only",
        top_k=args.top_k,
        split_name="test",
        method_name="ORCA",
    )

    print("\n" + "-" * 80)
    print("[PCA baseline]")
    print("-" * 80)
    pca_basis = get_pca_basis(
        test_corpus_embs,
        max_k=best_k,
        normalize_for_svd=args.normalize_for_svd,
        center_for_svd=args.center_for_svd,
    )
    result["test"]["pca_symmetric"] = run_projected(
        test_corpus_embs,
        test_query_embs,
        data["test_gt"],
        basis=pca_basis,
        mode="symmetric",
        top_k=args.top_k,
        split_name="test",
        method_name="PCA",
    )

    dense_r5 = result["dense"]["test"]["R@5"]
    orca_r5 = result["test"]["orca_symmetric"]["R@5"]
    rel = None if dense_r5 <= 0 else (orca_r5 - dense_r5) / dense_r5

    result["summary"] = {
        "best_k": best_k,
        "test_dense_R@5": dense_r5,
        "test_orca_R@5": orca_r5,
        "relative_improvement_R@5": rel,
    }

    print("\n" + "=" * 80)
    print(f"[Summary][{model_key}]")
    print(f"Selected k: {best_k}")
    print(f"Dense test R@5: {dense_r5:.4f}")
    print(f"ORCA test R@5: {orca_r5:.4f}")
    if rel is not None:
        print(f"Relative improvement R@5: {rel * 100:.2f}%")
    print("=" * 80)

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

    parser.add_argument(
        "--models",
        type=str,
        default="gte,bge",
        help="Comma-separated model keys to run: gte,bge",
    )

    parser.add_argument("--output_json", type=str, default="../results/orca_full_local_experiments.json")
    parser.add_argument("--output_md", type=str, default="../results/orca_full_local_experiments_summary.md")
    parser.add_argument("--cache_dir", type=str, default="../cache/embeddings_final")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gte_batch_size", type=int, default=16)
    parser.add_argument("--bge_batch_size", type=int, default=32)

    parser.add_argument("--k_list", type=str, default="2,4,8,16,32,64")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--selection_metric", type=str, default="R@5")

    parser.add_argument("--normalize_for_svd", action="store_true")
    parser.add_argument("--center_for_svd", action="store_true")
    parser.add_argument("--no_cache", action="store_true")

    args = parser.parse_args()

    print("=" * 100)
    print("[ORCA full local experiments]")
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
        raise KeyError(
            f"Dataset missing fields: {missing}. "
            f"Please use psydt_task_retrieval_final.json with dev/test separated fields."
        )

    print("[Dataset]")
    print(f"Path: {args.data}")
    if "meta" in data and "counts" in data["meta"]:
        for k, v in data["meta"]["counts"].items():
            print(f"  {k}: {v}")

    all_model_paths = resolve_model_paths(
        models_root=args.models_root,
        gte_path=args.gte_path,
        bge_path=args.bge_path,
    )

    requested = [x.strip() for x in args.models.split(",") if x.strip()]
    model_paths = {}
    for key in requested:
        if key not in all_model_paths:
            raise KeyError(
                f"Requested model '{key}' not found. Available: {list(all_model_paths.keys())}"
            )
        model_paths[key] = all_model_paths[key]

    print("\n[Resolved local models]")
    for k, p in model_paths.items():
        print(f"  {k}: {p}")

    all_results = {
        "created_at": now_time(),
        "data_path": args.data,
        "dataset_meta": data.get("meta", {}),
        "global_config": {
            "models_root": args.models_root,
            "models": requested,
            "device": args.device,
            "k_list": parse_k_list(args.k_list),
            "top_k": args.top_k,
            "selection_metric": args.selection_metric,
            "normalize_for_svd": args.normalize_for_svd,
            "center_for_svd": args.center_for_svd,
        },
        "models": {},
    }

    for model_key, model_path in model_paths.items():
        result = run_one_model(model_key, model_path, data, args)
        all_results["models"][model_key] = result

        save_json(all_results, args.output_json)
        print(f"[Intermediate saved] {args.output_json}")

    md = make_markdown_summary(all_results)
    ensure_dir(args.output_md)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)

    save_json(all_results, args.output_json)

    print("\n" + "=" * 100)
    print("[Done]")
    print(f"JSON: {args.output_json}")
    print(f"Markdown summary: {args.output_md}")
    print("=" * 100)


if __name__ == "__main__":
    main()
