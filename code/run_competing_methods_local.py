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

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Dict[str, Any], path: str):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def safe_name(s: str) -> str:
    for ch in ["/", "\\", ":", " ", "\t", "\n"]:
        s = s.replace(ch, "_")
    return s


def print_metrics(name: str, m: Dict[str, float]):
    keys = ["R@1", "R@5", "R@10", "MRR", "nDCG@10", "latency_ms_per_query"]
    print(name + ": " + " | ".join([f"{k}={m[k]:.4f}" for k in keys if k in m]))

def is_model_dir(p: Path) -> bool:
    return p.exists() and p.is_dir() and ((p / "modules.json").exists() or (p / "config.json").exists())


def find_bge(root: Path) -> Optional[Path]:
    baai = root / "BAAI"
    if is_model_dir(baai):
        return baai
    if not baai.exists():
        return None
    names = [
        "bge-large-zh-v1.5", "bge-large-zh-v1___5",
        "bge-base-zh-v1.5", "bge-base-zh-v1___5",
        "bge-small-zh-v1.5", "bge-small-zh-v1___5",
    ]
    for name in names:
        p = baai / name
        if is_model_dir(p):
            return p
    cands = [p for p in baai.rglob("*") if is_model_dir(p)]
    if not cands:
        return None
    return sorted(cands, key=lambda p: (0 if "bge" in p.name.lower() else 1, len(str(p))))[0]


def resolve_models(models_root: str, gte_path: Optional[str], bge_path: Optional[str]) -> Dict[str, Path]:
    root = Path(models_root).resolve()
    out = {}
    t = Path(gte_path).resolve() if gte_path else (root / "thenlper" / "gte-large-zh").resolve()
    if is_model_dir(t):
        out["gte"] = t
    else:
        print(f"[Warning] gte not found: {t}")
    b = Path(bge_path).resolve() if bge_path else find_bge(root)
    if b is not None and is_model_dir(b):
        out["bge"] = b
    else:
        print(f"[Warning] bge not found under {root / 'BAAI'}")
    if not out:
        raise FileNotFoundError("No local model found. Check --models_root / --gte_path / --bge_path.")
    return out

def encode(model: SentenceTransformer, texts: List[str], batch_size: int, cache_path: Optional[str]) -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            print(f"[Cache] loaded {cache_path} {arr.shape}")
            return arr.astype("float32")
        print(f"[Cache warning] shape mismatch for {cache_path}; recomputing.")
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
        rank = [int(x) for x in indices[i].tolist()]
        if any(x in gold for x in rank[:1]):
            hit1 += 1
        if any(x in gold for x in rank[:5]):
            hit5 += 1
        if any(x in gold for x in rank[:10]):
            hit10 += 1
        found = None
        for r, doc in enumerate(rank, start=1):
            if doc in gold:
                found = r
                break
        if found is not None:
            mrr += 1.0 / found
            if found <= 10:
                ndcg10 += 1.0 / math.log2(found + 1)
    return {"R@1": hit1/n, "R@5": hit5/n, "R@10": hit10/n, "MRR": mrr/n, "nDCG@10": ndcg10/n}


def dense_search(q: np.ndarray, c: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray, float]:
    q = q.astype("float32").copy()
    c = c.astype("float32").copy()
    faiss.normalize_L2(q)
    faiss.normalize_L2(c)
    index = faiss.IndexFlatIP(c.shape[1])
    index.add(c)
    st = time.perf_counter()
    scores, idx = index.search(q, top_k)
    latency = (time.perf_counter() - st) * 1000.0 / max(len(q), 1)
    return scores, idx, latency


def run_dense(c: np.ndarray, q: np.ndarray, gt: List[List[int]], top_k: int, depth: int) -> Tuple[Dict[str, float], np.ndarray]:
    scores, idx, lat = dense_search(q, c, max(top_k, depth))
    m = evaluate(idx[:, :top_k], gt)
    m["latency_ms_per_query"] = lat
    return m, idx


def l2norm(x: np.ndarray) -> np.ndarray:
    y = x.astype("float32").copy()
    faiss.normalize_L2(y)
    return y


def get_tokenizer(mode: str):
    if mode in {"auto", "jieba"}:
        try:
            import jieba
            print("[BM25] tokenizer=jieba")
            return lambda text: [w.strip() for w in jieba.lcut(str(text)) if w.strip()]
        except Exception:
            if mode == "jieba":
                raise
            print("[BM25] jieba unavailable; fallback to char tokenizer")
    print("[BM25] tokenizer=char")
    return lambda text: re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", str(text))


class BM25:
    def __init__(self, docs: List[str], tokenizer, k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.tok = tokenizer
        self.k1 = k1
        self.b = b
        self.N = len(docs)
        self.dl = np.zeros(self.N, dtype="float32")
        self.avgdl = 0.0
        self.idf = {}
        self.inv = defaultdict(dict)
        self._build()

    def _build(self):
        df = Counter()
        total = 0
        print(f"[BM25] building index for {self.N} docs")
        for i, doc in enumerate(self.docs):
            tf = Counter(self.tok(doc))
            self.dl[i] = sum(tf.values())
            total += int(self.dl[i])
            for t, c in tf.items():
                df[t] += 1
                self.inv[t][i] = int(c)
        self.avgdl = total / max(self.N, 1)
        for t, dfi in df.items():
            self.idf[t] = math.log(1.0 + (self.N - dfi + 0.5) / (dfi + 0.5))
        print(f"[BM25] vocab={len(self.idf)}, avgdl={self.avgdl:.2f}")

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.N, dtype="float32")
        qtf = Counter(self.tok(query))
        for t in qtf:
            if t not in self.inv:
                continue
            idf = self.idf.get(t, 0.0)
            for doc_id, tf in self.inv[t].items():
                dl = self.dl[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
                scores[doc_id] += idf * tf * (self.k1 + 1.0) / max(denom, 1e-6)
        return scores

    def search(self, queries: List[str], top_k: int) -> Tuple[np.ndarray, np.ndarray, float]:
        idx = np.zeros((len(queries), top_k), dtype="int64")
        sc = np.zeros((len(queries), top_k), dtype="float32")
        st = time.perf_counter()
        for i, q in enumerate(queries):
            s = self.score(q)
            if top_k >= self.N:
                order = np.argsort(-s)
            else:
                part = np.argpartition(-s, top_k-1)[:top_k]
                order = part[np.argsort(-s[part])]
            idx[i] = order[:top_k]
            sc[i] = s[idx[i]]
        lat = (time.perf_counter() - st) * 1000.0 / max(len(queries), 1)
        return sc, idx, lat


def run_bm25(docs: List[str], queries: List[str], gt: List[List[int]], tokenizer, top_k: int, depth: int, k1: float, b: float):
    bm25 = BM25(docs, tokenizer, k1=k1, b=b)
    sc, idx, lat = bm25.search(queries, max(top_k, depth))
    m = evaluate(idx[:, :top_k], gt)
    m["latency_ms_per_query"] = lat
    return m, idx


def rrf(rankings: List[np.ndarray], top_k: int, depth: int = 1000, rrf_k: int = 60, weights: Optional[List[float]] = None) -> np.ndarray:
    if weights is None:
        weights = [1.0] * len(rankings)
    n = rankings[0].shape[0]
    out = np.zeros((n, top_k), dtype="int64")
    for i in range(n):
        scores = defaultdict(float)
        for R, w in zip(rankings, weights):
            d = min(depth, R.shape[1])
            for r in range(d):
                scores[int(R[i, r])] += float(w) / (rrf_k + r + 1)
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        out[i] = [x for x, _ in ordered]
    return out


def fit_svd_basis(X: np.ndarray, normalize_for_svd: bool = False, center_for_svd: bool = False) -> np.ndarray:
    T = torch.from_numpy(X.astype("float32"))
    if normalize_for_svd:
        T = F.normalize(T, p=2, dim=1)
    if center_for_svd:
        T = T - T.mean(dim=0, keepdim=True)
    T = T.double()
    with torch.no_grad():
        _, _, Vh = torch.linalg.svd(T, full_matrices=False)
        B = Vh.T.contiguous()
        B, _ = torch.linalg.qr(B, mode="reduced")
    return B.float().cpu().numpy().astype("float32")


def take_basis(B: np.ndarray, k: int) -> np.ndarray:
    return B[:, :min(k, B.shape[1])].astype("float32")


def project_remove(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    X = X.astype("float32")
    B = B.astype("float32")
    return (X - (X @ B) @ B.T).astype("float32")


def run_projected(c: np.ndarray, q: np.ndarray, gt: List[List[int]], B: np.ndarray, top_k: int, depth: int):
    return run_dense(project_remove(c, B), project_remove(q, B), gt, top_k, depth)


def select_k(name: str, Bfull: np.ndarray, k_list: List[int], cdev: np.ndarray, qdev: np.ndarray, gtdev, top_k: int, depth: int, metric: str):
    best_k, best_s = None, -1.0
    sweep = {}
    print(f"\n[{name}] dev selection")
    for k in k_list:
        m, _ = run_projected(cdev, qdev, gtdev, take_basis(Bfull, k), top_k, depth)
        sweep[str(k)] = m
        print_metrics(f"[{name}][k={k}]", m)
        if m[metric] > best_s:
            best_s = m[metric]
            best_k = k
    print(f"[{name}] selected k={best_k}, {metric}={best_s:.4f}")
    return int(best_k), sweep


def random_basis(dim: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(dim, k)).astype("float32")
    Q, _ = np.linalg.qr(A)
    return Q[:, :k].astype("float32")



def train_probe_direction(X: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced", random_state=seed),
        )
        clf.fit(X, y)
        logreg = clf.named_steps["logisticregression"]
        scaler = clf.named_steps["standardscaler"]
        w = logreg.coef_[0].astype("float32") / np.maximum(scaler.scale_.astype("float32"), 1e-6)
        return w.astype("float32")
    except Exception as e:
        print(f"[INLP] sklearn failed, fallback to mean-difference direction: {repr(e)}")
        return (X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)).astype("float32")


def orthogonalize_vec(w: np.ndarray, Bprev: Optional[np.ndarray]) -> Optional[np.ndarray]:
    w = w.astype("float32")
    if Bprev is not None and Bprev.shape[1] > 0:
        w = w - Bprev @ (Bprev.T @ w)
    n = np.linalg.norm(w)
    if n < 1e-8:
        return None
    return (w / n).astype("float32")


def fit_inlp(emotion: np.ndarray, negative: np.ndarray, n_dirs: int, seed: int, max_per_class: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = emotion.astype("float32")
    neg = negative.astype("float32")
    if len(pos) > max_per_class:
        pos = pos[rng.choice(len(pos), max_per_class, replace=False)]
    if len(neg) > max_per_class:
        neg = neg[rng.choice(len(neg), max_per_class, replace=False)]
    X = np.vstack([pos, neg]).astype("float32")
    y = np.concatenate([np.ones(len(pos), dtype="int64"), np.zeros(len(neg), dtype="int64")])
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    dirs = []
    Bprev = None
    print(f"[INLP] fit {n_dirs} dirs, train X={X.shape}")
    for i in range(n_dirs):
        w = train_probe_direction(X, y, seed + i)
        b = orthogonalize_vec(w, Bprev)
        if b is None:
            break
        dirs.append(b)
        Bprev = np.stack(dirs, axis=1).astype("float32")
        X = project_remove(X, b.reshape(-1, 1))
    if not dirs:
        raise RuntimeError("INLP failed to produce directions")
    B = np.stack(dirs, axis=1).astype("float32")
    Q, _ = np.linalg.qr(B)
    return Q.astype("float32")


EMO_WORDS = {
    "难过", "伤心", "孤独", "寂寞", "害怕", "焦虑", "委屈", "愤怒", "生气", "烦躁",
    "无助", "崩溃", "想哭", "失望", "后悔", "紧张", "担忧", "担心", "痛苦", "压抑",
    "绝望", "内疚", "自责", "羞愧", "恐惧", "不安", "迷茫", "空虚", "疲惫", "抑郁",
    "开心", "快乐", "欣慰", "感激", "满足", "放松", "很累", "很烦", "压力大",
    "撑不下去", "没人理解", "不想说话", "没有意义", "睡不着", "不知道怎么办",
}
FILLERS = ["我觉得", "我感觉", "感觉", "真的", "特别", "非常", "有点", "一直", "总是", "最近", "好像"]


def rule_rewrite(text: str) -> str:
    ori = str(text).strip()
    x = ori
    for w in sorted(EMO_WORDS, key=len, reverse=True):
        x = x.replace(w, "")
    for w in FILLERS:
        x = x.replace(w, "")
    x = re.sub(r"[，,。！？!?；;：:\s]+", "，", x).strip("，。！？!?；;：: ")
    return ori if len(x) < 6 else x


def row(method: str, m: Dict[str, float]) -> str:
    return f"| {method} | {m['R@1']:.4f} | {m['R@5']:.4f} | {m['R@10']:.4f} | {m['MRR']:.4f} | {m['nDCG@10']:.4f} |"


def markdown(results: Dict[str, Any]) -> str:
    lines = ["# ORCA Competing Methods Summary\n", f"Generated at: {results['created_at']}\n"]
    for model_key, r in results["models"].items():
        lines += [f"## {model_key}\n", f"- Model path: `{r['model_path']}`", f"- ORCA selected k: **{r['orca']['selected_k']}**", ""]
        lines += ["| Method | R@1 | R@5 | R@10 | MRR | nDCG@10 |", "|---|---:|---:|---:|---:|---:|"]
        order = ["BM25", "Dense", "Hybrid-RRF", "RuleRewrite-Dense", "PCA", "Random", "INLP", "ORCA"]
        mapping = {
            "BM25": r["bm25"]["test"],
            "Dense": r["dense"]["test"],
            "Hybrid-RRF": r["hybrid_rrf"]["test"],
            "RuleRewrite-Dense": r.get("rule_rewrite_dense", {}).get("test"),
            "PCA": r.get("pca", {}).get("test"),
            "Random": r.get("random", {}).get("test_mean"),
            "INLP": r.get("inlp", {}).get("test"),
            "ORCA": r["orca"]["test"],
        }
        for name in order:
            if mapping.get(name) is not None:
                lines.append(row(name, mapping[name]))
        lines.append("")
        dense_r5 = r["dense"]["test"]["R@5"]
        lines += ["### R@5 relative gain over Dense", "", "| Method | R@5 | Relative gain |", "|---|---:|---:|"]
        for name in order:
            m = mapping.get(name)
            if m is None:
                continue
            rel = 0.0 if name == "Dense" else (m["R@5"] - dense_r5) / dense_r5
            lines.append(f"| {name} | {m['R@5']:.4f} | {rel*100:.2f}% |")
        lines.append("")
        lines += ["### ORCA dev k sweep", "", "| k | R@1 | R@5 | R@10 | MRR | nDCG@10 |", "|---:|---:|---:|---:|---:|---:|"]
        for k, m in r["orca"]["dev_k_sweep"].items():
            lines.append(f"| {k} | {m['R@1']:.4f} | {m['R@5']:.4f} | {m['R@10']:.4f} | {m['MRR']:.4f} | {m['nDCG@10']:.4f} |")
        lines.append("")
    return "\n".join(lines)



def run_one_model(model_key: str, model_path: Path, data: Dict[str, Any], bm25_dev, bm25_test, args) -> Dict[str, Any]:
    print("\n" + "="*100)
    print(f"[Model] {model_key}: {model_path}")
    print("="*100)

    bs = args.bge_batch_size if model_key == "bge" else args.gte_batch_size
    model = SentenceTransformer(str(model_path), device=args.device)
    cache_root = Path(args.cache_dir) / safe_name(model_key)
    cache_root.mkdir(parents=True, exist_ok=True)

    def cache(name: str):
        return None if args.no_cache else str(cache_root / f"{name}.npy")

    emo = encode(model, data["emotion_corpus"], bs, cache("emotion_corpus"))
    cdev = encode(model, data["dev_task_corpus"], bs, cache("dev_task_corpus"))
    qdev = encode(model, data["dev_queries"], bs, cache("dev_queries"))
    ctest = encode(model, data["test_task_corpus"], bs, cache("test_task_corpus"))
    qtest = encode(model, data["test_queries"], bs, cache("test_queries"))

    top_k, depth = args.top_k, args.fusion_depth
    k_list = parse_ints(args.k_list)
    inlp_list = parse_ints(args.inlp_dir_list)

    result: Dict[str, Any] = {
        "model_key": model_key,
        "model_path": str(model_path),
        "embedding_dim": int(emo.shape[1]),
        "bm25": {"dev": bm25_dev["metrics"], "test": bm25_test["metrics"]},
    }

    print("\n[Dense]")
    dense_dev, dense_dev_idx = run_dense(cdev, qdev, data["dev_gt"], top_k, depth)
    dense_test, dense_test_idx = run_dense(ctest, qtest, data["test_gt"], top_k, depth)
    print_metrics("[Dense][dev]", dense_dev)
    print_metrics("[Dense][test]", dense_test)
    result["dense"] = {"dev": dense_dev, "test": dense_test}

    print("\n[Hybrid-RRF]")
    hdev_idx = rrf([dense_dev_idx, bm25_dev["indices"]], top_k, depth, args.rrf_k)
    htest_idx = rrf([dense_test_idx, bm25_test["indices"]], top_k, depth, args.rrf_k)
    hdev = evaluate(hdev_idx, data["dev_gt"])
    htest = evaluate(htest_idx, data["test_gt"])
    print_metrics("[Hybrid-RRF][dev]", hdev)
    print_metrics("[Hybrid-RRF][test]", htest)
    result["hybrid_rrf"] = {"dev": hdev, "test": htest, "rrf_k": args.rrf_k}

    if not args.skip_rule_rewrite:
        print("\n[RuleRewrite-Dense]")
        dev_rw = [rule_rewrite(x) for x in data["dev_queries"]]
        test_rw = [rule_rewrite(x) for x in data["test_queries"]]
        qdev_rw = encode(model, dev_rw, bs, cache("dev_queries_rule_rewrite"))
        qtest_rw = encode(model, test_rw, bs, cache("test_queries_rule_rewrite"))
        rw_dev, _ = run_dense(cdev, qdev_rw, data["dev_gt"], top_k, depth)
        rw_test, _ = run_dense(ctest, qtest_rw, data["test_gt"], top_k, depth)
        print_metrics("[RuleRewrite-Dense][dev]", rw_dev)
        print_metrics("[RuleRewrite-Dense][test]", rw_test)
        result["rule_rewrite_dense"] = {
            "dev": rw_dev,
            "test": rw_test,
            "examples": [{"original": data["test_queries"][i], "rewritten": test_rw[i]} for i in range(min(10, len(test_rw)))]
        }

    print("\n[ORCA]")
    Borca_full = fit_svd_basis(emo, normalize_for_svd=args.normalize_for_svd, center_for_svd=args.center_for_svd)
    orca_k, orca_sweep = select_k("ORCA", Borca_full, k_list, cdev, qdev, data["dev_gt"], top_k, depth, args.selection_metric)
    Borca = take_basis(Borca_full, orca_k)
    orca_dev, _ = run_projected(cdev, qdev, data["dev_gt"], Borca, top_k, depth)
    orca_test, _ = run_projected(ctest, qtest, data["test_gt"], Borca, top_k, depth)
    print_metrics("[ORCA][test]", orca_test)
    result["orca"] = {"selected_k": orca_k, "dev": orca_dev, "test": orca_test, "dev_k_sweep": orca_sweep}

    if not args.skip_pca:
        print("\n[PCA / All-but-the-Top]")
        Bpca_dev = fit_svd_basis(cdev, normalize_for_svd=args.pca_normalize_for_svd, center_for_svd=args.pca_center_for_svd)
        pca_k, pca_sweep = select_k("PCA", Bpca_dev, k_list, cdev, qdev, data["dev_gt"], top_k, depth, args.selection_metric)
        Bpca_test = fit_svd_basis(ctest, normalize_for_svd=args.pca_normalize_for_svd, center_for_svd=args.pca_center_for_svd)
        pca_test, _ = run_projected(ctest, qtest, data["test_gt"], take_basis(Bpca_test, pca_k), top_k, depth)
        print_metrics("[PCA][test]", pca_test)
        result["pca"] = {"selected_k": pca_k, "test": pca_test, "dev_k_sweep": pca_sweep}

    if not args.skip_random:
        print("\n[Random Subspace Removal]")
        trials = []
        for t in range(args.random_trials):
            B = random_basis(ctest.shape[1], orca_k, args.seed + 9973*t)
            m, _ = run_projected(ctest, qtest, data["test_gt"], B, top_k, depth)
            trials.append(m)
            print_metrics(f"[Random][trial={t}]", m)
        mean = {}
        for metric in ["R@1", "R@5", "R@10", "MRR", "nDCG@10"]:
            vals = np.array([x[metric] for x in trials], dtype="float32")
            mean[metric] = float(vals.mean())
            mean[metric + "_std"] = float(vals.std())
        result["random"] = {"k": orca_k, "trials": trials, "test_mean": mean}

    if not args.skip_inlp:
        print("\n[INLP-style Linear Probe Removal]")
        Binlp_full = fit_inlp(emo, cdev, max(inlp_list), args.seed, args.inlp_max_train_per_class)
        inlp_k, inlp_sweep = select_k("INLP", Binlp_full, inlp_list, cdev, qdev, data["dev_gt"], top_k, depth, args.selection_metric)
        inlp_test, _ = run_projected(ctest, qtest, data["test_gt"], take_basis(Binlp_full, inlp_k), top_k, depth)
        print_metrics("[INLP][test]", inlp_test)
        result["inlp"] = {"selected_dirs": inlp_k, "test": inlp_test, "dev_dir_sweep": inlp_sweep}

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/cleaned/psydt_task_retrieval_final.json")
    ap.add_argument("--models_root", default="../models")
    ap.add_argument("--gte_path", default=None)
    ap.add_argument("--bge_path", default=None)
    ap.add_argument("--models", default="gte,bge")
    ap.add_argument("--output_json", default="../results/competing_methods.json")
    ap.add_argument("--output_md", default="../results/competing_methods_summary.md")
    ap.add_argument("--cache_dir", default="../cache/embeddings_final")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gte_batch_size", type=int, default=16)
    ap.add_argument("--bge_batch_size", type=int, default=32)
    ap.add_argument("--no_cache", action="store_true")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--fusion_depth", type=int, default=1000)
    ap.add_argument("--selection_metric", default="R@5")
    ap.add_argument("--k_list", default="2,4,8,16,32,64")
    ap.add_argument("--normalize_for_svd", action="store_true")
    ap.add_argument("--center_for_svd", action="store_true")
    ap.add_argument("--pca_normalize_for_svd", action="store_true")
    ap.add_argument("--pca_center_for_svd", action="store_true")
    ap.add_argument("--bm25_tokenizer", default="auto", choices=["auto", "jieba", "char"])
    ap.add_argument("--bm25_k1", type=float, default=1.5)
    ap.add_argument("--bm25_b", type=float, default=0.75)
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--inlp_dir_list", default="1,2,4,8,16")
    ap.add_argument("--inlp_max_train_per_class", type=int, default=5000)
    ap.add_argument("--random_trials", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_rule_rewrite", action="store_true")
    ap.add_argument("--skip_pca", action="store_true")
    ap.add_argument("--skip_random", action="store_true")
    ap.add_argument("--skip_inlp", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("="*100)
    print("[ORCA competing methods experiments]")
    print(f"Start time: {now()}")
    print("="*100)

    data = load_json(args.data)
    required = ["emotion_corpus", "dev_task_corpus", "dev_queries", "dev_gt", "test_task_corpus", "test_queries", "test_gt"]
    miss = [x for x in required if x not in data]
    if miss:
        raise KeyError(f"Dataset missing fields: {miss}")

    print("[Dataset]")
    print(f"Path: {args.data}")
    if "meta" in data and "counts" in data["meta"]:
        for k, v in data["meta"]["counts"].items():
            print(f"  {k}: {v}")

    tokenizer = get_tokenizer(args.bm25_tokenizer)

    print("\n" + "="*100)
    print("[BM25 dev]")
    print("="*100)
    bm25_dev_m, bm25_dev_idx = run_bm25(data["dev_task_corpus"], data["dev_queries"], data["dev_gt"], tokenizer, args.top_k, args.fusion_depth, args.bm25_k1, args.bm25_b)
    print_metrics("[BM25][dev]", bm25_dev_m)

    print("\n" + "="*100)
    print("[BM25 test]")
    print("="*100)
    bm25_test_m, bm25_test_idx = run_bm25(data["test_task_corpus"], data["test_queries"], data["test_gt"], tokenizer, args.top_k, args.fusion_depth, args.bm25_k1, args.bm25_b)
    print_metrics("[BM25][test]", bm25_test_m)

    paths_all = resolve_models(args.models_root, args.gte_path, args.bge_path)
    requested = [x.strip() for x in args.models.split(",") if x.strip()]
    paths = {}
    for key in requested:
        if key not in paths_all:
            raise KeyError(f"Requested model {key} not found. Available={list(paths_all.keys())}")
        paths[key] = paths_all[key]

    results = {
        "created_at": now(),
        "data_path": args.data,
        "dataset_meta": data.get("meta", {}),
        "global_config": vars(args),
        "bm25_model_independent": {"dev": bm25_dev_m, "test": bm25_test_m},
        "models": {},
    }

    bm25_dev = {"metrics": bm25_dev_m, "indices": bm25_dev_idx}
    bm25_test = {"metrics": bm25_test_m, "indices": bm25_test_idx}

    for key, path in paths.items():
        res = run_one_model(key, path, data, bm25_dev, bm25_test, args)
        results["models"][key] = res
        save_json(results, args.output_json)
        print(f"[Intermediate saved] {args.output_json}")

    md = markdown(results)
    ensure_parent(args.output_md)
    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write(md)
    save_json(results, args.output_json)

    print("\n" + "="*100)
    print("[Done]")
    print(f"JSON: {args.output_json}")
    print(f"Markdown: {args.output_md}")
    print("="*100)


if __name__ == "__main__":
    main()
