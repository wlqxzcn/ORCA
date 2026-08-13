import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import faiss
except Exception as e:
    raise RuntimeError(
        "faiss is required. Install faiss-cpu/faiss-gpu in your environment."
    ) from e


USER_ROLES = {
    "user", "seeker", "client", "patient", "用户", "来访者", "求助者", "咨询者", "患者"
}
ASSISTANT_ROLES = {
    "assistant", "counselor", "therapist", "supporter", "doctor",
    "咨询师", "治疗师", "助手", "医生", "回复者"
}


from orca_source_lexicons import (
    FIRST_PERSON_PATTERNS,
    AFFECTIVE_PATTERNS,
    EXCLUSION_PATTERNS,
)


def now() -> float:
    return time.perf_counter()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def stable_hash_texts(texts: Sequence[str]) -> str:
    h = hashlib.md5()
    for t in texts:
        h.update(t.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).replace("\r", " ").replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def unique_keep_order(items: Iterable[str]) -> List[str]:
    od = OrderedDict()
    for x in items:
        x = normalize_text(x)
        if x:
            od.setdefault(x, None)
    return list(od.keys())


def get_role(turn: Dict[str, Any]) -> str:
    for k in ["role", "speaker", "from", "sender", "author", "type"]:
        if k in turn and turn[k] is not None:
            return str(turn[k]).strip().lower()
    return ""


def get_text(turn: Dict[str, Any]) -> str:
    for k in ["content", "text", "utterance", "message", "value", "response"]:
        if k in turn:
            t = normalize_text(turn.get(k))
            if t:
                return t
    for v in turn.values():
        if isinstance(v, str):
            t = normalize_text(v)
            if len(t) >= 2:
                return t
    return ""


def looks_like_user_role(role: str) -> bool:
    r = role.lower()
    return r in USER_ROLES or any(x in r for x in ["user", "client", "seeker", "patient", "来访", "用户", "求助"])


def looks_like_assistant_role(role: str) -> bool:
    r = role.lower()
    return r in ASSISTANT_ROLES or any(x in r for x in ["assistant", "counselor", "therapist", "咨询师", "治疗师", "助手", "医生"])


def iter_turn_lists(obj: Any) -> Iterable[List[Dict[str, Any]]]:
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, list) and all(isinstance(x, dict) for x in item):
                yield item
            elif isinstance(item, dict):
                for key in ["messages", "dialogue", "dialog", "conversation", "conversations", "turns", "data"]:
                    val = item.get(key)
                    if isinstance(val, list) and all(isinstance(x, dict) for x in val):
                        yield val
    elif isinstance(obj, dict):
        for key in ["messages", "dialogue", "dialog", "conversation", "conversations", "turns", "data", "dialogs"]:
            val = obj.get(key)
            if isinstance(val, list):
                if val and all(isinstance(x, dict) for x in val):
                    roles = [get_role(x) for x in val[:5]]
                    if any(looks_like_user_role(r) or looks_like_assistant_role(r) for r in roles):
                        yield val
                    else:
                        for sub in iter_turn_lists(val):
                            yield sub
                else:
                    for sub in iter_turn_lists(val):
                        yield sub


def extract_query_gold_pairs(path: Path) -> List[Tuple[str, str]]:
    obj = load_json(path)
    pairs: List[Tuple[str, str]] = []

    records: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        records = [x for x in obj if isinstance(x, dict)]
    elif isinstance(obj, dict):
        for key in ["pairs", "examples", "data", "items", "queries"]:
            val = obj.get(key)
            if isinstance(val, list):
                records.extend([x for x in val if isinstance(x, dict)])
        records.append(obj)

    q_keys = ["query", "question", "user", "user_utterance", "src", "input", "context"]
    g_keys = ["gold", "answer", "response", "assistant", "counselor_response", "target", "label"]

    for rec in records:
        q = ""
        g = ""
        for k in q_keys:
            if k in rec:
                q = normalize_text(rec.get(k))
                if q:
                    break
        for k in g_keys:
            if k in rec:
                val = rec.get(k)
                if isinstance(val, list) and val:
                    val = val[0]
                g = normalize_text(val)
                if g:
                    break
        if q and g and q != g:
            pairs.append((q, g))
            
    for turns in iter_turn_lists(obj):
        last_user: Optional[str] = None
        for turn in turns:
            role = get_role(turn)
            text = get_text(turn)
            if not text:
                continue
            if looks_like_user_role(role):
                last_user = text
            elif looks_like_assistant_role(role):
                if last_user:
                    pairs.append((last_user, text))
                    last_user = None
            else:
                continue

    od = OrderedDict()
    for q, g in pairs:
        q = normalize_text(q)
        g = normalize_text(g)
        if q and g and q != g:
            od.setdefault((q, g), None)
    out = list(od.keys())
    if not out:
        raise RuntimeError(f"No query-gold pairs extracted from {path}")
    return out


def extract_counselor_responses(path: Path) -> List[str]:
    obj = load_json(path)
    responses: List[str] = []

    records: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        records = [x for x in obj if isinstance(x, dict)]
    elif isinstance(obj, dict):
        for key in ["pairs", "examples", "data", "items", "responses"]:
            val = obj.get(key)
            if isinstance(val, list):
                records.extend([x for x in val if isinstance(x, dict)])
        records.append(obj)

    response_keys = ["response", "assistant", "counselor_response", "answer", "target", "label"]
    for rec in records:
        for k in response_keys:
            if k in rec:
                val = rec.get(k)
                if isinstance(val, list):
                    for x in val:
                        t = normalize_text(x)
                        if t:
                            responses.append(t)
                else:
                    t = normalize_text(val)
                    if t:
                        responses.append(t)
                break

    for turns in iter_turn_lists(obj):
        for turn in turns:
            role = get_role(turn)
            text = get_text(turn)
            if text and looks_like_assistant_role(role):
                responses.append(text)

    responses = unique_keep_order(responses)
    if not responses:
        raise RuntimeError(f"No counselor responses extracted from {path}")
    return responses


def extract_user_utterances(path: Path) -> List[str]:
    obj = load_json(path)
    users: List[str] = []

    records: List[Dict[str, Any]] = []
    if isinstance(obj, list):
        records = [x for x in obj if isinstance(x, dict)]
    elif isinstance(obj, dict):
        for key in ["pairs", "examples", "data", "items", "queries"]:
            val = obj.get(key)
            if isinstance(val, list):
                records.extend([x for x in val if isinstance(x, dict)])
        records.append(obj)

    user_keys = ["query", "question", "user", "user_utterance", "src", "input"]
    for rec in records:
        for k in user_keys:
            if k in rec:
                t = normalize_text(rec.get(k))
                if t:
                    users.append(t)
                break

    for turns in iter_turn_lists(obj):
        for turn in turns:
            role = get_role(turn)
            text = get_text(turn)
            if text and looks_like_user_role(role):
                users.append(text)

    users = unique_keep_order(users)
    if not users:
        raise RuntimeError(f"No user utterances extracted from {path}")
    return users


def contains_any(text: str, patterns: Sequence[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(p.lower() in lower for p in patterns)


def select_source_texts(
    user_texts: Sequence[str],
    source_mode: str,
    source_limit: int,
    seed: int,
    allow_fallback: bool = True,
) -> Tuple[List[str], Dict[str, Any]]:
    rng = random.Random(seed)
    all_users = unique_keep_order(user_texts)

    def is_first_person(t: str) -> bool:
        return contains_any(t, FIRST_PERSON_PATTERNS)

    def is_affective(t: str) -> bool:
        return contains_any(t, AFFECTIVE_PATTERNS)

    def is_medical_or_topic(t: str) -> bool:
        return contains_any(t, EXCLUSION_PATTERNS)

    strict_aff = [t for t in all_users if is_first_person(t) and is_affective(t) and not is_medical_or_topic(t)]
    aff_only = [t for t in all_users if is_affective(t) and not is_medical_or_topic(t)]
    fp_only = [t for t in all_users if is_first_person(t) and not is_medical_or_topic(t)]
    non_med = [t for t in all_users if not is_medical_or_topic(t)]
    non_aff = [t for t in all_users if (not is_affective(t)) and not is_medical_or_topic(t)]
    fp_non_aff = [t for t in all_users if is_first_person(t) and (not is_affective(t)) and not is_medical_or_topic(t)]

    if source_mode == "affective_user":
        candidates = strict_aff
        fallback_used = "none"
        if allow_fallback and len(candidates) < min(source_limit, 100):
            candidates = unique_keep_order(strict_aff + aff_only + fp_only + non_med)
            fallback_used = "strict_affective_user+fallback"
    elif source_mode == "all_user":
        candidates = all_users
        fallback_used = "none"
    elif source_mode == "non_affective_user":
        candidates = non_aff
        fallback_used = "none"
    elif source_mode == "first_person_non_affective_user":
        candidates = fp_non_aff
        fallback_used = "none"
    else:
        raise ValueError(f"Unsupported source_mode: {source_mode}")

    candidates = unique_keep_order(candidates)
    rng.shuffle(candidates)
    if source_limit and source_limit > 0:
        selected = candidates[:source_limit]
    else:
        selected = candidates

    stats = {
        "source_mode": source_mode,
        "source_limit_requested": source_limit,
        "source_selected_n": len(selected),
        "train_user_total_unique": len(all_users),
        "strict_first_person_affective_nonmedical_n": len(strict_aff),
        "affective_nonmedical_n": len(aff_only),
        "first_person_nonmedical_n": len(fp_only),
        "nonmedical_n": len(non_med),
        "non_affective_nonmedical_n": len(non_aff),
        "first_person_non_affective_nonmedical_n": len(fp_non_aff),
        "fallback_used": fallback_used,
        "source_hash": stable_hash_texts(selected),
    }

    if not selected:
        raise RuntimeError(
            f"Source selection produced zero utterances. mode={source_mode}, "
            f"train_user_total={len(all_users)}"
        )
    return selected, stats

class Encoder:
    def __init__(self, model_name: str, batch_size: int = 64, device: Optional[str] = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.backend = None
        self.model = None
        self.tokenizer = None

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
            self.backend = "sentence_transformers"
            return
        except Exception:
            pass

        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.torch = torch
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_name, local_files_only=True).to(device)
            self.model.eval()
            self.backend = "transformers_mean_pooling"
            return
        except Exception as e:
            raise RuntimeError(
                "Failed to load encoder. Install sentence-transformers or transformers, "
                "and make sure the local model path is valid."
            ) from e

    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        texts = [normalize_text(t) for t in texts]
        if self.backend == "sentence_transformers":
            arr = self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=True,
            )
            arr = np.asarray(arr, dtype="float32")
        elif self.backend == "transformers_mean_pooling":
            arrs: List[np.ndarray] = []
            torch = self.torch
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                with torch.no_grad():
                    enc = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                    out = self.model(**enc)
                    last = out.last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1).float()
                    emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
                    emb = emb.detach().cpu().numpy().astype("float32")
                    arrs.append(emb)
            arr = np.vstack(arrs).astype("float32")
            if normalize:
                arr = l2_normalize(arr)
        else:
            raise RuntimeError("Encoder backend not initialized.")
        return arr.astype("float32")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def load_or_encode(
    encoder: Encoder,
    texts: Sequence[str],
    cache_path: Optional[Path],
    normalize: bool = True,
    no_cache: bool = False,
) -> np.ndarray:
    if cache_path is not None:
        ensure_parent(cache_path)
    meta_path = None if cache_path is None else cache_path.with_suffix(cache_path.suffix + ".meta.json")
    text_hash = stable_hash_texts(texts)

    if (not no_cache) and cache_path is not None and cache_path.exists() and meta_path is not None and meta_path.exists():
        meta = load_json(meta_path)
        if meta.get("text_hash") == text_hash and meta.get("model_name") == encoder.model_name:
            return np.load(cache_path).astype("float32")

    arr = encoder.encode(texts, normalize=normalize)
    if cache_path is not None:
        np.save(cache_path, arr)
        dump_json(
            {
                "model_name": encoder.model_name,
                "backend": encoder.backend,
                "normalize": normalize,
                "n_texts": len(texts),
                "text_hash": text_hash,
                "shape": list(arr.shape),
            },
            meta_path,
        )
    return arr


def fit_orca_basis(source_emb: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    t0 = now()
    _, _, vt = np.linalg.svd(source_emb.astype("float32"), full_matrices=False)
    basis = vt[:k].T.astype("float32")
    elapsed = now() - t0
    return basis, elapsed


def project_embeddings(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - (x @ basis) @ basis.T).astype("float32")


def faiss_search(cand_emb: np.ndarray, query_emb: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray, float, float]:
    cand_emb = np.ascontiguousarray(cand_emb.astype("float32"))
    query_emb = np.ascontiguousarray(query_emb.astype("float32"))

    t_index = now()
    index = faiss.IndexFlatIP(cand_emb.shape[1])
    index.add(cand_emb)
    index_sec = now() - t_index

    t_search = now()
    scores, indices = index.search(query_emb, topk)
    search_sec = now() - t_search
    return scores, indices, index_sec, search_sec


def recall_at_k(indices: np.ndarray, gold_indices: Sequence[int], topk: int, pool_label: str) -> float:
    if len(indices) != len(gold_indices):
        raise ValueError("indices and gold_indices length mismatch.")
    missing = sum(1 for g in gold_indices if g < 0)
    if missing:
        raise RuntimeError(
            f"{missing} query golds are missing from pool {pool_label}. "
            "This makes Recall@k incomparable across pool sizes."
        )
    hits = 0
    for row, g in zip(indices, gold_indices):
        if int(g) in set(map(int, row[:topk])):
            hits += 1
    return hits / len(gold_indices)


def mrr(indices: np.ndarray, gold_indices: Sequence[int]) -> float:
    vals = []
    for row, g in zip(indices, gold_indices):
        rank = None
        for j, idx in enumerate(row, start=1):
            if int(idx) == int(g):
                rank = j
                break
        vals.append(0.0 if rank is None else 1.0 / rank)
    return float(np.mean(vals))


def ndcg_at_k(indices: np.ndarray, gold_indices: Sequence[int], k: int) -> float:
    vals = []
    for row, g in zip(indices, gold_indices):
        val = 0.0
        for j, idx in enumerate(row[:k], start=1):
            if int(idx) == int(g):
                val = 1.0 / np.log2(j + 1)
                break
        vals.append(val)
    return float(np.mean(vals))


def build_pool_for_size(
    selected_pairs: Sequence[Tuple[str, str]],
    train_responses: Sequence[str],
    pool_size_label: str,
    seed: int,
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    golds = unique_keep_order([g for _, g in selected_pairs])
    gold_set = set(golds)

    if pool_size_label == "full":
        target_n = len(golds) + len([r for r in train_responses if normalize_text(r) and normalize_text(r) not in gold_set])
    else:
        target_n = int(pool_size_label)
        if target_n < len(golds):
            raise RuntimeError(
                f"Pool size {target_n} is smaller than the number of unique gold responses {len(golds)}. "
                "Increase pool size or reduce max_queries."
            )

    distractors = []
    seen = set(golds)
    for r in train_responses:
        r = normalize_text(r)
        if not r or r in seen:
            continue
        distractors.append(r)
        seen.add(r)
        if pool_size_label != "full" and len(golds) + len(distractors) >= target_n:
            break

    pool = golds + distractors
    if pool_size_label != "full":
        pool = pool[:target_n]

    rng = random.Random(seed + int(hashlib.md5(str(pool_size_label).encode()).hexdigest()[:6], 16))
    rng.shuffle(pool)
    gold_to_idx = {g: i for i, g in enumerate(pool)}
    gold_indices = [gold_to_idx.get(g, -1) for _, g in selected_pairs]

    missing = sum(1 for x in gold_indices if x < 0)
    if missing:
        raise RuntimeError(
            f"{missing} gold responses missing from candidate pool {pool_size_label}. "
            "This should never happen because golds are forced into each pool."
        )

    meta = {
        "pool_label": pool_size_label,
        "pool_n": len(pool),
        "unique_gold_n": len(golds),
        "distractor_n": len(pool) - len(golds),
        "missing_gold_n": missing,
        "pool_hash": stable_hash_texts(pool),
    }
    return pool, gold_indices, meta


def environment_report() -> Dict[str, Any]:
    rep: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "faiss_version": getattr(faiss, "__version__", "unknown"),
    }
    try:
        import torch
        rep["torch_version"] = torch.__version__
        rep["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            rep["cuda_device_count"] = torch.cuda.device_count()
            rep["cuda_device_0"] = torch.cuda.get_device_name(0)
    except Exception as e:
        rep["torch_error"] = str(e)
    try:
        import sentence_transformers
        rep["sentence_transformers_version"] = sentence_transformers.__version__
    except Exception:
        pass
    try:
        import transformers
        rep["transformers_version"] = transformers.__version__
    except Exception:
        pass
    return rep




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test_pairs_file", type=str, required=True,
                   help="Explicit test/dev pair file. No automatic file selection is performed.")
    p.add_argument("--train_responses_file", type=str, required=True,
                   help="Explicit TRAIN file used to extract counselor responses as distractors.")
    p.add_argument("--train_user_file", type=str, required=True,
                   help="Explicit TRAIN file used to extract user utterances for ORCA source corpus.")
    p.add_argument("--encoder_name", type=str, required=True)
    p.add_argument("--pool_sizes", nargs="+", default=["5000", "10000", "20000", "50000", "full"],
                   help="Candidate pool sizes. Use integers or 'full'.")
    p.add_argument("--orca_k", type=int, default=2)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max_queries", type=int, default=2000)
    p.add_argument("--source_limit", type=int, default=5000,
                   help="Maximum source utterances. Use <=0 for all selected source texts.")
    p.add_argument("--source_mode", type=str, default="affective_user",
                   choices=["affective_user", "all_user", "non_affective_user", "first_person_non_affective_user"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default="")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--out_scaling", type=str, required=True)
    p.add_argument("--out_runtime", type=str, required=True)
    p.add_argument("--out_metadata", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    test_pairs_file = Path(args.test_pairs_file)
    train_responses_file = Path(args.train_responses_file)
    train_user_file = Path(args.train_user_file)

    if not test_pairs_file.exists():
        raise FileNotFoundError(test_pairs_file)
    if not train_responses_file.exists():
        raise FileNotFoundError(train_responses_file)
    if not train_user_file.exists():
        raise FileNotFoundError(train_user_file)

    print("[load] extracting query-gold pairs:", test_pairs_file)
    all_pairs = extract_query_gold_pairs(test_pairs_file)
    print(f"[load] extracted pairs: {len(all_pairs)}")

    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)
    selected_pairs = all_pairs[:args.max_queries] if args.max_queries and args.max_queries > 0 else all_pairs
    queries = [q for q, _ in selected_pairs]
    golds = [g for _, g in selected_pairs]
    unique_golds = unique_keep_order(golds)
    print(f"[data] selected queries={len(queries)}, unique_golds={len(unique_golds)}")

    print("[load] extracting TRAIN counselor responses:", train_responses_file)
    train_responses = extract_counselor_responses(train_responses_file)
    print(f"[load] train counselor responses unique={len(train_responses)}")

    print("[load] extracting TRAIN user utterances:", train_user_file)
    train_users = extract_user_utterances(train_user_file)
    print(f"[load] train user utterances unique={len(train_users)}")

    source_texts, source_stats = select_source_texts(
        train_users,
        source_mode=args.source_mode,
        source_limit=args.source_limit,
        seed=args.seed,
        allow_fallback=True,
    )
    print(f"[source] selected source utterances={len(source_texts)}")
    print("[source] stats:", json.dumps(source_stats, ensure_ascii=False, indent=2))

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    encoder = Encoder(args.encoder_name, batch_size=args.batch_size, device=args.device)

    def cp(name: str) -> Optional[Path]:
        if cache_dir is None:
            return None
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        return cache_dir / f"{safe}.npy"

    t0 = now()
    query_emb = load_or_encode(
        encoder, queries, cp("queries"), normalize=True, no_cache=args.no_cache
    )
    query_encode_sec = now() - t0

    t0 = now()
    source_emb = load_or_encode(
        encoder, source_texts, cp("source"), normalize=True, no_cache=args.no_cache
    )
    source_encode_sec = now() - t0

    basis, orca_svd_sec = fit_orca_basis(source_emb, args.orca_k)

    t0 = now()
    orca_query_emb = l2_normalize(project_embeddings(query_emb, basis))
    orca_project_queries_sec = now() - t0

    scaling_rows: List[Dict[str, Any]] = []
    runtime_rows: List[Dict[str, Any]] = []
    pool_metas: List[Dict[str, Any]] = []

    for pool_label in args.pool_sizes:
        pool_label = str(pool_label)
        print(f"\n[pool] building pool={pool_label}")
        pool, gold_indices, pool_meta = build_pool_for_size(
            selected_pairs=selected_pairs,
            train_responses=train_responses,
            pool_size_label=pool_label,
            seed=args.seed,
        )
        pool_metas.append(pool_meta)
        print("[pool] meta:", pool_meta)

        t0 = now()
        cand_emb = load_or_encode(
            encoder, pool, cp(f"candidates_{pool_label}"), normalize=True, no_cache=args.no_cache
        )
        candidate_encode_sec = now() - t0

        _, dense_idx, dense_index_sec, dense_search_sec = faiss_search(cand_emb, query_emb, topk=max(args.topk, 10))
        dense_r_at_k = recall_at_k(dense_idx, gold_indices, args.topk, pool_label)
        dense_r1 = recall_at_k(dense_idx, gold_indices, 1, pool_label)
        dense_r10 = recall_at_k(dense_idx, gold_indices, 10, pool_label)
        dense_mrr = mrr(dense_idx, gold_indices)
        dense_ndcg10 = ndcg_at_k(dense_idx, gold_indices, 10)

        t0 = now()
        orca_cand_emb = l2_normalize(project_embeddings(cand_emb, basis))
        orca_project_candidates_sec = now() - t0

        _, orca_idx, orca_index_sec, orca_search_sec = faiss_search(
            orca_cand_emb, orca_query_emb, topk=max(args.topk, 10)
        )
        orca_r_at_k = recall_at_k(orca_idx, gold_indices, args.topk, pool_label)
        orca_r1 = recall_at_k(orca_idx, gold_indices, 1, pool_label)
        orca_r10 = recall_at_k(orca_idx, gold_indices, 10, pool_label)
        orca_mrr = mrr(orca_idx, gold_indices)
        orca_ndcg10 = ndcg_at_k(orca_idx, gold_indices, 10)

        row = {
            "encoder": args.encoder_name,
            "pool_label": pool_label,
            "pool_n": len(pool),
            "query_n": len(queries),
            "unique_gold_n": len(unique_golds),
            "source_mode": args.source_mode,
            "source_n": len(source_texts),
            "orca_k": args.orca_k,
            "topk": args.topk,
            "dense_r1": dense_r1,
            f"dense_r{args.topk}": dense_r_at_k,
            "dense_r10": dense_r10,
            "dense_mrr": dense_mrr,
            "dense_ndcg10": dense_ndcg10,
            "orca_r1": orca_r1,
            f"orca_r{args.topk}": orca_r_at_k,
            "orca_r10": orca_r10,
            "orca_mrr": orca_mrr,
            "orca_ndcg10": orca_ndcg10,
            f"delta_r{args.topk}": orca_r_at_k - dense_r_at_k,
        }
        scaling_rows.append(row)

        runtime_row = {
            "encoder": args.encoder_name,
            "pool_label": pool_label,
            "pool_n": len(pool),
            "query_n": len(queries),
            "source_n": len(source_texts),
            "query_encode_sec": query_encode_sec,
            "source_encode_sec": source_encode_sec,
            "candidate_encode_sec": candidate_encode_sec,
            "orca_svd_sec": orca_svd_sec,
            "orca_project_queries_sec": orca_project_queries_sec,
            "orca_project_candidates_sec": orca_project_candidates_sec,
            "dense_index_sec": dense_index_sec,
            "dense_search_sec": dense_search_sec,
            "orca_index_sec": orca_index_sec,
            "orca_search_sec": orca_search_sec,
            "offline_orca_extra_sec": orca_svd_sec + orca_project_candidates_sec,
            "online_orca_query_project_sec": orca_project_queries_sec,
        }
        runtime_rows.append(runtime_row)

        print(
            f"[result] pool={pool_label} Dense R@{args.topk}={dense_r_at_k:.4f} "
            f"ORCA R@{args.topk}={orca_r_at_k:.4f} "
            f"Delta={orca_r_at_k - dense_r_at_k:+.4f}"
        )
        print(
            f"[runtime] cand_encode={candidate_encode_sec:.4f}s "
            f"svd={orca_svd_sec:.4f}s cand_proj={orca_project_candidates_sec:.4f}s "
            f"dense_search={dense_search_sec:.4f}s orca_search={orca_search_sec:.4f}s"
        )

    scaling_df = pd.DataFrame(scaling_rows)
    runtime_df = pd.DataFrame(runtime_rows)

    out_scaling = Path(args.out_scaling)
    out_runtime = Path(args.out_runtime)
    ensure_parent(out_scaling)
    ensure_parent(out_runtime)
    scaling_df.to_csv(out_scaling, index=False)
    runtime_df.to_csv(out_runtime, index=False)
    print("\n[write]", out_scaling)
    print("[write]", out_runtime)

    metadata = {
        "script": Path(__file__).name,
        "args": vars(args),
        "test_pairs_file": str(test_pairs_file),
        "train_responses_file": str(train_responses_file),
        "train_user_file": str(train_user_file),
        "pair_n_all": len(all_pairs),
        "query_n_selected": len(queries),
        "unique_gold_n_selected": len(unique_golds),
        "train_responses_n": len(train_responses),
        "train_user_n": len(train_users),
        "source_stats": source_stats,
        "pool_metadata": pool_metas,
        "encoder_backend": encoder.backend,
        "environment": environment_report(),
    }
    out_metadata = Path(args.out_metadata) if args.out_metadata else out_scaling.with_suffix(".metadata.json")
    dump_json(metadata, out_metadata)
    print("[write]", out_metadata)


if __name__ == "__main__":
    main()
