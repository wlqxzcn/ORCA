import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import numpy as np

try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss is required. Please install faiss-gpu or faiss-cpu.") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError("sentence-transformers is required.") from e


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


from orca_source_lexicons import (
    FIRST_PERSON_WORDS,
    EMOTION_WORDS,
    PHYSICAL_MEDICAL_WORDS,
    TOPIC_ENTITY_WORDS,
)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stable_dedup(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item is None:
            continue
        item = str(item).strip()
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def sample_items(items: List[str], max_n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    items = stable_dedup(items)
    rng.shuffle(items)
    return items[:max_n]


def sample_or_repeat(items: List[str], n: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    items = stable_dedup(items)
    if not items:
        return []
    if len(items) >= n:
        rng.shuffle(items)
        return items[:n]
    out = list(items)
    while len(out) < n:
        out.append(rng.choice(items))
    return out[:n]


def contains_any(text: str, words: Iterable[str]) -> bool:
    return any(w in text for w in words)


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    return chinese_chars / max(len(text), 1)


def clean_text(
    text: Any,
    max_len: int = 256,
    min_len: int = 5,
    keep_chinese_ratio: float = 0.5,
) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[#@$%&*=|\\<>{}~`]", "", text).strip()
    if len(text) < min_len or len(text) > max_len:
        return ""
    if chinese_ratio(text) < keep_chinese_ratio:
        return ""
    return text


def is_affective_utterance(text: str) -> bool:
    if text is None:
        return False
    if len(text) < 5 or len(text) > 60:
        return False
    if not contains_any(text, FIRST_PERSON_WORDS):
        return False
    if not contains_any(text, EMOTION_WORDS):
        return False
    if contains_any(text, PHYSICAL_MEDICAL_WORDS):
        return False
    if contains_any(text, TOPIC_ENTITY_WORDS):
        return False
    return True


def is_non_affective_first_person(text: str) -> bool:
    if text is None:
        return False
    if len(text) < 5 or len(text) > 80:
        return False
    if not contains_any(text, FIRST_PERSON_WORDS):
        return False
    if contains_any(text, EMOTION_WORDS):
        return False
    if contains_any(text, PHYSICAL_MEDICAL_WORDS):
        return False
    return True



def extract_split_from_final_dataset(data: Dict[str, Any], split: str) -> Tuple[List[str], List[int], List[str]]:
    q_key = f"{split}_queries"
    gt_key = f"{split}_gt"
    c_key = f"{split}_task_corpus"

    if q_key in data and gt_key in data and c_key in data:
        queries = [str(x).strip() for x in data[q_key]]
        corpus = [str(x).strip() for x in data[c_key]]
        raw_gt = data[gt_key]

        gold_indices = []
        for g in raw_gt:
            if isinstance(g, list):
                if not g:
                    raise ValueError(f"Empty gold list found in {gt_key}.")
                gold_indices.append(int(g[0]))
            else:
                gold_indices.append(int(g))

        if len(queries) != len(gold_indices):
            raise ValueError(f"{split}: len(queries)={len(queries)} != len(gold_indices)={len(gold_indices)}")
        if not corpus:
            raise ValueError(f"{split}: empty corpus.")
        return queries, gold_indices, corpus


    items_key = f"{split}_items"
    if items_key in data and c_key in data:
        corpus = [str(x).strip() for x in data[c_key]]
        queries = []
        gold_indices = []
        for item in data[items_key]:
            if not isinstance(item, dict):
                continue
            q = item.get("query", "")
            gt = item.get("gt", None)
            if q and gt is not None:
                queries.append(str(q).strip())
                if isinstance(gt, list):
                    gold_indices.append(int(gt[0]))
                else:
                    gold_indices.append(int(gt))
        if queries and gold_indices and corpus:
            return queries, gold_indices, corpus

    raise RuntimeError(
        f"Could not parse split={split}. Expected keys "
        f"{q_key}, {gt_key}, {c_key}. Available keys: {list(data.keys())[:50]}"
    )




def load_psydt_dialogs(path: str) -> List[Dict[str, Any]]:
    raw = load_json(path)
    if not isinstance(raw, list):
        raise ValueError("Expected PsyDT file to contain a JSON list.")
    dialogs = []
    for idx, item in enumerate(raw):
        messages = item.get("messages", []) if isinstance(item, dict) else []
        if isinstance(messages, list):
            dialogs.append({"dialog_id": idx, "messages": messages})
    return dialogs


def split_dialogs(
    dialogs: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    dev_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    dialogs = list(dialogs)
    rng.shuffle(dialogs)
    n = len(dialogs)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train = dialogs[:n_train]
    dev = dialogs[n_train:n_train + n_dev]
    test = dialogs[n_train + n_dev:]
    return train, dev, test


def collect_messages(
    dialogs: Sequence[Dict[str, Any]],
    role: str,
    max_len: int,
    min_len: int,
    keep_chinese_ratio: float = 0.5,
) -> List[str]:
    out = []
    for dialog in dialogs:
        for msg in dialog.get("messages", []):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != role:
                continue
            text = clean_text(
                msg.get("content", ""),
                max_len=max_len,
                min_len=min_len,
                keep_chinese_ratio=keep_chinese_ratio,
            )
            if text:
                out.append(text)
    return stable_dedup(out)


def build_source_corpora(
    data: Dict[str, Any],
    psydt_path: str,
    max_source_texts: int,
    seed: int = 42,
) -> Dict[str, List[str]]:
    dialogs = load_psydt_dialogs(psydt_path)
    train_dialogs, _, _ = split_dialogs(dialogs, seed=seed)

    train_user = collect_messages(
        train_dialogs,
        role="user",
        max_len=150,
        min_len=5,
        keep_chinese_ratio=0.5,
    )
    train_counselor = collect_messages(
        train_dialogs,
        role="assistant",
        max_len=256,
        min_len=10,
        keep_chinese_ratio=0.5,
    )

    emotion_corpus = data.get("emotion_corpus", [])
    if not isinstance(emotion_corpus, list):
        emotion_corpus = []
    emotion_corpus = stable_dedup([str(x).strip() for x in emotion_corpus if str(x).strip()])

    if not emotion_corpus:
        emotion_corpus = [x for x in train_user if is_affective_utterance(x)]
        emotion_corpus = stable_dedup(emotion_corpus)

    aff_set = set(emotion_corpus)

    random_non_affective = [
        x for x in train_user
        if x not in aff_set and not is_affective_utterance(x)
    ]

    non_affective_first_person = [
        x for x in random_non_affective
        if is_non_affective_first_person(x)
    ]

    corpora = {
        "heuristic_affective_user": sample_or_repeat(emotion_corpus, max_source_texts, seed + 1),
        "all_train_user": sample_or_repeat(train_user, max_source_texts, seed + 2),
        "random_non_affective_user": sample_or_repeat(random_non_affective, max_source_texts, seed + 3),
        "non_affective_first_person_user": sample_or_repeat(non_affective_first_person, max_source_texts, seed + 4),
        "train_counselor": sample_or_repeat(train_counselor, max_source_texts, seed + 5),
    }

    print("[Source counts]", {k: len(v) for k, v in corpora.items()})
    print("[Raw pools]", {
        "train_user": len(train_user),
        "train_counselor": len(train_counselor),
        "emotion_corpus": len(emotion_corpus),
        "random_non_affective_pool": len(random_non_affective),
        "non_affective_first_person_pool": len(non_affective_first_person),
    })

    return corpora


def encode_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
    )
    return emb.astype("float32")


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)).astype("float32")


def preprocess_for_svd(source_emb: np.ndarray, mode: str) -> np.ndarray:
    x = source_emb.astype("float32")
    if mode in ("norm", "norm_center"):
        x = l2_normalize(x)
    if mode in ("center", "norm_center"):
        x = x - x.mean(axis=0, keepdims=True)
    return x.astype("float32")


def compute_svd_basis(source_emb: np.ndarray, k: int, mode: str) -> np.ndarray:
    x = preprocess_for_svd(source_emb, mode)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    basis = vt[:k].T.astype("float32")


    q, _ = np.linalg.qr(basis)
    return q[:, :k].astype("float32")


def project_remove(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return (x - (x @ basis) @ basis.T).astype("float32")


def faiss_search(query_emb: np.ndarray, doc_emb: np.ndarray, top_k: int) -> np.ndarray:
    q = l2_normalize(query_emb)
    d = l2_normalize(doc_emb)
    index = faiss.IndexFlatIP(d.shape[1])
    index.add(d)
    _, ids = index.search(q, top_k)
    return ids


def compute_metrics(rank_ids: np.ndarray, gold_indices: List[int]) -> Dict[str, float]:
    n = len(gold_indices)
    out = {}
    for k in (1, 5, 10):
        hits = 0
        for i, g in enumerate(gold_indices):
            if int(g) in set(rank_ids[i, :k].tolist()):
                hits += 1
        out[f"R@{k}"] = hits / n

    rr = []
    ndcg10 = []
    for i, g in enumerate(gold_indices):
        row = rank_ids[i, :10].tolist()
        if int(g) in row:
            rank = row.index(int(g)) + 1
            rr.append(1.0 / rank)
            ndcg10.append(1.0 / math.log2(rank + 1))
        else:
            rr.append(0.0)
            ndcg10.append(0.0)

    out["MRR"] = float(np.mean(rr))
    out["nDCG@10"] = float(np.mean(ndcg10))
    return out


def evaluate_embeddings(
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    gold_indices: List[int],
    top_k: int,
) -> Dict[str, float]:
    rank_ids = faiss_search(query_emb, doc_emb, top_k=top_k)
    return compute_metrics(rank_ids, gold_indices)


def evaluate_projected(
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    gold_indices: List[int],
    basis: np.ndarray,
    top_k: int,
) -> Dict[str, float]:
    q_proj = project_remove(query_emb, basis)
    d_proj = project_remove(doc_emb, basis)
    return evaluate_embeddings(q_proj, d_proj, gold_indices, top_k=top_k)


def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def run_one_model(
    model_key: str,
    model_path: str,
    data: Dict[str, Any],
    source_corpora: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    print(f"\n=== Loading {model_key}: {model_path} ===")
    model = SentenceTransformer(model_path, device=args.device)
    batch_size = args.bge_batch_size if model_key == "bge" else args.gte_batch_size

    dev_q, dev_gt, dev_corpus = extract_split_from_final_dataset(data, "dev")
    test_q, test_gt, test_corpus = extract_split_from_final_dataset(data, "test")

    print(f"[{model_key}] dev queries={len(dev_q)}, dev corpus={len(dev_corpus)}")
    print(f"[{model_key}] test queries={len(test_q)}, test corpus={len(test_corpus)}")

    dev_q_emb = encode_texts(model, dev_q, batch_size=batch_size)
    dev_d_emb = encode_texts(model, dev_corpus, batch_size=batch_size)
    test_q_emb = encode_texts(model, test_q, batch_size=batch_size)
    test_d_emb = encode_texts(model, test_corpus, batch_size=batch_size)

    dense_dev = evaluate_embeddings(dev_q_emb, dev_d_emb, dev_gt, top_k=args.top_k)
    dense_test = evaluate_embeddings(test_q_emb, test_d_emb, test_gt, top_k=args.top_k)

    print("Dense dev", dense_dev)
    print("Dense test", dense_test)

    fixed_k_list = parse_csv_ints(args.fixed_k_list)
    svd_modes = parse_csv_strs(args.svd_modes)

    result = {
        "dense": {
            "dev": dense_dev,
            "test": dense_test,
        },
        "sources": {},
    }

    for source_name, texts in source_corpora.items():
        print(f"\n[Source] {source_name}: {len(texts)} texts")
        source_emb = encode_texts(model, texts, batch_size=batch_size)

        source_result = {
            "fixed_k": {},
            "best_by_dev": None,
        }

        best_row = None

        for mode in svd_modes:
            source_result["fixed_k"].setdefault(mode, {})

            for k in fixed_k_list:
                print(f"  mode={mode}, k={k}")
                basis = compute_svd_basis(source_emb, k=k, mode=mode)

                dev_metrics = evaluate_projected(
                    dev_q_emb, dev_d_emb, dev_gt, basis=basis, top_k=args.top_k
                )
                test_metrics = evaluate_projected(
                    test_q_emb, test_d_emb, test_gt, basis=basis, top_k=args.top_k
                )

                row = {
                    "source": source_name,
                    "mode": mode,
                    "k": k,
                    "dev": dev_metrics,
                    "test": test_metrics,
                }

                source_result["fixed_k"][mode][str(k)] = row

                if best_row is None:
                    best_row = row
                elif dev_metrics[args.selection_metric] > best_row["dev"][args.selection_metric]:
                    best_row = row

                print(f"    dev {dev_metrics} test {test_metrics}")

        source_result["best_by_dev"] = best_row
        result["sources"][source_name] = source_result

        print(
            f"[Best by dev] {source_name}: mode={best_row['mode']} k={best_row['k']} "
            f"dev_R@5={best_row['dev'].get('R@5')} test_R@5={best_row['test'].get('R@5')}"
        )

    return result


def make_summary_markdown(results: Dict[str, Any]) -> str:
    lines = []
    lines.append("# ORCA fixed-k source-corpus ablation")
    lines.append("")
    lines.append("This summary reports Recall@5 under fixed k and SVD modes.")
    lines.append("")
    for model_key, model_result in results["models"].items():
        lines.append(f"## {model_key}")
        lines.append("")
        dense = model_result["dense"]["test"]
        lines.append(f"Dense test R@5: **{dense['R@5']:.4f}**")
        lines.append("")

        lines.append("### Fixed k=2, raw SVD")
        lines.append("")
        lines.append("| Source corpus | Test R@5 | Test R@1 | Test R@10 | MRR | nDCG@10 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        lines.append(f"| Dense baseline | {dense['R@5']:.4f} | {dense['R@1']:.4f} | {dense['R@10']:.4f} | {dense['MRR']:.4f} | {dense['nDCG@10']:.4f} |")
        for source_name, source_result in model_result["sources"].items():
            raw = source_result["fixed_k"].get("raw", {})
            if "2" not in raw:
                continue
            test = raw["2"]["test"]
            lines.append(
                f"| {source_name} | {test['R@5']:.4f} | {test['R@1']:.4f} | "
                f"{test['R@10']:.4f} | {test['MRR']:.4f} | {test['nDCG@10']:.4f} |"
            )
        lines.append("")

        lines.append("### Best by dev within fixed-k grid")
        lines.append("")
        lines.append("| Source corpus | Best mode | Best k | Dev R@5 | Test R@5 |")
        lines.append("|---|---:|---:|---:|---:|")
        for source_name, source_result in model_result["sources"].items():
            best = source_result["best_by_dev"]
            lines.append(
                f"| {source_name} | {best['mode']} | {best['k']} | "
                f"{best['dev']['R@5']:.4f} | {best['test']['R@5']:.4f} |"
            )
        lines.append("")
    return "\n".join(lines)


def resolve_model_paths(args: argparse.Namespace) -> Dict[str, str]:
    requested = set(parse_csv_strs(args.models))
    paths = {}
    if "gte" in requested:
        paths["gte"] = args.gte_path or os.path.join(args.models_root, "thenlper", "gte-large-zh")
    if "bge" in requested:
        paths["bge"] = args.bge_path or os.path.join(args.models_root, "BAAI", "bge-large-zh-v1.5")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--psydt_path", required=True)

    parser.add_argument("--models", default="gte,bge")
    parser.add_argument("--models_root", default="../models")
    parser.add_argument("--gte_path", default=None)
    parser.add_argument("--bge_path", default=None)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gte_batch_size", type=int, default=16)
    parser.add_argument("--bge_batch_size", type=int, default=32)

    parser.add_argument("--fixed_k_list", default="1,2,4,8")
    parser.add_argument("--svd_modes", default="raw,norm,center,norm_center")

    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--selection_metric", default="R@5")
    parser.add_argument("--max_source_texts", type=int, default=5000)

    parser.add_argument("--output", default="../results/orca_fixed_k_source_ablation.json")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_json(args.data)
    source_corpora = build_source_corpora(
        data=data,
        psydt_path=args.psydt_path,
        max_source_texts=args.max_source_texts,
        seed=SEED,
    )

    results = {
        "args": vars(args),
        "source_counts": {k: len(v) for k, v in source_corpora.items()},
        "models": {},
    }

    model_paths = resolve_model_paths(args)
    if not model_paths:
        raise ValueError(f"No valid models requested: {args.models}")

    for model_key, model_path in model_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path not found for {model_key}: {model_path}")
        results["models"][model_key] = run_one_model(
            model_key=model_key,
            model_path=model_path,
            data=data,
            source_corpora=source_corpora,
            args=args,
        )

    save_json(results, args.output)
    print(f"\nSaved JSON: {args.output}")

    summary_path = str(Path(args.output).with_name(Path(args.output).stem + "_summary.md"))
    summary = make_summary_markdown(results)
    Path(summary_path).write_text(summary, encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
