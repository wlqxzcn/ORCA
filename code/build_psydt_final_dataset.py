import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_PSYDT_PATH = "../data/PsyDTCorpus/PsyDTCorpus_train_mulit_turn_packing.json"
DEFAULT_OUTPUT_DIR = "../data/cleaned"

DEFAULT_MAX_TASK_DOCS = 5000
DEFAULT_MAX_DEV_QUERIES = 500
DEFAULT_MAX_TEST_QUERIES = 2000
DEFAULT_MAX_EMOTION_CORPUS = 5000
DEFAULT_MAX_MEM_CORPUS = 50

DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_DEV_RATIO = 0.10
DEFAULT_TEST_RATIO = 0.20
DEFAULT_SEED = 42


FIRST_PERSON_WORDS = {
    "我", "自己", "本人", "咱", "咱们", "我们"
}

EMOTION_WORDS = {
    "难过", "伤心", "孤独", "寂寞", "害怕", "焦虑", "委屈",
    "愤怒", "生气", "烦躁", "无助", "崩溃", "想哭", "失望",
    "后悔", "紧张", "担忧", "担心", "痛苦", "压抑", "绝望",
    "内疚", "自责", "迷茫", "害羞", "恐惧", "不安", "难受",
    "郁闷", "烦", "累", "心累", "害怕", "恐慌", "慌", "空虚",
    "低落", "沮丧", "委屈", "烦恼", "崩溃", "抑郁", "焦虑",
    "开心", "快乐", "欣慰", "感激", "幸福", "放松", "安心"
}

PHYSICAL_MEDICAL_WORDS = {
    "头疼", "头痛", "头晕", "胃疼", "胃痛", "肚子疼", "腹痛",
    "血压", "血糖", "关节", "疼痛", "恶心", "呕吐", "腹泻",
    "便秘", "心慌", "胸闷", "发烧", "发热", "咳嗽", "医院",
    "医生", "护士", "药", "吃药", "用药", "治疗", "诊断", "检查",
    "手术", "康复", "处方", "挂号", "门诊", "住院", "CT", "ct",
    "B超", "核磁", "磁共振", "新冠", "感冒", "发炎", "感染",
    "医保", "报销", "病历", "化验", "输液"
}

MENTAL_HEALTH_WORDS = {
    "抑郁症", "焦虑症", "强迫症", "双相", "躁郁", "创伤",
    "心理咨询", "心理医生", "心理治疗", "咨询师", "精神科"
}

TOPIC_ENTITY_WORDS = {
    "男朋友", "女朋友", "前任", "老公", "老婆", "丈夫", "妻子",
    "妈妈", "爸爸", "父母", "孩子", "儿子", "女儿", "老师",
    "同学", "同事", "领导", "老板", "朋友", "室友",
    "工作", "上班", "考试", "学习", "学校", "论文", "作业",
    "成绩", "考研", "毕业", "面试", "辞职", "工资",
    "怎么办", "选择", "要不要", "该不该", "不知道怎么"
}


def stable_dedup(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


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
) -> Optional[str]:
    if text is None:
        return None

    text = str(text)

    text = re.sub(r"[\n\r\t]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"[#@$%&*=|\\<>{}~`]", "", text).strip()

    if len(text) < min_len or len(text) > max_len:
        return None

    if chinese_ratio(text) < keep_chinese_ratio:
        return None

    return text


def is_affective_utterance(
    text: str,
    min_len: int = 5,
    max_len: int = 60,
    require_first_person: bool = True,
    require_emotion_word: bool = True,
    exclude_physical_medical: bool = True,
    exclude_topic_entities: bool = True,
) -> bool:
    if text is None:
        return False

    if len(text) < min_len or len(text) > max_len:
        return False

    if require_first_person and not contains_any(text, FIRST_PERSON_WORDS):
        return False

    if require_emotion_word and not contains_any(text, EMOTION_WORDS):
        return False

    if exclude_physical_medical and contains_any(text, PHYSICAL_MEDICAL_WORDS):
        return False

    if exclude_topic_entities and contains_any(text, TOPIC_ENTITY_WORDS):
        return False

    return True


def is_affective_memory_candidate(text: str) -> bool:
    if text is None:
        return False

    if len(text) < 5 or len(text) > 120:
        return False

    if contains_any(text, PHYSICAL_MEDICAL_WORDS):
        return False

    return contains_any(text, EMOTION_WORDS)


def load_psydt(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("Expected the PsyDT file to contain a JSON list.")

    dialogs = []
    for idx, item in enumerate(raw):
        messages = item.get("messages", [])
        if not isinstance(messages, list):
            continue
        dialogs.append(
            {
                "dialog_id": idx,
                "messages": messages,
                "raw": item,
            }
        )
    return dialogs


def split_dialogs(
    dialogs: List[Dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    total = train_ratio + dev_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train/dev/test ratios must sum to 1.0, got {total}"
        )

    rng = random.Random(seed)
    dialogs_shuffled = list(dialogs)
    rng.shuffle(dialogs_shuffled)

    n = len(dialogs_shuffled)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)

    train_dialogs = dialogs_shuffled[:n_train]
    dev_dialogs = dialogs_shuffled[n_train:n_train + n_dev]
    test_dialogs = dialogs_shuffled[n_train + n_dev:]

    return train_dialogs, dev_dialogs, test_dialogs


def extract_adjacent_pairs(
    dialogs: Sequence[Dict[str, Any]],
    query_max_len: int = 150,
    answer_max_len: int = 256,
    query_min_len: int = 5,
    answer_min_len: int = 10,
    keep_chinese_ratio: float = 0.5,
) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []

    for dialog in dialogs:
        dialog_id = dialog["dialog_id"]
        messages = dialog.get("messages", [])

        for i in range(len(messages) - 1):
            cur = messages[i]
            nxt = messages[i + 1]

            if cur.get("role") != "user" or nxt.get("role") != "assistant":
                continue

            q = clean_text(
                cur.get("content", ""),
                max_len=query_max_len,
                min_len=query_min_len,
                keep_chinese_ratio=keep_chinese_ratio,
            )
            a = clean_text(
                nxt.get("content", ""),
                max_len=answer_max_len,
                min_len=answer_min_len,
                keep_chinese_ratio=keep_chinese_ratio,
            )

            if q and a:
                pairs.append(
                    {
                        "dialog_id": dialog_id,
                        "turn_id": i,
                        "query": q,
                        "answer": a,
                    }
                )

    return pairs

def collect_user_messages(
    dialogs: Sequence[Dict[str, Any]],
    max_len: int = 150,
    min_len: int = 5,
    keep_chinese_ratio: float = 0.5,
) -> List[Dict[str, Any]]:
    out = []
    for dialog in dialogs:
        dialog_id = dialog["dialog_id"]
        messages = dialog.get("messages", [])
        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            text = clean_text(
                msg.get("content", ""),
                max_len=max_len,
                min_len=min_len,
                keep_chinese_ratio=keep_chinese_ratio,
            )
            if text:
                out.append(
                    {
                        "dialog_id": dialog_id,
                        "turn_id": i,
                        "text": text,
                    }
                )
    return out


def collect_assistant_messages(
    dialogs: Sequence[Dict[str, Any]],
    max_len: int = 256,
    min_len: int = 10,
    keep_chinese_ratio: float = 0.5,
) -> List[Dict[str, Any]]:
    out = []
    for dialog in dialogs:
        dialog_id = dialog["dialog_id"]
        messages = dialog.get("messages", [])
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            text = clean_text(
                msg.get("content", ""),
                max_len=max_len,
                min_len=min_len,
                keep_chinese_ratio=keep_chinese_ratio,
            )
            if text:
                out.append(
                    {
                        "dialog_id": dialog_id,
                        "turn_id": i,
                        "text": text,
                    }
                )
    return out


def sample_items(items: List[Any], max_n: int, seed: int) -> List[Any]:
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    if max_n is not None and max_n > 0:
        items = items[:max_n]
    return items


def build_task_corpus_and_gt(
    eval_pairs: List[Dict[str, Any]],
    distractor_answers: List[str],
    max_task_docs: int,
    seed: int,
) -> Tuple[List[str], List[List[int]], List[Dict[str, Any]]]:
    if not eval_pairs:
        raise ValueError("eval_pairs is empty.")

    gold_answers = stable_dedup([p["answer"] for p in eval_pairs])

    if len(gold_answers) > max_task_docs:
        raise ValueError(
            f"Number of unique gold answers ({len(gold_answers)}) exceeds "
            f"max_task_docs ({max_task_docs}). Increase max_task_docs or reduce queries."
        )

    rng = random.Random(seed)
    distractors = [x for x in stable_dedup(distractor_answers) if x not in set(gold_answers)]
    rng.shuffle(distractors)

    task_corpus = list(gold_answers)
    remaining = max_task_docs - len(task_corpus)
    task_corpus.extend(distractors[:remaining])

    doc_to_idx = {doc: idx for idx, doc in enumerate(task_corpus)}

    gt = []
    items = []
    for p in eval_pairs:
        answer = p["answer"]
        if answer not in doc_to_idx:
            raise RuntimeError("Gold answer missing from task corpus. This should never happen.")

        idx = doc_to_idx[answer]
        gt.append([idx])

        item = dict(p)
        item["gt"] = [idx]
        items.append(item)

    return task_corpus, gt, items


def build_emotion_corpus(
    train_user_messages: List[Dict[str, Any]],
    max_emotion_corpus: int,
    seed: int,
) -> List[str]:
    candidates = []
    for item in train_user_messages:
        text = item["text"]
        if is_affective_utterance(text):
            candidates.append(text)

    candidates = stable_dedup(candidates)
    return sample_items(candidates, max_emotion_corpus, seed)


def build_memory_corpus(
    train_user_messages: List[Dict[str, Any]],
    used_texts: Iterable[str],
    max_mem_corpus: int,
    seed: int,
) -> List[str]:
    used = set(used_texts)
    candidates = []
    for item in train_user_messages:
        text = item["text"]
        if text in used:
            continue
        if is_affective_memory_candidate(text):
            candidates.append(text)

    candidates = stable_dedup(candidates)
    return sample_items(candidates, max_mem_corpus, seed)


def count_roles(dialogs: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    c = Counter()
    for dialog in dialogs:
        for msg in dialog.get("messages", []):
            c[msg.get("role", "unknown")] += 1
    return dict(c)


def validate_no_dialog_overlap(
    train_dialogs: Sequence[Dict[str, Any]],
    dev_dialogs: Sequence[Dict[str, Any]],
    test_dialogs: Sequence[Dict[str, Any]],
):
    train_ids = {d["dialog_id"] for d in train_dialogs}
    dev_ids = {d["dialog_id"] for d in dev_dialogs}
    test_ids = {d["dialog_id"] for d in test_dialogs}

    if train_ids & dev_ids:
        raise RuntimeError("Train/dev dialog overlap detected.")
    if train_ids & test_ids:
        raise RuntimeError("Train/test dialog overlap detected.")
    if dev_ids & test_ids:
        raise RuntimeError("Dev/test dialog overlap detected.")


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)

    dialogs = load_psydt(args.psydt_path)
    if not dialogs:
        raise ValueError("No valid dialogues loaded.")

    train_dialogs, dev_dialogs, test_dialogs = split_dialogs(
        dialogs,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    validate_no_dialog_overlap(train_dialogs, dev_dialogs, test_dialogs)

    train_user_messages = collect_user_messages(
        train_dialogs,
        max_len=args.query_max_len,
        min_len=args.query_min_len,
        keep_chinese_ratio=args.keep_chinese_ratio,
    )
    train_assistant_messages = collect_assistant_messages(
        train_dialogs,
        max_len=args.answer_max_len,
        min_len=args.answer_min_len,
        keep_chinese_ratio=args.keep_chinese_ratio,
    )

    dev_pairs_all = extract_adjacent_pairs(
        dev_dialogs,
        query_max_len=args.query_max_len,
        answer_max_len=args.answer_max_len,
        query_min_len=args.query_min_len,
        answer_min_len=args.answer_min_len,
        keep_chinese_ratio=args.keep_chinese_ratio,
    )
    test_pairs_all = extract_adjacent_pairs(
        test_dialogs,
        query_max_len=args.query_max_len,
        answer_max_len=args.answer_max_len,
        query_min_len=args.query_min_len,
        answer_min_len=args.answer_min_len,
        keep_chinese_ratio=args.keep_chinese_ratio,
    )

    dev_pairs = sample_items(dev_pairs_all, args.max_dev_queries, args.seed + 101)
    test_pairs = sample_items(test_pairs_all, args.max_test_queries, args.seed + 202)

    if not dev_pairs:
        raise ValueError("No dev pairs extracted. Check cleaning thresholds.")
    if not test_pairs:
        raise ValueError("No test pairs extracted. Check cleaning thresholds.")

    train_assistant_texts = [x["text"] for x in train_assistant_messages]

    dev_task_corpus, dev_gt, dev_items = build_task_corpus_and_gt(
        eval_pairs=dev_pairs,
        distractor_answers=train_assistant_texts,
        max_task_docs=args.max_task_docs,
        seed=args.seed + 303,
    )
    test_task_corpus, test_gt, test_items = build_task_corpus_and_gt(
        eval_pairs=test_pairs,
        distractor_answers=train_assistant_texts,
        max_task_docs=args.max_task_docs,
        seed=args.seed + 404,
    )

    emotion_corpus = build_emotion_corpus(
        train_user_messages=train_user_messages,
        max_emotion_corpus=args.max_emotion_corpus,
        seed=args.seed + 505,
    )

    used_for_memory = set(emotion_corpus)
    used_for_memory.update([p["query"] for p in dev_pairs])
    used_for_memory.update([p["query"] for p in test_pairs])

    mem_corpus = build_memory_corpus(
        train_user_messages=train_user_messages,
        used_texts=used_for_memory,
        max_mem_corpus=args.max_mem_corpus,
        seed=args.seed + 606,
    )
    
    data_pack = {
        "meta": {
            "dataset": "PsyDTCorpus",
            "builder": "build_psydt_final_dataset.py",
            "seed": args.seed,
            "source_path": args.psydt_path,
            "split": {
                "strategy": "dialogue-level split",
                "train_ratio": args.train_ratio,
                "dev_ratio": args.dev_ratio,
                "test_ratio": args.test_ratio,
                "train_dialogs": len(train_dialogs),
                "dev_dialogs": len(dev_dialogs),
                "test_dialogs": len(test_dialogs),
            },
            "construction": {
                "pairing": "adjacent user-assistant turns only",
                "ground_truth": "immediately following assistant response as conversation-derived pseudo relevance label",
                "emotion_source": "train user utterances only",
                "memory_source": "train affective user utterances not used in emotion_corpus",
                "task_corpus": "gold answers forced into corpus plus train assistant distractors",
                "dev_and_test_task_corpora": "separate corpora for strict evaluation",
            },
            "cleaning": {
                "query_max_len": args.query_max_len,
                "answer_max_len": args.answer_max_len,
                "query_min_len": args.query_min_len,
                "answer_min_len": args.answer_min_len,
                "keep_chinese_ratio": args.keep_chinese_ratio,
                "emotion_filter": {
                    "require_first_person": True,
                    "require_emotion_word": True,
                    "exclude_physical_medical": True,
                    "exclude_topic_entities": True,
                    "max_len": 60,
                },
            },
            "counts": {
                "raw_dialogs": len(dialogs),
                "train_user_messages": len(train_user_messages),
                "train_assistant_messages": len(train_assistant_messages),
                "dev_pairs_all": len(dev_pairs_all),
                "test_pairs_all": len(test_pairs_all),
                "dev_queries": len(dev_pairs),
                "test_queries": len(test_pairs),
                "dev_task_corpus": len(dev_task_corpus),
                "test_task_corpus": len(test_task_corpus),
                "emotion_corpus": len(emotion_corpus),
                "mem_corpus": len(mem_corpus),
            },
            "role_counts": {
                "train": count_roles(train_dialogs),
                "dev": count_roles(dev_dialogs),
                "test": count_roles(test_dialogs),
            },
        },

        "emotion_corpus": emotion_corpus,
        "mem_corpus": mem_corpus,

        "dev_task_corpus": dev_task_corpus,
        "dev_queries": [x["query"] for x in dev_items],
        "dev_gt": dev_gt,
        "dev_items": dev_items,


        "test_task_corpus": test_task_corpus,
        "test_queries": [x["query"] for x in test_items],
        "test_gt": test_gt,
        "test_items": test_items,

        "task_corpus": test_task_corpus,
        "med_corpus": test_task_corpus,
        "med_gt": test_gt,
    }

    if len(emotion_corpus) < args.min_emotion_required:
        print(
            f"[Warning] emotion_corpus only has {len(emotion_corpus)} items, "
            f"less than min_emotion_required={args.min_emotion_required}. "
            f"Consider loosening filters or lowering k."
        )

    if len(mem_corpus) < args.max_mem_corpus:
        print(
            f"[Warning] mem_corpus only has {len(mem_corpus)} items, "
            f"less than requested {args.max_mem_corpus}."
        )

    return data_pack


def save_json(data: Dict[str, Any], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final PsyDT retrieval dataset for n-DOD experiments."
    )

    parser.add_argument("--psydt_path", type=str, default=DEFAULT_PSYDT_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output_name", type=str, default="psydt_task_retrieval_final.json")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--dev_ratio", type=float, default=DEFAULT_DEV_RATIO)
    parser.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)

    parser.add_argument("--max_task_docs", type=int, default=DEFAULT_MAX_TASK_DOCS)
    parser.add_argument("--max_dev_queries", type=int, default=DEFAULT_MAX_DEV_QUERIES)
    parser.add_argument("--max_test_queries", type=int, default=DEFAULT_MAX_TEST_QUERIES)
    parser.add_argument("--max_emotion_corpus", type=int, default=DEFAULT_MAX_EMOTION_CORPUS)
    parser.add_argument("--max_mem_corpus", type=int, default=DEFAULT_MAX_MEM_CORPUS)

    parser.add_argument("--query_max_len", type=int, default=150)
    parser.add_argument("--answer_max_len", type=int, default=256)
    parser.add_argument("--query_min_len", type=int, default=5)
    parser.add_argument("--answer_min_len", type=int, default=10)
    parser.add_argument("--keep_chinese_ratio", type=float, default=0.5)

    parser.add_argument(
        "--min_emotion_required",
        type=int,
        default=500,
        help="Only used for warning. If emotion_corpus is smaller than this, print a warning.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    output_path = os.path.join(args.output_dir, args.output_name)

    data = build_dataset(args)
    save_json(data, output_path)

    meta = data["meta"]
    counts = meta["counts"]

    print("\n[Done] Final dataset saved.")
    print(f"Output path: {output_path}")
    print("\n[Counts]")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    print("\n[Recommended usage]")
    print("  - Use emotion_corpus to fit n-DOD emotion basis.")
    print("  - Use dev_task_corpus/dev_queries/dev_gt for k selection.")
    print("  - Use test_task_corpus/test_queries/test_gt for final reporting.")
    print("  - Backward-compatible fields med_corpus and med_gt point to the TEST split.")


if __name__ == "__main__":
    main()
