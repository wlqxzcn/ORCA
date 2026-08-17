from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from paper_experiment_utils import extract_split, load_json


def safe_key(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--per-query-npz", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--method-a", default="ORCA-k2")
    p.add_argument("--method-b", default="Dense")
    p.add_argument("--output-prefix", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = load_json(args.data)
    test = extract_split(data, "test")
    if test.cluster_ids is None:
        raise ValueError("No dialogue IDs found in test_items")

    payload = np.load(args.per_query_npz)
    key_a = f"{safe_key(args.model)}__{safe_key(args.method_a)}__hit5"
    key_b = f"{safe_key(args.model)}__{safe_key(args.method_b)}__hit5"
    if key_a not in payload or key_b not in payload:
        raise KeyError(f"Missing keys. Need {key_a!r} and {key_b!r}; available={list(payload.keys())[:30]}")

    a = np.asarray(payload[key_a], dtype=np.int8)
    b = np.asarray(payload[key_b], dtype=np.int8)
    if len(a) != len(test.cluster_ids):
        raise ValueError(f"Hit arrays have {len(a)} rows but cluster IDs have {len(test.cluster_ids)}")

    diff = a - b
    clusters = np.unique(test.cluster_ids)
    a_only_dialogues = 0
    b_only_dialogues = 0
    net_a_better = 0
    net_b_better = 0
    net_tied = 0
    both_discordance = 0

    for c in clusters:
        d = diff[test.cluster_ids == c]
        has_a = bool(np.any(d > 0))
        has_b = bool(np.any(d < 0))
        a_only_dialogues += int(has_a)
        b_only_dialogues += int(has_b)
        both_discordance += int(has_a and has_b)
        s = int(d.sum())
        net_a_better += int(s > 0)
        net_b_better += int(s < 0)
        net_tied += int(s == 0)

    row = {
        "model": args.model,
        "method_a": args.method_a,
        "method_b": args.method_b,
        "n_queries": int(len(a)),
        "n_dialogues": int(len(clusters)),
        "a_R@5": float(a.mean()),
        "b_R@5": float(b.mean()),
        "delta_R@5": float((a - b).mean()),
        "a_only_queries": int(np.sum((a == 1) & (b == 0))),
        "b_only_queries": int(np.sum((a == 0) & (b == 1))),
        "a_only_dialogues": int(a_only_dialogues),
        "b_only_dialogues": int(b_only_dialogues),
        "dialogues_with_both_discordance_types": int(both_discordance),
        "net_a_better_dialogues": int(net_a_better),
        "net_b_better_dialogues": int(net_b_better),
        "net_tied_dialogues": int(net_tied),
    }

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(f"{prefix}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    print(f"Saved: {prefix}.csv")


if __name__ == "__main__":
    main()
