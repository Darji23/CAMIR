"""Script to download, format, and stratify benchmark datasets for CAMIR.

Datasets:
- ARC-Easy (Science multiple-choice)
- MMLU (Multitask multiple-choice)
- GSM8K (Grade school math reasoning)
- HumanEval (Python code completion)

Splits:
- Smoke set: 20 prompts (5 per benchmark) for Phase 1 validation
- Full set: ~1,364 prompts split 40% Train / 20% Calibration / 40% Test
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List
from datasets import load_dataset

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def format_arc_prompt(item: Dict[str, Any]) -> Dict[str, Any]:
    question = item["question"]
    labels = item["choices"]["label"]
    texts = item["choices"]["text"]
    choices_str = "\n".join(f"({l}) {t}" for l, t in zip(labels, texts))
    prompt = f"Question: {question}\nChoices:\n{choices_str}\nAnswer:"
    return {
        "prompt_id": f"arc_{item['id']}",
        "dataset": "arc_easy",
        "subset": "science",
        "prompt": prompt,
        "reference_answer": item["answerKey"],
        "metadata": {
            "choices": dict(zip(labels, texts)),
            "raw_id": item["id"],
        },
    }


def format_mmlu_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question = item["question"]
    subject = item["subject"].replace("_", " ")
    choices = item["choices"]
    labels = ["A", "B", "C", "D"]
    choices_str = "\n".join(f"({l}) {c}" for l, c in zip(labels, choices))
    prompt = (
        f"The following is a multiple choice question about {subject}.\n"
        f"Question: {question}\nChoices:\n{choices_str}\nAnswer:"
    )
    ans_idx = item["answer"]
    ref_ans = labels[ans_idx] if isinstance(ans_idx, int) and 0 <= ans_idx < 4 else str(ans_idx)
    return {
        "prompt_id": f"mmlu_{item['subject']}_{idx:04d}",
        "dataset": "mmlu",
        "subset": item["subject"],
        "prompt": prompt,
        "reference_answer": ref_ans,
        "metadata": {
            "subject": item["subject"],
            "choices": dict(zip(labels, choices)),
        },
    }


def extract_gsm8k_answer(answer_str: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_str.split("####")[-1].strip()


def format_gsm8k_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question = item["question"]
    prompt = (
        f"Question: {question}\n"
        "Solve the math problem step by step and conclude with '#### <answer>' "
        "where <answer> is the final numerical value.\nAnswer:"
    )
    ref_ans = extract_gsm8k_answer(item["answer"])
    return {
        "prompt_id": f"gsm8k_{idx:04d}",
        "dataset": "gsm8k",
        "subset": "math",
        "prompt": prompt,
        "reference_answer": ref_ans,
        "metadata": {
            "solution_text": item["answer"],
        },
    }


def format_humaneval_prompt(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_id": f"humaneval_{item['task_id'].replace('/', '_')}",
        "dataset": "humaneval",
        "subset": "code",
        "prompt": item["prompt"],
        "reference_answer": item["canonical_solution"],
        "metadata": {
            "entry_point": item["entry_point"],
            "test": item["test"],
        },
    }


def prepare_datasets():
    random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading datasets from Hugging Face...")
    # 1. ARC-Easy
    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    # 2. MMLU
    mmlu_ds = load_dataset("cais/mmlu", "all", split="test")
    # 3. GSM8K
    gsm_ds = load_dataset("openai/gsm8k", "main", split="test")
    # 4. HumanEval
    he_ds = load_dataset("openai/openai_humaneval", split="test")

    print(f"Raw sizes: ARC={len(arc_ds)}, MMLU={len(mmlu_ds)}, GSM8K={len(gsm_ds)}, HumanEval={len(he_ds)}")

    # Sample targets:
    # 400 ARC-Easy
    # 400 MMLU (stratified across subjects)
    # 400 GSM8K
    # 164 HumanEval (entire set)

    # ARC-Easy sample
    arc_indices = list(range(len(arc_ds)))
    random.shuffle(arc_indices)
    arc_items = [format_arc_prompt(arc_ds[i]) for i in arc_indices[:400]]

    # GSM8K sample
    gsm_indices = list(range(len(gsm_ds)))
    random.shuffle(gsm_indices)
    gsm_items = [format_gsm8k_prompt(gsm_ds[i], idx=i) for i in gsm_indices[:400]]

    # HumanEval (all 164)
    he_items = [format_humaneval_prompt(item) for item in he_ds]

    # MMLU stratified sample (across subjects)
    mmlu_by_subj: Dict[str, List[int]] = {}
    for i, item in enumerate(mmlu_ds):
        subj = item["subject"]
        mmlu_by_subj.setdefault(subj, []).append(i)

    mmlu_items = []
    # Pick ~7 items per subject across 57 subjects to reach 400
    subjects = sorted(mmlu_by_subj.keys())
    items_per_subj = 400 // len(subjects)
    remainder = 400 % len(subjects)

    for i, subj in enumerate(subjects):
        indices = mmlu_by_subj[subj]
        random.shuffle(indices)
        count = items_per_subj + (1 if i < remainder else 0)
        for idx in indices[:count]:
            mmlu_items.append(format_mmlu_prompt(mmlu_ds[idx], idx=idx))

    random.shuffle(mmlu_items)

    print(f"Sampled: ARC={len(arc_items)}, MMLU={len(mmlu_items)}, GSM8K={len(gsm_items)}, HumanEval={len(he_items)}")
    total_samples = len(arc_items) + len(mmlu_items) + len(gsm_items) + len(he_items)
    print(f"Total benchmark corpus size: {total_samples}")

    # Extract 20-Prompt Smoke Set (5 from each dataset)
    smoke_set = []
    for item in arc_items[:5]:
        smoke_set.append({**item, "split": "smoke"})
    for item in mmlu_items[:5]:
        smoke_set.append({**item, "split": "smoke"})
    for item in gsm_items[:5]:
        smoke_set.append({**item, "split": "smoke"})
    for item in he_items[:5]:
        smoke_set.append({**item, "split": "smoke"})

    # Remaining items for train/cal/test splits (40/20/40)
    def split_pool(pool: List[Dict[str, Any]]):
        random.shuffle(pool)
        n = len(pool)
        n_train = int(n * 0.40)
        n_cal = int(n * 0.20)
        train_p = [{**x, "split": "train"} for x in pool[:n_train]]
        cal_p = [{**x, "split": "calibration"} for x in pool[n_train:n_train + n_cal]]
        test_p = [{**x, "split": "test"} for x in pool[n_train + n_cal:]]
        return train_p, cal_p, test_p

    arc_train, arc_cal, arc_test = split_pool(arc_items)
    mmlu_train, mmlu_cal, mmlu_test = split_pool(mmlu_items)
    gsm_train, gsm_cal, gsm_test = split_pool(gsm_items)
    he_train, he_cal, he_test = split_pool(he_items)

    train_set = arc_train + mmlu_train + gsm_train + he_train
    cal_set = arc_cal + mmlu_cal + gsm_cal + he_cal
    test_set = arc_test + mmlu_test + gsm_test + he_test

    random.shuffle(train_set)
    random.shuffle(cal_set)
    random.shuffle(test_set)

    # Save to JSONL files
    def save_jsonl(filename: str, records: List[Dict[str, Any]]):
        path = DATA_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(records)} records to {path}")

    save_jsonl("smoke_20.jsonl", smoke_set)
    save_jsonl("train.jsonl", train_set)
    save_jsonl("calibration.jsonl", cal_set)
    save_jsonl("test.jsonl", test_set)

    manifest = {
        "seed": SEED,
        "total_prompts": total_samples,
        "splits": {
            "smoke": len(smoke_set),
            "train": len(train_set),
            "calibration": len(cal_set),
            "test": len(test_set),
        },
        "datasets": {
            "arc_easy": len(arc_items),
            "mmlu": len(mmlu_items),
            "gsm8k": len(gsm_items),
            "humaneval": len(he_items),
        },
    }
    manifest_path = DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    prepare_datasets()
