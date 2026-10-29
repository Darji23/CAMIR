"""Script to download, format, and stratify ALL benchmark datasets for CAMIR.

Datasets (per CAMIR_Project_Proposal_v3 — 11 sources):
1.  ARC-Easy (400 sampled)             — multiple-choice science, option match
2.  MMLU (400)                         — broad knowledge 57 subjects, option match
3.  GSM8K (400)                        — math word problems, final number
4.  HumanEval (164)                    — Python code generation, unit tests
5.  MATH-500 Levels 4–5               — hard math where small model fails, exact match
6.  MMLU-Pro (400 sampled)             — harder multiple-choice, option match (A–J)
7.  BigCodeBench subset (150)          — harder code with library calls, unit tests
8.  WildChat slice (200–300)           — real messy chat prompts, pairwise judge
9.  IFEval (~541)                      — following constraints in instructions, scripted
10. Tool-use set (via BFCL-v3)         — multi-step function-call agent tasks, task success
11. Perturbed twins (generated later   — noise robustness, in prepare_phase2_data.py)

Splits:
- smoke_20.jsonl: 20 prompts (2 per core dataset) for Phase 1 validation
- train.jsonl  : 40% of corpus
- calibration.jsonl: 20%
- test.jsonl   : 40%
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_jsonl(filename: str, records: List[Dict[str, Any]]) -> None:
    path = DATA_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):>5} records → {path.name}")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# 1. ARC-Easy
# ─────────────────────────────────────────────────────────────────────────────

def format_arc_prompt(item: Dict[str, Any]) -> Dict[str, Any]:
    labels = item["choices"]["label"]
    texts = item["choices"]["text"]
    choices_str = "\n".join(f"({l}) {t}" for l, t in zip(labels, texts))
    return {
        "prompt_id": f"arc_{item['id']}",
        "dataset": "arc_easy",
        "subset": "science",
        "prompt": f"Question: {item['question']}\nChoices:\n{choices_str}\nAnswer:",
        "reference_answer": item["answerKey"],
        "scorer": "multiple_choice",
        "metadata": {"choices": dict(zip(labels, texts))},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. MMLU
# ─────────────────────────────────────────────────────────────────────────────

def format_mmlu_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    subject = item["subject"].replace("_", " ")
    choices = item["choices"]
    labels = ["A", "B", "C", "D"]
    choices_str = "\n".join(f"({l}) {c}" for l, c in zip(labels, choices))
    ans_idx = item["answer"]
    ref_ans = labels[ans_idx] if isinstance(ans_idx, int) and 0 <= ans_idx < 4 else str(ans_idx)
    return {
        "prompt_id": f"mmlu_{item['subject']}_{idx:04d}",
        "dataset": "mmlu",
        "subset": item["subject"],
        "prompt": (
            f"The following is a multiple choice question about {subject}.\n"
            f"Question: {item['question']}\nChoices:\n{choices_str}\nAnswer:"
        ),
        "reference_answer": ref_ans,
        "scorer": "multiple_choice",
        "metadata": {"subject": item["subject"], "choices": dict(zip(labels, choices))},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. GSM8K
# ─────────────────────────────────────────────────────────────────────────────

def _extract_gsm8k_answer(answer_str: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_str)
    if match:
        return match.group(1).replace(",", "").strip()
    return answer_str.split("####")[-1].strip()


def format_gsm8k_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    return {
        "prompt_id": f"gsm8k_{idx:04d}",
        "dataset": "gsm8k",
        "subset": "math",
        "prompt": (
            f"Question: {item['question']}\n"
            "Solve the math problem step by step and conclude with '#### <answer>' "
            "where <answer> is the final numerical value.\nAnswer:"
        ),
        "reference_answer": _extract_gsm8k_answer(item["answer"]),
        "scorer": "math",
        "metadata": {"solution_text": item["answer"]},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. HumanEval
# ─────────────────────────────────────────────────────────────────────────────

def format_humaneval_prompt(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt_id": f"humaneval_{item['task_id'].replace('/', '_')}",
        "dataset": "humaneval",
        "subset": "code",
        "prompt": item["prompt"],
        "reference_answer": item["canonical_solution"],
        "scorer": "code",
        "metadata": {"entry_point": item["entry_point"], "test": item["test"]},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MATH-500 Levels 4–5  (HuggingFace: lighteval/MATH)
# ─────────────────────────────────────────────────────────────────────────────

def format_math500_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    level = str(item.get("level", ""))
    subject = item.get("type", "math")
    return {
        "prompt_id": f"math500_{idx:04d}",
        "dataset": "math_500",
        "subset": subject,
        "prompt": (
            f"Problem: {item['problem']}\n"
            "Solve the problem and put your final answer in a \\boxed{{...}} environment.\nAnswer:"
        ),
        "reference_answer": item["solution"],
        "scorer": "math_latex",
        "metadata": {"level": level, "subject": subject},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. MMLU-Pro  (HuggingFace: TIGER-Lab/MMLU-Pro)
# ─────────────────────────────────────────────────────────────────────────────

def format_mmlu_pro_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    options = item.get("options", [])
    labels = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join(f"({l}) {c}" for l, c in zip(labels, options))
    answer_index = item.get("answer_index", 0)
    ref_ans = labels[answer_index] if 0 <= answer_index < len(labels) else "A"
    return {
        "prompt_id": f"mmlu_pro_{item.get('question_id', idx):06d}",
        "dataset": "mmlu_pro",
        "subset": item.get("category", "general"),
        "prompt": (
            f"The following is a challenging multiple choice question about {item.get('category', 'general')}.\n"
            f"Question: {item['question']}\nChoices:\n{choices_str}\nAnswer:"
        ),
        "reference_answer": ref_ans,
        "scorer": "multiple_choice",
        "metadata": {"category": item.get("category", ""), "choices": dict(zip(labels, options))},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. BigCodeBench  (HuggingFace: bigcode/bigcodebench)
# ─────────────────────────────────────────────────────────────────────────────

def format_bigcodebench_prompt(item: Dict[str, Any]) -> Dict[str, Any]:
    task_id = str(item.get("task_id", ""))
    return {
        "prompt_id": f"bigcodebench_{task_id.replace('/', '_')}",
        "dataset": "bigcodebench",
        "subset": "code_hard",
        "prompt": item.get("instruct_prompt", item.get("complete_prompt", "")),
        "reference_answer": item.get("canonical_solution", ""),
        "scorer": "code",
        "metadata": {
            "entry_point": item.get("entry_point", ""),
            "test": item.get("test", ""),
            "libs": item.get("libs", []),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. WildChat chat slice  (HuggingFace: allenai/WildChat-1M)
# ─────────────────────────────────────────────────────────────────────────────

def format_wildchat_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    # Each row has a list of turns; use the first user turn as the prompt
    conversation = item.get("conversation", [])
    user_turn = ""
    for turn in conversation:
        if turn.get("role") == "user":
            user_turn = turn.get("content", "")
            break
    if not user_turn:
        return None  # skip empty
    return {
        "prompt_id": f"wildchat_{idx:05d}",
        "dataset": "wildchat",
        "subset": "chat",
        "prompt": user_turn.strip(),
        "reference_answer": "",  # no ground truth — pairwise judge
        "scorer": "pairwise_judge",
        "metadata": {
            "language": item.get("language", "English"),
            "toxic": item.get("toxic", False),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. IFEval  (HuggingFace: google/IFEval)
# ─────────────────────────────────────────────────────────────────────────────

def format_ifeval_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Format a real IFEval row. Each row has a prompt and a list of instruction_id_list."""
    prompt = item.get("prompt", "")
    instructions = item.get("instruction_id_list", [])
    kwargs_list = item.get("kwargs", [{}] * len(instructions))
    return {
        "prompt_id": f"ifeval_{idx:04d}",
        "dataset": "ifeval",
        "subset": "instruction_following",
        "prompt": prompt,
        "reference_answer": "",  # checked via scripted rules, not string match
        "scorer": "ifeval",
        "metadata": {
            "instruction_id_list": instructions,
            "kwargs": kwargs_list,
            "key": item.get("key", idx),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10. Tool-use (BFCL-v3 — Berkeley Function-Calling Leaderboard v3)
#     HuggingFace: gorilla-llm/Berkeley-Function-Calling-Leaderboard
# ─────────────────────────────────────────────────────────────────────────────

def format_bfcl_prompt(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    question = item.get("question", [])
    # question is a list of message dicts; extract user content
    user_msg = ""
    if isinstance(question, list):
        for msg in question:
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break
    elif isinstance(question, str):
        user_msg = question

    func_list = item.get("function", [])
    func_descriptions = ""
    if func_list:
        try:
            func_descriptions = "\n\nAvailable functions:\n" + json.dumps(func_list, indent=2)
        except Exception:
            pass

    return {
        "prompt_id": f"bfcl_{idx:05d}",
        "dataset": "bfcl",
        "subset": "tool_use",
        "prompt": user_msg.strip() + func_descriptions,
        "reference_answer": json.dumps(item.get("ground_truth", "")),
        "scorer": "tool_use",
        "metadata": {
            "id": item.get("id", idx),
            "function": func_list,
            "ground_truth": item.get("ground_truth", ""),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_datasets():
    random.seed(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset  # imported lazily to allow dry-runs

    print("\n" + "─" * 70)
    print("  CAMIR Full Dataset Pipeline — 11 Sources (CAMIR_Project_Proposal_v3)")
    print("─" * 70)

    all_items: List[Dict[str, Any]] = []

    # ── 1. ARC-Easy ──────────────────────────────────────────────────────────
    print("\n[1/10] ARC-Easy …")
    arc_ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    arc_indices = random.sample(range(len(arc_ds)), min(400, len(arc_ds)))
    arc_items = [format_arc_prompt(arc_ds[i]) for i in arc_indices]
    all_items.extend(arc_items)
    print(f"       → {len(arc_items)} prompts")

    # ── 2. MMLU ──────────────────────────────────────────────────────────────
    print("[2/10] MMLU …")
    mmlu_ds = load_dataset("cais/mmlu", "all", split="test")
    mmlu_by_subj: Dict[str, List[int]] = {}
    for i, item in enumerate(mmlu_ds):
        mmlu_by_subj.setdefault(item["subject"], []).append(i)
    mmlu_items = []
    subjects = sorted(mmlu_by_subj.keys())
    items_per_subj = 400 // len(subjects)
    remainder = 400 % len(subjects)
    for k, subj in enumerate(subjects):
        idxs = random.sample(mmlu_by_subj[subj], min(items_per_subj + (1 if k < remainder else 0), len(mmlu_by_subj[subj])))
        for idx in idxs:
            mmlu_items.append(format_mmlu_prompt(mmlu_ds[idx], idx=idx))
    random.shuffle(mmlu_items)
    all_items.extend(mmlu_items)
    print(f"       → {len(mmlu_items)} prompts across {len(subjects)} subjects")

    # ── 3. GSM8K ─────────────────────────────────────────────────────────────
    print("[3/10] GSM8K …")
    gsm_ds = load_dataset("openai/gsm8k", "main", split="test")
    gsm_indices = random.sample(range(len(gsm_ds)), min(400, len(gsm_ds)))
    gsm_items = [format_gsm8k_prompt(gsm_ds[i], idx=i) for i in gsm_indices]
    all_items.extend(gsm_items)
    print(f"       → {len(gsm_items)} prompts")

    # ── 4. HumanEval ─────────────────────────────────────────────────────────
    print("[4/10] HumanEval …")
    he_ds = load_dataset("openai/openai_humaneval", split="test")
    he_items = [format_humaneval_prompt(item) for item in he_ds]
    all_items.extend(he_items)
    print(f"       → {len(he_items)} prompts")

    # ── 5. MATH-500 Levels 4–5 ───────────────────────────────────────────────
    print("[5/10] MATH-500 (Levels 4–5) …")
    try:
        math_ds = load_dataset("lighteval/MATH", "all", split="test", trust_remote_code=True)
        hard_math = [item for item in math_ds if str(item.get("level", "")).replace("Level ", "").strip() in ("4", "5")]
        if len(hard_math) > 500:
            hard_math = random.sample(hard_math, 500)
        math_items = [format_math500_prompt(item, idx) for idx, item in enumerate(hard_math)]
    except Exception as e:
        print(f"       [WARN] lighteval/MATH failed ({e}); trying hendrycks/competition_math …")
        try:
            math_ds = load_dataset("hendrycks/competition_math", split="test", trust_remote_code=True)
            hard_math = [item for item in math_ds if str(item.get("level", "")).replace("Level ", "").strip() in ("4", "5")]
            if len(hard_math) > 500:
                hard_math = random.sample(hard_math, 500)
            math_items = [format_math500_prompt(item, idx) for idx, item in enumerate(hard_math)]
        except Exception as e2:
            print(f"       [WARN] MATH fallback also failed ({e2}); skipping MATH-500.")
            math_items = []
    all_items.extend(math_items)
    print(f"       → {len(math_items)} hard math prompts (Levels 4–5)")

    # ── 6. MMLU-Pro ──────────────────────────────────────────────────────────
    print("[6/10] MMLU-Pro …")
    try:
        mmlup_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
        mmlup_indices = random.sample(range(len(mmlup_ds)), min(400, len(mmlup_ds)))
        mmlup_items = [format_mmlu_pro_prompt(mmlup_ds[i], idx=i) for i in mmlup_indices]
    except Exception as e:
        print(f"       [WARN] MMLU-Pro failed ({e}); skipping.")
        mmlup_items = []
    all_items.extend(mmlup_items)
    print(f"       → {len(mmlup_items)} prompts")

    # ── 7. BigCodeBench ───────────────────────────────────────────────────────
    print("[7/10] BigCodeBench …")
    try:
        bcb_ds = load_dataset("bigcode/bigcodebench", split="v0.1.2", trust_remote_code=True)
        bcb_list = list(bcb_ds)
        if len(bcb_list) > 150:
            bcb_list = random.sample(bcb_list, 150)
        bcb_items = [format_bigcodebench_prompt(item) for item in bcb_list]
    except Exception as e:
        print(f"       [WARN] BigCodeBench failed ({e}); skipping.")
        bcb_items = []
    all_items.extend(bcb_items)
    print(f"       → {len(bcb_items)} prompts")

    # ── 8. WildChat slice ─────────────────────────────────────────────────────
    print("[8/10] WildChat …")
    try:
        wc_ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        wc_items = []
        seen = 0
        for item in wc_ds:
            if seen >= 5000:   # scan first 5k to pick 250 English, non-toxic
                break
            seen += 1
            if item.get("language") != "English" or item.get("toxic", True):
                continue
            formatted = format_wildchat_prompt(item, idx=len(wc_items))
            if formatted and len(formatted["prompt"]) > 20:
                wc_items.append(formatted)
            if len(wc_items) >= 250:
                break
        # Deterministic shuffle by index so results are reproducible
        random.shuffle(wc_items)
        wc_items = wc_items[:250]
    except Exception as e:
        print(f"       [WARN] WildChat failed ({e}); skipping.")
        wc_items = []
    all_items.extend(wc_items)
    print(f"       → {len(wc_items)} chat prompts")

    # ── 9. IFEval (real Google IFEval dataset) ───────────────────────────────
    print("[9/10] IFEval …")
    try:
        ifeval_ds = load_dataset("google/IFEval", split="train", trust_remote_code=True)
        ifeval_items = [format_ifeval_prompt(item, idx) for idx, item in enumerate(ifeval_ds)]
    except Exception as e:
        print(f"       [WARN] google/IFEval failed ({e}); generating synthetic fallback …")
        # Synthetic fallback: 500 IFEval-style prompts (10 templates × 50 variants)
        templates = [
            ("min_words", 50, "Write an informative essay about {topic} that is at least 50 words long."),
            ("max_words", 20, "Summarize the concept of {topic} in at most 20 words."),
            ("contains_keyword", "{keyword}", "Explain {topic}. Your response must include the keyword '{keyword}'."),
            ("json_format", "{}", "Provide a JSON object with keys 'name', 'category', and 'description' for {topic}."),
            ("all_lowercase", "true", "Write a sentence about {topic} using exclusively lowercase letters."),
            ("bullet_list", "3", "List 3 key facts about {topic} as a bullet list starting with '- '."),
            ("starts_with", "The", "Describe {topic}. Your response must start with the word 'The'."),
            ("no_comma", "true", "Explain {topic} in one sentence without using any commas."),
            ("ends_with", ".", "Write a sentence about {topic} that ends with a period."),
            ("word_count_between", "10,30", "Describe {topic} in between 10 and 30 words."),
        ]
        topics = ["photosynthesis", "machine learning", "the solar system", "quantum computing",
                  "climate change", "neural networks", "the water cycle", "gravity",
                  "cryptocurrency", "plate tectonics", "RNA sequencing", "Fourier transforms",
                  "supply chain logistics", "reinforcement learning", "black holes",
                  "CRISPR gene editing", "the internet protocol", "fluid dynamics",
                  "computational complexity", "epigenetics", "the Krebs cycle", "convex optimization",
                  "parallel programming", "Bayesian inference", "natural language processing",
                  "protein folding", "nuclear fusion", "prime numbers", "graph theory",
                  "immunology", "compiler design", "operating systems", "database indexing",
                  "elliptic curves", "transformer architecture", "evolutionary algorithms",
                  "electromagnetism", "thermodynamics", "formal logic", "set theory",
                  "information entropy", "distributed systems", "memory management", "cache coherence",
                  "signal processing", "linear algebra", "differential equations", "topology", "number theory"]
        keywords = ["algorithm", "model", "system", "data", "process", "network",
                    "function", "structure", "pattern", "analysis", "framework", "pipeline"]
        ifeval_items = []
        for i in range(500):
            rule_type, rule_arg, template = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            keyword = keywords[i % len(keywords)]
            prompt_text = template.format(topic=topic, keyword=keyword)
            ifeval_items.append({
                "prompt_id": f"ifeval_{i:04d}",
                "dataset": "ifeval",
                "subset": "instruction_following",
                "prompt": prompt_text,
                "reference_answer": str(rule_arg),
                "scorer": "ifeval",
                "metadata": {"rule_type": rule_type, "rule_arg": rule_arg,
                             "instruction_id_list": [rule_type], "kwargs": [{"rule_arg": rule_arg}]},
            })
    all_items.extend(ifeval_items)
    print(f"       → {len(ifeval_items)} IFEval prompts")

    # ── 10. Tool-use — Berkeley Function-Calling Leaderboard v3 ─────────────
    print("[10/10] Tool-Use (BFCL-v3) …")
    try:
        bfcl_ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="train", trust_remote_code=True)
        bfcl_list = list(bfcl_ds)
        if len(bfcl_list) > 300:
            bfcl_list = random.sample(bfcl_list, 300)
        bfcl_items = [fmt for i, item in enumerate(bfcl_list) if (fmt := format_bfcl_prompt(item, i)) is not None]
    except Exception as e:
        print(f"       [WARN] BFCL failed ({e}); generating synthetic tool-use fallback …")
        # Synthetic tool-use with structured function schemas
        tool_templates = [
            ("get_weather", {"location": "New York"}, "What is the weather like in New York?"),
            ("search_web", {"query": "Python tutorial"}, "Search for a Python tutorial online."),
            ("get_stock_price", {"symbol": "AAPL"}, "What is the current stock price of Apple?"),
            ("translate_text", {"text": "Hello", "target_lang": "Spanish"}, "Translate 'Hello' to Spanish."),
            ("send_email", {"to": "bob@example.com", "subject": "Meeting"}, "Send an email to Bob about the upcoming meeting."),
            ("calculate", {"expr": "2 + 2"}, "What is 2 + 2?"),
            ("get_news", {"topic": "AI"}, "Get me today's top news about artificial intelligence."),
            ("set_reminder", {"time": "3pm", "message": "Call dentist"}, "Set a reminder for 3pm to call the dentist."),
            ("book_restaurant", {"name": "Luigi's", "date": "Friday"}, "Book a table at Luigi's for Friday."),
            ("get_directions", {"from": "Times Square", "to": "Central Park"}, "Give me directions from Times Square to Central Park."),
        ]
        bfcl_items = []
        for i in range(200):
            func_name, func_args, prompt = tool_templates[i % len(tool_templates)]
            bfcl_items.append({
                "prompt_id": f"bfcl_{i:05d}",
                "dataset": "bfcl",
                "subset": "tool_use",
                "prompt": prompt,
                "reference_answer": json.dumps({"name": func_name, "arguments": func_args}),
                "scorer": "tool_use",
                "metadata": {"ground_truth": {"name": func_name, "arguments": func_args}},
            })
    all_items.extend(bfcl_items)
    print(f"       → {len(bfcl_items)} tool-use prompts")

    # ─────────────────────────────────────────────────────────────────────────
    # Build splits
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\nTotal corpus: {len(all_items)} prompts across all 10 sources.")
    random.shuffle(all_items)

    # Smoke set: up to 2 from each dataset (≤ 20 total)
    by_dataset: Dict[str, List[Dict]] = {}
    for item in all_items:
        by_dataset.setdefault(item["dataset"], []).append(item)

    smoke_set = []
    for ds_items in by_dataset.values():
        smoke_set.extend(ds_items[:2])
    smoke_set = smoke_set[:20]
    for it in smoke_set:
        it["split"] = "smoke"

    # Exclude smoke from main pool
    smoke_ids = {it["prompt_id"] for it in smoke_set}
    pool = [it for it in all_items if it["prompt_id"] not in smoke_ids]
    random.shuffle(pool)

    n = len(pool)
    n_train = int(n * 0.40)
    n_cal = int(n * 0.20)

    train_set = [{**it, "split": "train"} for it in pool[:n_train]]
    cal_set = [{**it, "split": "calibration"} for it in pool[n_train:n_train + n_cal]]
    test_set = [{**it, "split": "test"} for it in pool[n_train + n_cal:]]

    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────
    print("\nSaving splits …")
    _save_jsonl("smoke_20.jsonl", smoke_set)
    _save_jsonl("train.jsonl", train_set)
    _save_jsonl("calibration.jsonl", cal_set)
    _save_jsonl("test.jsonl", test_set)

    # Dataset-level counts for manifest
    ds_counts = {ds: len(items) for ds, items in by_dataset.items()}
    ds_counts_in_splits = {}
    for split_name, split_data in [("train", train_set), ("calibration", cal_set), ("test", test_set)]:
        for it in split_data:
            key = it["dataset"]
            ds_counts_in_splits.setdefault(key, {"train": 0, "calibration": 0, "test": 0})
            ds_counts_in_splits[key][split_name] += 1

    manifest = {
        "seed": SEED,
        "total_prompts": len(all_items),
        "splits": {
            "smoke": len(smoke_set),
            "train": len(train_set),
            "calibration": len(cal_set),
            "test": len(test_set),
        },
        "datasets_total": ds_counts,
        "datasets_by_split": ds_counts_in_splits,
        "scorer_types": list({it["scorer"] for it in all_items}),
    }
    manifest_path = DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest → {manifest_path.name}")

    # Summary table
    print("\n" + "─" * 70)
    print(f"{'Dataset':<20} {'Scorer':<18} {'Prompts':>8}")
    print("─" * 70)
    for ds, items in sorted(by_dataset.items()):
        scorer = items[0].get("scorer", "unknown") if items else "—"
        print(f"  {ds:<18} {scorer:<18} {len(items):>8}")
    print("─" * 70)
    print(f"  {'TOTAL':<18} {'—':<18} {len(all_items):>8}")
    print("─" * 70)
    print(f"\n✓ Dataset pipeline complete. {len(all_items)} prompts in data/")


if __name__ == "__main__":
    prepare_datasets()
