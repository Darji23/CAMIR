"""Deterministic perturbation generator for CAMIR datasets.

Generates noisy twins for evaluation of noise robustness (E14) and
enforces strict semantic invariants:
1. Multiple Choice (ARC, MMLU): Perturbs stem prose ONLY; choices (A, B, C, D) remain verbatim.
2. Math (GSM8K, MATH-500): Perturbs prose words ONLY; all digits, decimals, and numbers remain untouched.
3. Code Generation (HumanEval): Perturbs docstring prose ONLY; function signature and tests remain untouched.

Noise types:
- 'typos': Character swaps, deletions, and keyboard neighbor substitutions.
- 'casing_punct': Lowercase conversion and punctuation removal on prose.
- 'filler': Irrelevant contextual preamble prepended to the prompt.
"""

import random
import re
from typing import Any, Dict, List, Optional, Tuple

KEYBOARD_NEIGHBORS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "ersfxc", "e": "wsdr",
    "f": "rtgvcd", "g": "tyhbvf", "h": "yujnbg", "i": "ujko", "j": "uikmnh",
    "k": "ijlm", "l": "okp", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx"
}

IRRELEVANT_FILLERS = [
    "Note for review: Several unrelated experiments were performed earlier this week. Regarding your inquiry:\n",
    "Background note: Please ignore unrelated logs from previous operations. Here is the question:\n",
    "Internal notice: System environment verification completed successfully. Question for consideration:\n",
    "FYI: The following problem description was submitted by an operator for evaluation:\n",
]


def _perturb_word_typos(word: str, rng: random.Random, rate: float = 0.15) -> str:
    """Apply typo, swap, or drop to a single word if it contains no digits."""
    if len(word) <= 3 or any(c.isdigit() for c in word):
        return word

    if rng.random() > rate:
        return word

    action = rng.choice(["swap", "drop", "substitute"])
    chars = list(word)

    if action == "swap" and len(chars) > 2:
        idx = rng.randint(0, len(chars) - 2)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif action == "drop" and len(chars) > 3:
        idx = rng.randint(1, len(chars) - 2)  # avoid dropping first/last char
        chars.pop(idx)
    elif action == "substitute":
        idx = rng.randint(0, len(chars) - 1)
        orig = chars[idx].lower()
        if orig in KEYBOARD_NEIGHBORS:
            neighbor = rng.choice(KEYBOARD_NEIGHBORS[orig])
            chars[idx] = neighbor.upper() if chars[idx].isupper() else neighbor

    return "".join(chars)


def perturb_prose_typos(text: str, seed: int = 42, rate: float = 0.15) -> str:
    """Apply deterministic character typos and swaps to prose words."""
    rng = random.Random(seed)
    tokens = re.split(r"(\s+)", text)
    result = []
    for tok in tokens:
        if tok.isspace() or not tok:
            result.append(tok)
        else:
            result.append(_perturb_word_typos(tok, rng, rate=rate))
    return "".join(result)


def perturb_prose_casing_punct(text: str) -> str:
    """Apply lowercase conversion and strip punctuation from prose."""
    # Retain newlines and word spacing
    lines = text.split("\n")
    processed_lines = []
    for line in lines:
        cleaned = re.sub(r"[,\.\?!;:\"]", "", line).lower()
        processed_lines.append(cleaned)
    return "\n".join(processed_lines)


def perturb_prose_filler(text: str, seed: int = 42) -> str:
    """Prepend an irrelevant context filler before the prompt."""
    rng = random.Random(seed)
    filler = rng.choice(IRRELEVANT_FILLERS)
    return f"{filler}{text}"


# ==============================================================================
# Invariant-Preserving Dataset Perturbations
# ==============================================================================

def perturb_multiple_choice(prompt: str, noise_type: str, seed: int = 42) -> str:
    """Perturb MC question stem while preserving choices verbatim."""
    choices_marker = "\nChoices:"
    if choices_marker not in prompt:
        choices_marker = "Choices:"

    if choices_marker in prompt:
        stem, choices_part = prompt.split(choices_marker, 1)
        choices_block = f"{choices_marker}{choices_part}"
    else:
        # Fallback: find (A)
        match = re.search(r"(\([A-D]\)|\b[A-D]\))", prompt)
        if match:
            idx = match.start()
            stem, choices_block = prompt[:idx], prompt[idx:]
        else:
            stem, choices_block = prompt, ""

    # Perturb only the stem
    if noise_type == "typos":
        new_stem = perturb_prose_typos(stem, seed=seed)
    elif noise_type == "casing_punct":
        new_stem = perturb_prose_casing_punct(stem)
    elif noise_type == "filler":
        new_stem = perturb_prose_filler(stem, seed=seed)
    else:
        new_stem = stem

    return f"{new_stem}{choices_block}"


def perturb_gsm8k(prompt: str, noise_type: str, seed: int = 42) -> str:
    """Perturb GSM8K math prompt preserving every number verbatim."""
    # Find all number tokens so we can verify and preserve them
    number_pattern = re.compile(r"(\b\d+(?:[\.,]\d+)?\b|\$\d+)")
    
    if noise_type == "typos":
        # _perturb_word_typos already skips tokens with digits
        new_prompt = perturb_prose_typos(prompt, seed=seed)
    elif noise_type == "casing_punct":
        # Only lowercase words, keep all numbers intact
        tokens = re.split(r"(\s+)", prompt)
        out = []
        for tok in tokens:
            if number_pattern.search(tok) or tok.isspace():
                out.append(tok)
            else:
                out.append(re.sub(r"[,\.\?!;]", "", tok).lower())
        new_prompt = "".join(out)
    elif noise_type == "filler":
        new_prompt = perturb_prose_filler(prompt, seed=seed)
    else:
        new_prompt = prompt

    return new_prompt


def perturb_humaneval(prompt: str, noise_type: str, seed: int = 42) -> str:
    """Perturb HumanEval code prompt docstring only, keeping def signature intact."""
    # Detect docstring delimiters
    for delimiter in ['"""', "'''"]:
        if delimiter in prompt:
            parts = prompt.split(delimiter)
            if len(parts) >= 3:
                # parts[0] is signature and indent, parts[1] is docstring prose, parts[2] is remaining code
                doc_prose = parts[1]
                if noise_type == "typos":
                    new_doc = perturb_prose_typos(doc_prose, seed=seed)
                elif noise_type == "casing_punct":
                    new_doc = perturb_prose_casing_punct(doc_prose)
                elif noise_type == "filler":
                    new_doc = f" Note: Internal developer review note.\n{doc_prose}"
                else:
                    new_doc = doc_prose
                return f"{parts[0]}{delimiter}{new_doc}{delimiter}{parts[2]}"

    # If no docstring delimiter found, fall back to non-code lines
    return prompt


def generate_perturbed_prompt(
    prompt: str,
    dataset: str,
    noise_type: str,
    seed: int = 42,
) -> str:
    """Generate a single perturbed prompt adhering to dataset invariants."""
    ds = dataset.lower()
    if ds in ("arc_easy", "mmlu", "mmlu_pro"):
        return perturb_multiple_choice(prompt, noise_type, seed=seed)
    elif ds in ("gsm8k", "math_500"):
        return perturb_gsm8k(prompt, noise_type, seed=seed)
    elif ds in ("humaneval", "bigcodebench"):
        return perturb_humaneval(prompt, noise_type, seed=seed)
    else:
        if noise_type == "typos":
            return perturb_prose_typos(prompt, seed=seed)
        elif noise_type == "casing_punct":
            return perturb_prose_casing_punct(prompt)
        elif noise_type == "filler":
            return perturb_prose_filler(prompt, seed=seed)
        return prompt


def generate_noisy_twins(
    clean_item: Dict[str, Any],
    noise_types: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Produce noisy twin items sharing the same group_id with the clean item."""
    if noise_types is None:
        noise_types = ["typos", "casing_punct", "filler"]

    clean_id = clean_item["prompt_id"]
    group_id = clean_item.get("group_id", clean_id)
    dataset = clean_item["dataset"]
    clean_prompt = clean_item["prompt"]

    twins = []
    for idx, ntype in enumerate(noise_types):
        twin_seed = seed + idx * 101
        perturbed_prompt = generate_perturbed_prompt(
            prompt=clean_prompt,
            dataset=dataset,
            noise_type=ntype,
            seed=twin_seed,
        )
        twin_item = dict(clean_item)
        twin_item["prompt_id"] = f"{clean_id}_noise_{ntype}"
        twin_item["group_id"] = group_id
        twin_item["prompt"] = perturbed_prompt
        twin_item["noise_type"] = ntype
        twin_item["is_perturbed"] = True
        twin_item["clean_prompt_id"] = clean_id
        twins.append(twin_item)

    return twins
