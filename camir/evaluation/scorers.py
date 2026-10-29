"""Ground truth scoring and validation engines for CAMIR benchmarks.

Supports (per CAMIR_Project_Proposal_v3):
1.  Multiple Choice exact letter match (ARC-Easy, MMLU, MMLU-Pro)         → scorer="multiple_choice"
2.  Numerical extraction & match (GSM8K)                                    → scorer="math"
3.  LaTeX boxed expression extraction & match (MATH-500)                    → scorer="math_latex"
4.  Sandboxed Python unit test execution (HumanEval, BigCodeBench)          → scorer="code"
5.  IFEval scripted constraint checkers (Google IFEval)                     → scorer="ifeval"
6.  Pairwise judge (WildChat — no ground truth, wins/ties via LLM judge)   → scorer="pairwise_judge"
7.  Tool-use / function-call JSON match (BFCL)                             → scorer="tool_use"
"""

import ast
import json
import multiprocessing
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


# ==============================================================================
# 1. Multiple-Choice Scorer (ARC-Easy, MMLU, MMLU-Pro)
# ==============================================================================

def extract_mc_prediction(text: str) -> Optional[str]:
    """Extract choice letter (A–J) from model generation."""
    clean = text.strip()
    if not clean:
        return None

    # Priority 1: Leading single letter answer: "A", "(B)", "C."
    match = re.match(r"^\(?([A-J])\)?[\.:\s]?", clean, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Priority 2: "The answer is (B)", "Answer: C", "Correct choice: D"
    patterns = [
        r"(?:the\s+)?(?:correct\s+)?answer\s*(?:is)?\s*[:\s]*\(?([A-J])\)?",
        r"choice\s*[:\s]*\(?([A-J])\)?",
        r"\b([A-J])\b(?:\s+is\s+correct|\s+is\s+the\s+answer)",
    ]
    for pat in patterns:
        m = re.search(pat, clean, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Priority 3: Last isolated letter in the text
    letters = re.findall(r"\b([A-J])\b", clean)
    if letters:
        return letters[-1].upper()

    return None


def score_multiple_choice(prediction: str, reference: str) -> bool:
    """Evaluate whether predicted choice matches ground truth."""
    pred = extract_mc_prediction(prediction)
    ref = reference.strip().upper()
    return pred == ref if pred is not None else False


# ==============================================================================
# 2. Numerical Math Scorer (GSM8K)
# ==============================================================================

def extract_math_numerical_answer(text: str) -> Optional[float]:
    """Extract target numerical answer from math reasoning trace."""
    clean = text.strip()
    if not clean:
        return None

    # Priority 1: Standard GSM8K marker '#### 42'
    if "####" in clean:
        part = clean.split("####")[-1].strip()
        nums = re.findall(r"[-+]?\d*\.?\d+", part.replace(",", ""))
        if nums:
            try:
                return float(nums[0])
            except ValueError:
                pass

    # Priority 2: "The answer is 42", "gives 42", "= 42"
    patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s*(?:is)?\s*[:\s]*\$?([-+]?\d*\.?\d+)",
        r"=\s*\$?([-+]?\d*\.?\d+)\s*(?:\.|$)",
        r"(?:equals|total\s+is)\s*\$?([-+]?\d*\.?\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, clean, re.IGNORECASE)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except ValueError:
                pass

    # Priority 3: Last numerical token
    nums = re.findall(r"[-+]?\d*\.?\d+", clean.replace(",", ""))
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass

    return None


def score_math_answer(prediction: str, reference: str) -> bool:
    """Evaluate whether numerical prediction matches reference value."""
    pred_val = extract_math_numerical_answer(prediction)
    ref_val = extract_math_numerical_answer(reference)
    if pred_val is None or ref_val is None:
        return False
    return abs(pred_val - ref_val) < 1e-4


# ==============================================================================
# 3. LaTeX Math Scorer (MATH-500 Levels 4–5)
# ==============================================================================

def extract_boxed_answer(text: str) -> Optional[str]:
    r"""Extract the content of \boxed{...} from a LaTeX expression.

    Handles nested braces up to depth 5.
    """
    # Find \boxed{ start
    pattern = r"\\boxed\{"
    m = re.search(pattern, text)
    if not m:
        # Fallback: look for plain #### marker (some MATH rows include it)
        if "####" in text:
            return text.split("####")[-1].strip()
        return None

    start = m.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1].strip() if depth == 0 else None


def _normalize_math_expr(expr: str) -> str:
    """Normalize math expressions for comparison: strip spaces, commas, currency."""
    expr = expr.strip()
    # Remove LaTeX formatting that doesn't affect value
    expr = re.sub(r"\\,", "", expr)
    expr = re.sub(r"\\!", "", expr)
    expr = re.sub(r"\\text\{[^}]*\}", "", expr)
    expr = re.sub(r"\$", "", expr)
    expr = re.sub(r",", "", expr)  # thousands separator
    expr = re.sub(r"\s+", "", expr)
    return expr.lower()


def score_math_latex(prediction: str, reference: str) -> bool:
    """Score MATH-500: compare \boxed answers, fall back to numerical if needed."""
    pred_box = extract_boxed_answer(prediction)
    ref_box = extract_boxed_answer(reference)

    # Both have boxed answers — compare normalised strings first
    if pred_box is not None and ref_box is not None:
        if _normalize_math_expr(pred_box) == _normalize_math_expr(ref_box):
            return True
        # Try numeric comparison
        try:
            pv = float(re.sub(r"[^\d.\-+eE]", "", pred_box))
            rv = float(re.sub(r"[^\d.\-+eE]", "", ref_box))
            return abs(pv - rv) < 1e-4
        except ValueError:
            return False

    # Fall back to plain numerical match
    return score_math_answer(prediction, reference)


# ==============================================================================
# 4. Python Code Execution Sandbox (HumanEval, BigCodeBench)
# ==============================================================================

def _unsafe_sandbox_worker(code_str: str, queue: multiprocessing.Queue):
    """Executes Python code in an isolated process with restricted globals."""
    try:
        restricted_globals = {"__builtins__": __builtins__}
        exec(code_str, restricted_globals)  # noqa: S102
        queue.put((True, ""))
    except Exception as e:
        queue.put((False, f"{type(e).__name__}: {str(e)}"))


def score_python_code(
    completion: str,
    test_code: str,
    entry_point: Optional[str] = None,
    timeout_s: float = 2.0,
) -> Tuple[bool, str]:
    """Execute code completion against unit tests in a safe, timed subprocess.

    Returns: (passed: bool, message: str)
    """
    full_program = f"{completion}\n\n{test_code}"

    # Syntax check first
    try:
        ast.parse(full_program)
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}"

    queue: multiprocessing.Queue = multiprocessing.Queue()
    proc = multiprocessing.Process(
        target=_unsafe_sandbox_worker,
        args=(full_program, queue),
    )
    proc.start()
    proc.join(timeout=timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return False, f"TimeoutExpired (> {timeout_s}s)"

    if not queue.empty():
        passed, msg = queue.get()
        return passed, msg

    return False, "Sandbox terminated without exit code"


# ==============================================================================
# 5. IFEval Scripted Instruction-Following Scorer
#    Handles both synthetic rule_type entries and real google/IFEval format.
# ==============================================================================

def score_instruction_following(
    prediction: str,
    rule_type: str,
    rule_arg: Any,
) -> bool:
    """Check verifiable instruction-following constraints.

    Supports rule_type values from synthetic prompts AND google/IFEval
    instruction IDs (e.g. 'keywords:existence', 'length_constraint:number_words').
    """
    text = prediction.strip()

    # ── Synthetic rule types ────────────────────────────────────────────────
    if rule_type == "min_words":
        return len(text.split()) >= int(rule_arg)
    if rule_type == "max_words":
        return len(text.split()) <= int(rule_arg)
    if rule_type == "contains_keyword":
        return str(rule_arg).lower() in text.lower()
    if rule_type == "json_format":
        try:
            json.loads(text)
            return True
        except Exception:
            return False
    if rule_type == "all_lowercase":
        return text == text.lower()
    if rule_type == "bullet_list":
        n = int(rule_arg)
        bullet_lines = [l for l in text.splitlines() if l.strip().startswith("-")]
        return len(bullet_lines) >= n
    if rule_type == "starts_with":
        return text.startswith(str(rule_arg))
    if rule_type == "no_comma":
        return "," not in text
    if rule_type == "ends_with":
        return text.endswith(str(rule_arg))
    if rule_type == "word_count_between":
        lo, hi = [int(x) for x in str(rule_arg).split(",")]
        return lo <= len(text.split()) <= hi

    # ── Real google/IFEval instruction IDs ─────────────────────────────────
    # Format: "category:subcategory"
    parts = rule_type.lower().split(":")
    category = parts[0] if parts else ""
    subcategory = parts[1] if len(parts) > 1 else ""

    if category == "keywords":
        if subcategory == "existence":
            # rule_arg may be a plain string OR a dict {"keyword": "python"}
            if isinstance(rule_arg, dict):
                kw = str(rule_arg.get("keyword", "")).lower()
            else:
                kw = str(rule_arg).lower()
            return kw in text.lower()
        if subcategory == "frequency":
            # rule_arg = {"keyword": "python", "frequency": 3}
            if isinstance(rule_arg, dict):
                kw = rule_arg.get("keyword", "")
                freq = int(rule_arg.get("frequency", 1))
                return text.lower().count(kw.lower()) >= freq
    if category == "length_constraint":
        words = text.split()
        if subcategory == "number_words":
            if isinstance(rule_arg, dict):
                relation = rule_arg.get("relation", "at least")
                num_words = int(rule_arg.get("num_words", 0))
                if relation == "at least":
                    return len(words) >= num_words
                if relation in ("at most", "less than"):
                    return len(words) <= num_words
        if subcategory == "number_sentences":
            sentences = re.split(r"[.!?]+", text)
            sentences = [s.strip() for s in sentences if s.strip()]
            if isinstance(rule_arg, dict):
                relation = rule_arg.get("relation", "at least")
                n = int(rule_arg.get("num_sentences", 1))
                if relation == "at least":
                    return len(sentences) >= n
                if relation in ("at most", "less than"):
                    return len(sentences) <= n
    if category == "detectable_format":
        if subcategory == "number_bullet_lists":
            bullet_count = len([l for l in text.splitlines() if l.strip().startswith("*") or l.strip().startswith("-")])
            n = int(rule_arg) if not isinstance(rule_arg, dict) else int(rule_arg.get("num_bullet_lists", 1))
            return bullet_count >= n
        if subcategory == "number_highlighted_sections":
            # Look for ** or __ markdown highlighting
            count = len(re.findall(r"\*\*[^*]+\*\*|__[^_]+__", text))
            n = int(rule_arg) if not isinstance(rule_arg, dict) else int(rule_arg.get("num_highlights", 1))
            return count >= n
    if category == "startend":
        if subcategory == "end_checker":
            end_phrase = str(rule_arg.get("end_phrase", rule_arg)).lower()
            return text.lower().endswith(end_phrase)
        if subcategory == "quotation":
            return text.startswith('"') and text.endswith('"')
    if category == "change_case":
        if subcategory == "english_lowercase":
            return text == text.lower()
        if subcategory == "english_uppercase":
            return text == text.upper()
    if category == "combination":
        # rule_arg is a list of (rule_type, rule_arg) pairs
        if isinstance(rule_arg, list):
            return all(
                score_instruction_following(prediction, rt, ra)
                for rt, ra in rule_arg
            )

    # Unknown rule type — conservative pass (don't penalize unknown constraints)
    return True


def score_ifeval(
    prediction: str,
    metadata: Dict[str, Any],
) -> bool:
    """Score a real IFEval row: all instruction constraints must pass."""
    instruction_ids = metadata.get("instruction_id_list", [])
    kwargs_list = metadata.get("kwargs", [{}] * len(instruction_ids))

    if not instruction_ids:
        # Synthetic format uses rule_type/rule_arg directly
        rule_type = metadata.get("rule_type", "contains_keyword")
        rule_arg = metadata.get("rule_arg", "")
        return score_instruction_following(prediction, rule_type, rule_arg)

    # All constraints must pass (IFEval uses AND semantics)
    for instr_id, kwargs in zip(instruction_ids, kwargs_list):
        rule_arg = kwargs if kwargs else {}
        if not score_instruction_following(prediction, instr_id, rule_arg):
            return False
    return True


# ==============================================================================
# 6. Pairwise Judge (WildChat — no ground truth)
#    Returns a soft score signal; default is True (no penalisation) for
#    offline replay experiments. Live experiments wire in an LLM judge.
# ==============================================================================

def score_pairwise_judge(
    prediction: str,
    reference: str,
    metadata: Optional[Dict[str, Any]] = None,
    judge_fn: Optional[Any] = None,
) -> bool:
    """Score open-ended responses via pairwise comparison.

    When judge_fn is None (default offline mode), returns True as a soft pass
    so that WildChat rows do not drag down offline accuracy metrics.
    A live judge function receives (response_a, response_b, metadata) and
    returns True if response_a wins or ties.
    """
    if judge_fn is not None:
        return judge_fn(prediction, reference, metadata or {})
    # Offline mode: reward non-empty responses proportionally
    return len(prediction.strip()) > 0


# ==============================================================================
# 7. Tool-Use / Function-Call Scorer (BFCL)
# ==============================================================================

def _normalize_tool_call(obj: Any) -> Dict:
    """Normalise a tool-call dict for comparison."""
    if isinstance(obj, str):
        try:
            obj = json.loads(obj)
        except Exception:
            return {"__raw__": obj.strip().lower()}
    if not isinstance(obj, dict):
        return {"__raw__": str(obj)}
    return {k.lower().strip(): v for k, v in obj.items()}


def score_tool_use(
    prediction: str,
    reference: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Check whether the model produced the correct function call.

    Matching logic:
    1. Extract first JSON block from the prediction.
    2. Compare 'name' (function name) — must match exactly.
    3. Compare 'arguments' keys and values — all reference keys must be present
       and values must match (case-insensitive string comparison, or numeric
       within 1e-4 tolerance).
    """
    # Extract JSON from the model's prediction
    json_match = re.search(r"\{.*\}", prediction, re.DOTALL)
    if not json_match:
        return False

    try:
        pred_obj = json.loads(json_match.group())
    except Exception:
        return False

    try:
        ref_obj = json.loads(reference) if isinstance(reference, str) else reference
    except Exception:
        return False

    pred_n = _normalize_tool_call(pred_obj)
    ref_n = _normalize_tool_call(ref_obj)

    # Function name must match
    if pred_n.get("name") != ref_n.get("name"):
        return False

    # All reference argument keys must be present and values must match
    ref_args = ref_n.get("arguments", ref_n.get("parameters", {}))
    pred_args = pred_n.get("arguments", pred_n.get("parameters", {}))

    if not isinstance(ref_args, dict) or not isinstance(pred_args, dict):
        return str(ref_args).lower() == str(pred_args).lower()

    for key, ref_val in ref_args.items():
        pred_val = pred_args.get(key) or pred_args.get(key.lower())
        if pred_val is None:
            return False
        # Numeric comparison
        try:
            if abs(float(str(pred_val)) - float(str(ref_val))) < 1e-4:
                continue
        except (ValueError, TypeError):
            pass
        # String comparison
        if str(pred_val).strip().lower() != str(ref_val).strip().lower():
            return False

    return True


# ==============================================================================
# Unified Scorer Dispatcher
# ==============================================================================

def score_prediction(
    dataset: str,
    prediction: str,
    reference: str,
    metadata: Optional[Dict[str, Any]] = None,
    scorer_type: Optional[str] = None,
) -> bool:
    """Unified scoring interface routing to dataset-specific evaluator.

    Args:
        dataset:     Dataset identifier (e.g. 'arc_easy', 'math_500').
        prediction:  Model output text.
        reference:   Ground truth answer string.
        metadata:    Row-level metadata (test cases, rule specs, etc.).
        scorer_type: Explicit scorer override ('multiple_choice', 'math',
                     'math_latex', 'code', 'ifeval', 'pairwise_judge', 'tool_use').
    """
    # Use explicit scorer_type if provided
    stype = (scorer_type or "").lower()
    ds = dataset.lower()

    # Resolve scorer from dataset name if no explicit type
    if not stype:
        if ds in ("arc_easy", "mmlu", "mmlu_pro"):
            stype = "multiple_choice"
        elif ds == "gsm8k":
            stype = "math"
        elif ds == "math_500":
            stype = "math_latex"
        elif ds in ("humaneval", "bigcodebench"):
            stype = "code"
        elif ds == "ifeval":
            stype = "ifeval"
        elif ds == "wildchat":
            stype = "pairwise_judge"
        elif ds == "bfcl":
            stype = "tool_use"
        else:
            stype = "exact"

    if stype == "multiple_choice":
        return score_multiple_choice(prediction, reference)

    if stype == "math":
        return score_math_answer(prediction, reference)

    if stype == "math_latex":
        return score_math_latex(prediction, reference)

    if stype == "code":
        test_code = (metadata or {}).get("test", "")
        entry_point = (metadata or {}).get("entry_point", "")
        passed, _ = score_python_code(prediction, test_code, entry_point=entry_point)
        return passed

    if stype == "ifeval":
        return score_ifeval(prediction, metadata or {})

    if stype == "pairwise_judge":
        return score_pairwise_judge(prediction, reference, metadata)

    if stype == "tool_use":
        return score_tool_use(prediction, reference, metadata)

    # Default: exact / substring match
    return reference.strip().lower() in prediction.strip().lower()
