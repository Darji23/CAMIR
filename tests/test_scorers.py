"""Unit tests for dataset scorers and ground truth extractors."""

import json
import pytest
from camir.evaluation.scorers import (
    extract_boxed_answer,
    extract_math_numerical_answer,
    extract_mc_prediction,
    score_ifeval,
    score_instruction_following,
    score_math_answer,
    score_math_latex,
    score_multiple_choice,
    score_prediction,
    score_python_code,
    score_pairwise_judge,
    score_tool_use,
)


def test_mc_extraction_and_scoring():
    # Leading format
    assert extract_mc_prediction("A") == "A"
    assert extract_mc_prediction("(B)") == "B"
    assert extract_mc_prediction("C. This is correct.") == "C"

    # Embedded format
    assert extract_mc_prediction("Therefore, the correct answer is (D).") == "D"
    assert extract_mc_prediction("The answer is B") == "B"

    # Scoring
    assert score_multiple_choice("The answer is (A)", "A") is True
    assert score_multiple_choice("The answer is (B)", "A") is False
    assert score_multiple_choice("", "A") is False


def test_math_extraction_and_scoring():
    # GSM8K standard delimiter
    assert extract_math_numerical_answer("Some steps... #### 42") == 42.0
    assert extract_math_numerical_answer("#### 1,250.50") == 1250.50

    # Conversational answers
    assert extract_math_numerical_answer("Therefore, the total answer is $15.") == 15.0
    assert extract_math_numerical_answer("It equals 3.14.") == 3.14

    # Scoring
    assert score_math_answer("Final result is #### 42", "42") is True
    assert score_math_answer("Final result is #### 42", "43") is False
    assert score_math_answer("The answer is 12.50", "#### 12.5") is True


def test_python_code_sandbox():
    good_code = (
        "def add(a, b):\n"
        "    return a + b\n"
    )
    test_code = (
        "assert add(2, 3) == 5\n"
        "assert add(-1, 1) == 0\n"
    )
    passed, msg = score_python_code(good_code, test_code)
    assert passed is True

    bad_code = (
        "def add(a, b):\n"
        "    return a * b\n"
    )
    passed_bad, msg_bad = score_python_code(bad_code, test_code)
    assert passed_bad is False
    assert "AssertionError" in msg_bad


def test_ifeval_instruction_scoring():
    # Min words rule
    assert score_instruction_following("one two three four five six", "min_words", 5) is True
    assert score_instruction_following("one two three", "min_words", 5) is False

    # Max words rule
    assert score_instruction_following("short answer", "max_words", 5) is True
    assert score_instruction_following("this sentence contains way too many words here", "max_words", 5) is False

    # JSON format rule
    assert score_instruction_following('{"key": "value", "count": 10}', "json_format", None) is True
    assert score_instruction_following('not a json string', "json_format", None) is False

    # All lowercase rule
    assert score_instruction_following("all lowercase text here", "all_lowercase", None) is True
    assert score_instruction_following("Has Capital Letters", "all_lowercase", None) is False

    # Bullet list rule
    assert score_instruction_following("- item one\n- item two\n- item three", "bullet_list", 3) is True
    assert score_instruction_following("just a sentence", "bullet_list", 3) is False

    # Starts with rule
    assert score_instruction_following("The quick brown fox", "starts_with", "The") is True
    assert score_instruction_following("Quick brown fox", "starts_with", "The") is False

    # No comma rule
    assert score_instruction_following("a simple sentence without commas", "no_comma", "true") is True
    assert score_instruction_following("hello, world", "no_comma", "true") is False

    # IFEval-style instruction IDs (keywords:existence)
    assert score_instruction_following("This uses an algorithm", "keywords:existence", "algorithm") is True
    assert score_instruction_following("This has nothing special", "keywords:existence", "algorithm") is False

    # length_constraint:number_words
    assert score_instruction_following(
        "one two three four five",
        "length_constraint:number_words",
        {"relation": "at least", "num_words": 5}
    ) is True
    assert score_instruction_following(
        "one two",
        "length_constraint:number_words",
        {"relation": "at least", "num_words": 5}
    ) is False


def test_math_latex_scorer():
    # Boxed extraction
    assert extract_boxed_answer(r"Therefore \boxed{42}") == "42"
    assert extract_boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert extract_boxed_answer("No box here #### 99") == "99"
    assert extract_boxed_answer("No answer") is None

    # Scoring — exact boxed match
    assert score_math_latex(r"\boxed{42}", r"The answer is \boxed{42}.") is True
    assert score_math_latex(r"\boxed{42}", r"The answer is \boxed{43}.") is False

    # Scoring — numerical tolerance
    assert score_math_latex(r"\boxed{3.14159}", r"\boxed{3.14159}") is True

    # Fallback to numerical extraction
    assert score_math_latex("The answer is 100.", "#### 100") is True


def test_ifeval_full_row_scoring():
    # Synthetic format (uses rule_type/rule_arg in metadata)
    meta_synth = {"rule_type": "min_words", "rule_arg": 3}
    assert score_ifeval("one two three four", meta_synth) is True
    assert score_ifeval("one", meta_synth) is False

    # Real IFEval format (instruction_id_list + kwargs)
    meta_real = {
        "instruction_id_list": ["keywords:existence", "length_constraint:number_words"],
        "kwargs": [
            {"keyword": "python"},
            {"relation": "at least", "num_words": 5},
        ]
    }
    assert score_ifeval("python is a great language for scripting", meta_real) is True
    assert score_ifeval("java is good", meta_real) is False  # no "python" keyword


def test_pairwise_judge_scorer():
    # Offline mode: non-empty response passes
    assert score_pairwise_judge("The answer is 42.", "") is True
    assert score_pairwise_judge("", "") is False

    # With a custom judge function
    def perfect_judge(pred, ref, meta):
        return "correct" in pred.lower()

    assert score_pairwise_judge("This is the correct answer.", "ref", judge_fn=perfect_judge) is True
    assert score_pairwise_judge("This is wrong.", "ref", judge_fn=perfect_judge) is False


def test_tool_use_scorer():
    ref = json.dumps({"name": "get_weather", "arguments": {"location": "New York"}})

    # Exact match
    pred_good = '{"name": "get_weather", "arguments": {"location": "New York"}}'
    assert score_tool_use(pred_good, ref) is True

    # Wrong function name
    pred_wrong_fn = '{"name": "get_stock_price", "arguments": {"location": "New York"}}'
    assert score_tool_use(pred_wrong_fn, ref) is False

    # Wrong argument value
    pred_wrong_arg = '{"name": "get_weather", "arguments": {"location": "London"}}'
    assert score_tool_use(pred_wrong_arg, ref) is False

    # No JSON in response
    assert score_tool_use("I cannot call functions.", ref) is False

    # Case-insensitive value matching
    pred_case = '{"name": "get_weather", "arguments": {"location": "new york"}}'
    assert score_tool_use(pred_case, ref) is True


def test_score_prediction_dispatcher():
    # Verify the dispatcher correctly routes to each scorer
    assert score_prediction("arc_easy", "A", "A") is True
    assert score_prediction("mmlu", "B", "A") is False
    assert score_prediction("gsm8k", "#### 42", "42") is True
    assert score_prediction("math_500", r"\boxed{10}", r"\boxed{10}") is True
    assert score_prediction("wildchat", "some text", "") is True  # pairwise offline
    ref_tool = json.dumps({"name": "f", "arguments": {"x": "1"}})
    assert score_prediction("bfcl", '{"name": "f", "arguments": {"x": "1"}}', ref_tool) is True

    # Explicit scorer_type override
    assert score_prediction("custom_ds", "(C)", "C", scorer_type="multiple_choice") is True
