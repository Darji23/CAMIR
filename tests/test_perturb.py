"""Unit tests for deterministic perturbation generator and invariants."""

import re
import pytest
from camir.data.perturb import (
    generate_noisy_twins,
    generate_perturbed_prompt,
    perturb_gsm8k,
    perturb_humaneval,
    perturb_multiple_choice,
    perturb_prose_typos,
    perturb_prose_casing_punct,
    perturb_prose_filler,
)


def test_perturb_determinism():
    """Identical seed must yield identical perturbation; distinct seeds must differ."""
    text = "The quick brown fox jumps over the lazy dog."
    p1 = perturb_prose_typos(text, seed=123)
    p2 = perturb_prose_typos(text, seed=123)
    p3 = perturb_prose_typos(text, seed=999)

    assert p1 == p2, "Perturbation with identical seeds must be deterministic"
    assert p1 != p3, "Perturbation with different seeds should differ"


def test_multiple_choice_invariant():
    """MC options, letters, and labels must remain completely verbatim."""
    prompt = (
        "Question: Which organism is a primary producer in an ecosystem?\n"
        "Choices:\n"
        "(A) Green plant\n"
        "(B) Red fox\n"
        "(C) Earthworm\n"
        "(D) Mushroom\n"
        "Answer:"
    )
    for ntype in ["typos", "casing_punct", "filler"]:
        perturbed = perturb_multiple_choice(prompt, noise_type=ntype, seed=42)
        assert "(A) Green plant" in perturbed
        assert "(B) Red fox" in perturbed
        assert "(C) Earthworm" in perturbed
        assert "(D) Mushroom" in perturbed
        assert "Answer:" in perturbed


def test_gsm8k_number_preservation_invariant():
    """All digits, numbers, decimals, and monetary amounts must remain completely intact."""
    prompt = "Janet has 16 apples and 4.5 pounds of flour. She buys $12 worth of sugar."
    numbers_original = re.findall(r"\b\d+(?:\.\d+)?\b|\$\d+", prompt)

    for ntype in ["typos", "casing_punct", "filler"]:
        perturbed = perturb_gsm8k(prompt, noise_type=ntype, seed=42)
        numbers_perturbed = re.findall(r"\b\d+(?:\.\d+)?\b|\$\d+", perturbed)
        assert numbers_original == numbers_perturbed, (
            f"Number invariant broken for {ntype}: {numbers_original} vs {numbers_perturbed}"
        )


def test_humaneval_code_signature_invariant():
    """HumanEval function signature, def line, and return type must remain completely untouched."""
    prompt = (
        'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n'
        '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
        '    given threshold.\n'
        '    """\n'
    )
    for ntype in ["typos", "casing_punct", "filler"]:
        perturbed = perturb_humaneval(prompt, noise_type=ntype, seed=42)
        assert 'def has_close_elements(numbers: List[float], threshold: float) -> bool:' in perturbed
        assert '"""' in perturbed


def test_group_aware_twins_generation():
    """Clean prompt and all derived noisy twins must share the exact same group_id."""
    clean_item = {
        "prompt_id": "arc_001",
        "dataset": "arc_easy",
        "prompt": "Question: What is gravity?\nChoices:\n(A) Force\n(B) Color\nAnswer:",
        "reference_answer": "A",
    }
    twins = generate_noisy_twins(clean_item, noise_types=["typos", "filler"], seed=42)
    assert len(twins) == 2

    for twin in twins:
        assert twin["group_id"] == "arc_001"
        assert twin["clean_prompt_id"] == "arc_001"
        assert twin["is_perturbed"] is True
        assert twin["prompt_id"].startswith("arc_001_noise_")
        assert twin["reference_answer"] == "A"
