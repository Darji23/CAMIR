"""Evaluation and scoring modules for CAMIR."""

from camir.evaluation.scorers import (
    score_prediction,
    score_multiple_choice,
    score_math_answer,
    score_python_code,
    score_instruction_following,
    extract_mc_prediction,
    extract_math_numerical_answer,
)

__all__ = [
    "score_prediction",
    "score_multiple_choice",
    "score_math_answer",
    "score_python_code",
    "score_instruction_following",
    "extract_mc_prediction",
    "extract_math_numerical_answer",
]
