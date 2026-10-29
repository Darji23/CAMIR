"""Unit tests for MoE signals, AUROC, and calibration metrics (Contribution C2 / E2)."""

import numpy as np
import pytest
from camir.signals.moe import (
    calculate_auroc,
    calculate_ece,
    calculate_entropy,
    calculate_expert_instability,
    calculate_gate_margin,
    extract_moe_signals,
)


def test_entropy_properties():
    # Uniform distribution over 4 experts has maximal entropy log(4) = ~1.386 nats
    uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
    ent_uniform = calculate_entropy(uniform_probs)
    assert round(ent_uniform, 3) == 1.386

    # One-hot distribution has 0 entropy
    one_hot = np.array([1.0, 0.0, 0.0, 0.0])
    ent_one_hot = calculate_entropy(one_hot)
    assert ent_one_hot == 0.0


def test_gate_margin():
    logits = np.array([5.0, 3.0, 1.0, 0.0])
    margin = calculate_gate_margin(logits)
    assert margin == 2.0


def test_expert_instability():
    # If consecutive layers pick completely disjoint experts, distance = 1.0
    layers_disjoint = [[0, 1], [2, 3], [4, 5]]
    instab_disjoint = calculate_expert_instability(layers_disjoint)
    assert instab_disjoint == 1.0

    # If all layers pick identical experts, distance = 0.0
    layers_identical = [[0, 1], [0, 1], [0, 1]]
    instab_identical = calculate_expert_instability(layers_identical)
    assert instab_identical == 0.0


def test_extract_moe_signals():
    l1 = np.array([3.0, 2.0, 0.5, 0.1])
    l2 = np.array([2.5, 2.0, 0.8, 0.2])
    signals = extract_moe_signals([l1, l2], top_k=2)
    assert signals.router_entropy > 0.0
    assert signals.gate_margin > 0.0
    assert len(signals.selected_experts) == 2


def test_auroc_metric():
    # Perfect predictor: high failure risk score corresponds to true errors (1)
    scores = [0.9, 0.8, 0.7, 0.2, 0.1]
    labels = [1, 1, 1, 0, 0]
    auroc = calculate_auroc(scores, labels)
    assert auroc == 1.0


def test_ece_metric():
    # Perfect calibration: confidence 0.8 is correct 80% of the time, 0.2 is correct 20%
    confidences = [0.8] * 10 + [0.2] * 10
    labels = [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8
    ece = calculate_ece(confidences, labels, n_bins=5)
    assert ece <= 0.05
