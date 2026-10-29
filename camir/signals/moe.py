"""Mixture-of-Experts (MoE) uncertainty signal extraction and evaluation (Contribution C2 / E2).

Extracts:
1. Router Entropy: H_r = -sum(p_k * log(p_k)) across expert gate probabilities.
2. Top-2 Gate Logit Margin: M_r = g_(1) - g_(2).
3. Inter-Layer Routing Instability: Jaccard distance between active expert sets across layers.
4. Computes AUROC and Expected Calibration Error (ECE) comparing C2 signals against baselines.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel


class MoERouterSignals(BaseModel):
    """Container for MoE internal routing signals extracted from a query completion."""
    router_entropy: float
    gate_margin: float
    expert_instability: float
    selected_experts: List[List[int]]  # Experts per layer [layer][expert_idx]


def calculate_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy in nats from an expert probability distribution."""
    # Filter out zero probabilities to avoid log(0)
    p = probs[probs > 1e-12]
    return float(-np.sum(p * np.log(p)))


def calculate_gate_margin(logits: np.ndarray) -> float:
    """Compute margin between top-1 and top-2 gate logits."""
    if len(logits) < 2:
        return 0.0
    sorted_logits = np.sort(logits)[::-1]
    return float(sorted_logits[0] - sorted_logits[1])


def calculate_expert_instability(layer_experts: List[List[int]]) -> float:
    """Compute average Jaccard distance between active expert sets across consecutive layers."""
    if len(layer_experts) < 2:
        return 0.0

    distances = []
    for l in range(len(layer_experts) - 1):
        s1 = set(layer_experts[l])
        s2 = set(layer_experts[l + 1])
        union = s1.union(s2)
        if not union:
            continue
        inter = s1.intersection(s2)
        jaccard_dist = 1.0 - (len(inter) / len(union))
        distances.append(jaccard_dist)

    return float(np.mean(distances)) if distances else 0.0


def extract_moe_signals(
    router_logits_per_layer: List[np.ndarray],
    top_k: int = 2,
) -> MoERouterSignals:
    """Extract composite MoE uncertainty signals from layer-wise router logits."""
    if not router_logits_per_layer:
        return MoERouterSignals(
            router_entropy=0.0,
            gate_margin=0.0,
            expert_instability=0.0,
            selected_experts=[],
        )

    entropies = []
    margins = []
    layer_experts = []

    for logits in router_logits_per_layer:
        # Softmax over expert logits
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        entropies.append(calculate_entropy(probs))
        margins.append(calculate_gate_margin(logits))

        # Selected top-k experts for this layer
        top_k_indices = np.argsort(logits)[::-1][:top_k].tolist()
        layer_experts.append(top_k_indices)

    mean_entropy = float(np.mean(entropies))
    mean_margin = float(np.mean(margins))
    instability = calculate_expert_instability(layer_experts)

    return MoERouterSignals(
        router_entropy=round(mean_entropy, 4),
        gate_margin=round(mean_margin, 4),
        expert_instability=round(instability, 4),
        selected_experts=layer_experts,
    )


def calculate_auroc(scores: List[float], labels: List[int]) -> float:
    """Calculate Area Under the ROC Curve (AUROC) for error prediction.

    labels: 1 if small tier made an error (failure), 0 if correct.
    scores: higher score indicates higher predicted risk of failure.
    """
    if len(scores) != len(labels) or len(set(labels)) < 2:
        return 0.50

    # Sort scores descending
    order = np.argsort(scores)[::-1]
    sorted_labels = np.array(labels)[order]

    num_pos = np.sum(sorted_labels == 1)
    num_neg = len(labels) - num_pos

    if num_pos == 0 or num_neg == 0:
        return 0.50

    # Trapezoidal integration for ROC curve
    tpr = np.cumsum(sorted_labels == 1) / num_pos
    fpr = np.cumsum(sorted_labels == 0) / num_neg

    auroc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    return round(abs(auroc), 4)


def calculate_ece(confidences: List[float], labels: List[int], n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE) across confidence bins.

    confidences: predicted probability of correctness in [0, 1].
    labels: 1 if prediction was correct, 0 if incorrect.
    """
    if not confidences or len(confidences) != len(labels):
        return 0.0

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1

    ece = 0.0
    total_samples = len(confidences)

    for i in range(n_bins):
        in_bin = bin_indices == i
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            avg_confidence = np.mean(np.array(confidences)[in_bin])
            avg_accuracy = np.mean(np.array(labels)[in_bin])
            ece += (bin_size / total_samples) * abs(avg_accuracy - avg_confidence)

    return round(float(ece), 4)
