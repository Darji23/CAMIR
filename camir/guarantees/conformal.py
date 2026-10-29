"""Conformal Risk Control and Quality Guarantees for CAMIR (Contribution C3).

Provides finite-sample statistical risk bounds on relative loss:
    $\\ell(x) = \\mathbf{1}[\\text{accepted by router} \\wedge \\text{small model incorrect} \\wedge \\text{large model correct}]$

Guarantees:
    $\\mathbb{E}[\\ell(x)] \\le \\alpha$
with finite-sample conformal slack $\\frac{1}{n+1}$.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class ConformalCalibrationResult:
    """Result of conformal risk calibration over calibration traces."""
    threshold: float
    target_alpha: float
    empirical_risk: float
    sample_size_n: int
    conformal_slack: float
    acceptance_rate: float
    guarantee_satisfied: bool
    risk_curve: Dict[float, float]  # threshold -> empirical risk


def compute_loss(
    confidence: float,
    threshold: float,
    small_correct: bool,
    large_correct: bool,
) -> int:
    """Compute relative routing loss: 1 if accepted by router, small is wrong, large was right."""
    accepted = (confidence >= threshold)
    if accepted and (not small_correct) and large_correct:
        return 1
    return 0


def calibrate_conformal_threshold(
    calibration_records: List[Dict[str, Any]],
    target_alpha: float = 0.05,
    min_threshold: float = -5.0,
    max_threshold: float = 0.0,
    num_grid_points: int = 100,
) -> ConformalCalibrationResult:
    """Compute monotonic conformal threshold tau guaranteeing expected relative loss <= alpha.
    
    Args:
        calibration_records: List of dicts with:
            - 'confidence': float (min_logprob or confidence score)
            - 'small_correct': bool
            - 'large_correct': bool
        target_alpha: Maximum permissible relative error rate (e.g., 0.05 for 5%)
    
    Returns:
        ConformalCalibrationResult with finite-sample statistical validity.
    """
    n = len(calibration_records)
    if n == 0:
        raise ValueError("Cannot calibrate on empty calibration set.")

    slack = 1.0 / (n + 1)
    # Conservative target accounting for finite-sample calibration slack
    adjusted_alpha = max(0.001, target_alpha - slack)

    thresholds = np.linspace(min_threshold, max_threshold, num_grid_points)
    risk_curve: Dict[float, float] = {}
    chosen_tau: Optional[float] = None
    chosen_risk: float = 1.0
    chosen_acc_rate: float = 0.0

    # Test thresholds in descending order (from strict/high tau to permissive/low tau)
    # Higher tau = fewer accepted = strictly lower or equal risk
    for tau in sorted(thresholds, reverse=True):
        tau_val = round(float(tau), 4)
        losses = []
        accepted_count = 0

        for r in calibration_records:
            conf = r["confidence"]
            s_corr = r["small_correct"]
            l_corr = r.get("large_correct", True)

            l_i = compute_loss(conf, tau_val, s_corr, l_corr)
            losses.append(l_i)
            if conf >= tau_val:
                accepted_count += 1

        emp_risk = float(np.mean(losses))
        risk_curve[tau_val] = round(emp_risk, 4)

        # Conformal selection rule: select the most permissive (lowest) threshold
        # whose empirical risk + slack remains bounded by target_alpha
        if emp_risk <= adjusted_alpha:
            chosen_tau = tau_val
            chosen_risk = emp_risk
            chosen_acc_rate = accepted_count / n

    if chosen_tau is None:
        # If no threshold satisfies alpha, set the most conservative threshold
        chosen_tau = float(max_threshold)
        chosen_risk = risk_curve.get(chosen_tau, 0.0)
        chosen_acc_rate = sum(1 for r in calibration_records if r["confidence"] >= chosen_tau) / n

    return ConformalCalibrationResult(
        threshold=chosen_tau,
        target_alpha=target_alpha,
        empirical_risk=round(chosen_risk, 4),
        sample_size_n=n,
        conformal_slack=round(slack, 4),
        acceptance_rate=round(chosen_acc_rate, 4),
        guarantee_satisfied=(chosen_risk <= target_alpha),
        risk_curve=risk_curve,
    )


def calibrate_p5_conformal_pair(
    calibration_records: List[Dict[str, Any]],
    target_alpha: float = 0.05,
) -> Tuple[float, float, ConformalCalibrationResult]:
    """Jointly calibrate primary cascade gate threshold tau and verifier threshold lambda for P5.
    
    Enforces that the acceptance decision stays monotone in both signals:
    accept if conf >= tau OR (conf < tau AND v_score >= lambda).
    """
    n = len(calibration_records)
    slack = 1.0 / (n + 1)
    adjusted_alpha = max(0.001, target_alpha - slack)

    tau_candidates = np.linspace(-2.5, -0.2, 25)
    lambda_candidates = np.linspace(-4.0, -0.5, 25)

    best_pair = (-0.5, -1.0)
    best_risk = 1.0
    best_acc = 0.0

    for tau in tau_candidates:
        for lam in lambda_candidates:
            losses = []
            accepted = 0
            for r in calibration_records:
                conf = r["confidence"]
                v_score = r.get("verifier_score", -99.0)
                s_corr = r["small_correct"]
                l_corr = r.get("large_correct", True)

                is_acc = (conf >= tau) or (v_score >= lam)
                if is_acc:
                    accepted += 1
                    l_i = 1 if (not s_corr and l_corr) else 0
                else:
                    l_i = 0
                losses.append(l_i)

            emp_risk = float(np.mean(losses))
            if emp_risk <= adjusted_alpha and (accepted / n) > best_acc:
                best_pair = (round(float(tau), 3), round(float(lam), 3))
                best_risk = emp_risk
                best_acc = accepted / n

    res = ConformalCalibrationResult(
        threshold=best_pair[0],
        target_alpha=target_alpha,
        empirical_risk=round(best_risk, 4),
        sample_size_n=n,
        conformal_slack=round(slack, 4),
        acceptance_rate=round(best_acc, 4),
        guarantee_satisfied=(best_risk <= target_alpha),
        risk_curve={},
    )
    return best_pair[0], best_pair[1], res
