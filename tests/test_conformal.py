"""Unit tests for Conformal Risk Calibrator and Adaptive Recalibration (Contribution C3, X3)."""

import pytest
from camir.guarantees.adaptive import AdaptiveRecalibrator
from camir.guarantees.conformal import (
    calibrate_conformal_threshold,
    calibrate_p5_conformal_pair,
    compute_loss,
)


def test_conformal_loss_computation():
    # Accepted and small was wrong while large was right -> loss = 1
    assert compute_loss(confidence=-0.4, threshold=-0.5, small_correct=False, large_correct=True) == 1
    # Accepted and small was correct -> loss = 0
    assert compute_loss(confidence=-0.4, threshold=-0.5, small_correct=True, large_correct=True) == 0
    # Rejected -> escalated to large -> loss = 0
    assert compute_loss(confidence=-0.8, threshold=-0.5, small_correct=False, large_correct=True) == 0


def test_conformal_risk_calibration_guarantee():
    """Verify that calibrated tau bounds empirical risk within target_alpha."""
    # Synthetic calibration set: 100 samples
    records = []
    for i in range(100):
        conf = - (i / 50.0)  # -0.0 to -2.0
        # For low confidence (< -0.8), small model is mostly wrong
        s_corr = (conf >= -0.80)
        records.append({
            "confidence": conf,
            "small_correct": s_corr,
            "large_correct": True,
        })

    target_alpha = 0.08
    res = calibrate_conformal_threshold(records, target_alpha=target_alpha)

    assert res.sample_size_n == 100
    assert res.conformal_slack == round(1.0 / 101, 4)
    assert res.guarantee_satisfied is True
    assert res.empirical_risk <= target_alpha


def test_p5_joint_conformal_calibration():
    records = []
    for i in range(80):
        conf = - (i / 40.0)
        v_score = conf + 0.2
        records.append({
            "confidence": conf,
            "verifier_score": v_score,
            "small_correct": (conf >= -1.0),
            "large_correct": True,
        })

    tau, lam, res = calibrate_p5_conformal_pair(records, target_alpha=0.10)
    assert isinstance(tau, float)
    assert isinstance(lam, float)
    assert res.guarantee_satisfied is True


def test_adaptive_recalibrator_drift_response():
    recal = AdaptiveRecalibrator(initial_tau=-0.50, target_alpha=0.05, audit_rate_gamma=1.0)
    
    # Simulate high error stream (drift where small model fails frequently)
    for i in range(25):
        recal.process_query(
            query_id=f"q_{i}",
            confidence=-0.40,  # would accept
            small_correct=False,  # but wrong!
            large_correct=True,
            audit_cost_usd=0.0001,
        )

    summary = recal.get_audit_summary()
    assert summary["total_audits_performed"] == 25
    assert summary["total_audit_cost_usd"] > 0.0
    # Threshold should adapt upward (become more conservative / closer to 0.0)
    assert recal.current_tau > -0.50
