"""Statistical guarantees, conformal risk calibration, and regime map analysis for CAMIR."""

from camir.guarantees.adaptive import AdaptiveRecalibrator, AuditRecord
from camir.guarantees.conformal import (
    ConformalCalibrationResult,
    calibrate_conformal_threshold,
    calibrate_p5_conformal_pair,
    compute_loss,
)
from camir.guarantees.regime_map import (
    RegimePoint,
    calculate_optimal_regime,
    generate_regime_grid,
)

__all__ = [
    "ConformalCalibrationResult",
    "calibrate_conformal_threshold",
    "calibrate_p5_conformal_pair",
    "compute_loss",
    "AdaptiveRecalibrator",
    "AuditRecord",
    "RegimePoint",
    "calculate_optimal_regime",
    "generate_regime_grid",
]
