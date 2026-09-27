"""
Evaluation Metrics for IDR-GNSS-FUSION.

Computes navigation performance metrics:
- Position error (RMSE, MAE, max, percentiles)
- Drift percentage (error / distance travelled)
- Heading error
- Velocity error
- Per-outage-region analysis
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from edge_engine.core.data_types import haversine_distance


@dataclass
class NavigationMetrics:
    """Complete evaluation metrics for a navigation run."""
    # Position error (meters)
    mean_position_error: float = 0.0
    rmse_position: float = 0.0
    mae_position: float = 0.0
    max_position_error: float = 0.0
    p95_position_error: float = 0.0
    p99_position_error: float = 0.0

    # Drift
    total_distance: float = 0.0
    final_error: float = 0.0
    drift_percentage: float = 0.0  # error / distance * 100

    # Heading error (degrees)
    mean_heading_error: float = 0.0
    rmse_heading: float = 0.0

    # Velocity error (m/s)
    mean_velocity_error: float = 0.0
    rmse_velocity: float = 0.0

    # Metadata
    num_samples: int = 0
    outage_duration: float = 0.0
    outage_distance: float = 0.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 50,
            "NAVIGATION PERFORMANCE METRICS",
            "=" * 50,
            f"Samples:              {self.num_samples}",
            f"Total Distance:       {self.total_distance:.1f} m",
            f"Outage Duration:      {self.outage_duration:.1f} s",
            "",
            "Position Error:",
            f"  Mean:               {self.mean_position_error:.2f} m",
            f"  RMSE:               {self.rmse_position:.2f} m",
            f"  MAE:                {self.mae_position:.2f} m",
            f"  Max:                {self.max_position_error:.2f} m",
            f"  95th percentile:    {self.p95_position_error:.2f} m",
            f"  Final error:        {self.final_error:.2f} m",
            "",
            f"Drift:                {self.drift_percentage:.2f}%",
            "",
            f"Heading Error:",
            f"  Mean:               {self.mean_heading_error:.2f}°",
            f"  RMSE:               {self.rmse_heading:.2f}°",
            "",
            f"Velocity Error:",
            f"  Mean:               {self.mean_velocity_error:.2f} m/s",
            f"  RMSE:               {self.rmse_velocity:.2f} m/s",
            "=" * 50,
        ]
        return "\n".join(lines)


def compute_position_errors(
    true_lat: np.ndarray,
    true_lon: np.ndarray,
    est_lat: np.ndarray,
    est_lon: np.ndarray,
) -> np.ndarray:
    """Compute per-sample position error in meters using Haversine."""
    n = min(len(true_lat), len(est_lat))
    errors = np.zeros(n)
    for i in range(n):
        if true_lat[i] != 0 and true_lon[i] != 0 and est_lat[i] != 0 and est_lon[i] != 0:
            errors[i] = haversine_distance(true_lat[i], true_lon[i],
                                            est_lat[i], est_lon[i])
    return errors


def compute_heading_errors(
    true_heading: np.ndarray,
    est_heading: np.ndarray,
) -> np.ndarray:
    """Compute heading errors, handling 0/360 wraparound."""
    n = min(len(true_heading), len(est_heading))
    diff = true_heading[:n] - est_heading[:n]
    # Wrap to [-180, 180]
    diff = (diff + 180) % 360 - 180
    return np.abs(diff)


def evaluate_navigation(
    true_lat: np.ndarray,
    true_lon: np.ndarray,
    est_lat: np.ndarray,
    est_lon: np.ndarray,
    true_speed: Optional[np.ndarray] = None,
    est_speed: Optional[np.ndarray] = None,
    true_heading: Optional[np.ndarray] = None,
    est_heading: Optional[np.ndarray] = None,
    gnss_available: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
) -> NavigationMetrics:
    """
    Comprehensive navigation evaluation.

    If gnss_available mask is provided, metrics are computed only for
    outage periods (where gnss_available == False).
    """
    n = min(len(true_lat), len(est_lat))

    # Optionally restrict to outage regions
    if gnss_available is not None:
        mask = ~gnss_available[:n]
    else:
        mask = np.ones(n, dtype=bool)

    # Position errors
    pos_errors = compute_position_errors(true_lat[:n], true_lon[:n],
                                          est_lat[:n], est_lon[:n])
    pos_errors_masked = pos_errors[mask]

    # Total ground truth distance
    total_dist = 0.0
    for i in range(1, n):
        if true_lat[i] != 0 and true_lon[i] != 0:
            total_dist += haversine_distance(true_lat[i-1], true_lon[i-1],
                                              true_lat[i], true_lon[i])

    # Outage distance (distance travelled during outage)
    outage_dist = 0.0
    if gnss_available is not None:
        for i in range(1, n):
            if not gnss_available[i] and true_lat[i] != 0:
                outage_dist += haversine_distance(true_lat[i-1], true_lon[i-1],
                                                    true_lat[i], true_lon[i])

    metrics = NavigationMetrics(
        num_samples=int(np.sum(mask)),
        total_distance=total_dist,
        outage_distance=outage_dist,
    )

    if len(pos_errors_masked) > 0:
        metrics.mean_position_error = float(np.mean(pos_errors_masked))
        metrics.rmse_position = float(np.sqrt(np.mean(pos_errors_masked**2)))
        metrics.mae_position = float(np.mean(np.abs(pos_errors_masked)))
        metrics.max_position_error = float(np.max(pos_errors_masked))
        metrics.p95_position_error = float(np.percentile(pos_errors_masked, 95))
        metrics.p99_position_error = float(np.percentile(pos_errors_masked, 99))
        metrics.final_error = float(pos_errors_masked[-1]) if len(pos_errors_masked) > 0 else 0.0

        if outage_dist > 0:
            metrics.drift_percentage = (metrics.mean_position_error / outage_dist) * 100
        elif total_dist > 0:
            metrics.drift_percentage = (metrics.mean_position_error / total_dist) * 100

    # Heading error
    if true_heading is not None and est_heading is not None:
        head_err = compute_heading_errors(true_heading[:n], est_heading[:n])
        head_err_masked = head_err[mask]
        if len(head_err_masked) > 0:
            metrics.mean_heading_error = float(np.mean(head_err_masked))
            metrics.rmse_heading = float(np.sqrt(np.mean(head_err_masked**2)))

    # Velocity error
    if true_speed is not None and est_speed is not None:
        vel_err = np.abs(true_speed[:n] - est_speed[:n])
        vel_err_masked = vel_err[mask]
        if len(vel_err_masked) > 0:
            metrics.mean_velocity_error = float(np.mean(vel_err_masked))
            metrics.rmse_velocity = float(np.sqrt(np.mean(vel_err_masked**2)))

    # Outage duration
    if gnss_available is not None and timestamps is not None:
        outage_times = timestamps[:n][~gnss_available[:n]]
        if len(outage_times) > 0:
            metrics.outage_duration = float(outage_times[-1] - outage_times[0])

    return metrics
