"""
GNSS Outage Simulator for IDR-GNSS-FUSION.

Artificially removes, degrades, or corrupts GNSS data from trajectories
to simulate real-world GNSS-denied scenarios:
- Tunnel (complete blackout)
- Urban canyon (intermittent degradation)
- Underpass (short blackout)
- Multi-level parking (extended blackout + degradation)
- Jamming (sudden loss + recovery)

This is essential for training, evaluating, and demonstrating
dead reckoning performance.
"""

from dataclasses import dataclass, field
from typing import Optional
import copy

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.datasets.io_vnbd import TrajectoryData


@dataclass
class OutageRegion:
    """Defines a single GNSS outage or degradation region."""
    start_time: float       # seconds from trajectory start
    end_time: float         # seconds
    outage_type: str = "blackout"  # blackout | degraded | intermittent
    description: str = ""

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class OutageScenario:
    """A complete GNSS outage scenario with multiple regions."""
    name: str
    description: str
    regions: list[OutageRegion] = field(default_factory=list)

    @property
    def total_outage_duration(self) -> float:
        return sum(r.duration for r in self.regions)


def create_tunnel_scenario(
    trajectory_duration: float,
    tunnel_start_fraction: float = 0.35,
    tunnel_duration: float = 60.0,
) -> OutageScenario:
    """
    Simulate a long tunnel: complete GNSS blackout.

    Args:
        trajectory_duration: Total trajectory length in seconds.
        tunnel_start_fraction: Where tunnel starts (fraction of trajectory).
        tunnel_duration: Duration of tunnel traversal in seconds.
    """
    start = trajectory_duration * tunnel_start_fraction
    return OutageScenario(
        name="tunnel",
        description=f"Complete GNSS blackout for {tunnel_duration:.0f}s "
                    f"(simulating tunnel/underpass)",
        regions=[
            OutageRegion(
                start_time=start,
                end_time=start + tunnel_duration,
                outage_type="blackout",
                description="Tunnel"
            )
        ]
    )


def create_urban_canyon_scenario(
    trajectory_duration: float,
    num_outages: int = 5,
    seed: int = 42,
) -> OutageScenario:
    """
    Simulate urban canyon: frequent short degradations and blackouts.
    """
    rng = np.random.RandomState(seed)
    regions = []

    for i in range(num_outages):
        start = rng.uniform(trajectory_duration * 0.1, trajectory_duration * 0.9)
        duration = rng.uniform(5, 30)
        outage_type = rng.choice(["blackout", "degraded", "intermittent"],
                                  p=[0.3, 0.4, 0.3])
        regions.append(OutageRegion(
            start_time=start,
            end_time=min(start + duration, trajectory_duration),
            outage_type=outage_type,
            description=f"Urban canyon segment {i+1}"
        ))

    regions.sort(key=lambda r: r.start_time)

    return OutageScenario(
        name="urban_canyon",
        description=f"Urban canyon with {num_outages} degradation zones",
        regions=regions
    )


def create_progressive_scenario(
    trajectory_duration: float,
) -> OutageScenario:
    """
    Scenario for benchmarking: short → medium → long outages.
    Perfect for SIH demonstration.
    """
    t = trajectory_duration
    return OutageScenario(
        name="progressive",
        description="Progressive outages: 3s → 10s → 30s → 60s",
        regions=[
            OutageRegion(t*0.1, t*0.1 + 3, "blackout", "Short tunnel (3s)"),
            OutageRegion(t*0.25, t*0.25 + 10, "blackout", "Underpass (10s)"),
            OutageRegion(t*0.45, t*0.45 + 30, "blackout", "Medium tunnel (30s)"),
            OutageRegion(t*0.7, t*0.7 + 60, "blackout", "Long tunnel (60s)"),
        ]
    )


def create_custom_scenario(
    outage_specs: list[tuple[float, float, str]],
) -> OutageScenario:
    """
    Create custom outage scenario from (start_s, end_s, type) tuples.

    Example:
        create_custom_scenario([
            (100, 160, "blackout"),
            (200, 220, "degraded"),
        ])
    """
    regions = [
        OutageRegion(s, e, t, f"Custom outage at {s:.0f}s")
        for s, e, t in outage_specs
    ]
    return OutageScenario(
        name="custom",
        description=f"Custom scenario with {len(regions)} outage regions",
        regions=regions
    )


def apply_outage(
    trajectory: TrajectoryData,
    scenario: OutageScenario,
    seed: int = 42,
) -> tuple[TrajectoryData, np.ndarray]:
    """
    Apply GNSS outage scenario to a trajectory.

    Returns:
        - Modified trajectory with GNSS data removed/degraded in outage regions
        - Boolean mask: True where GNSS is available, False where outage applied

    The original trajectory is NOT modified (deep copy is made).
    """
    rng = np.random.RandomState(seed)
    traj = _deep_copy_trajectory(trajectory)
    n = traj.num_samples
    gnss_available = np.ones(n, dtype=bool)

    t0 = traj.timestamps[0] if n > 0 else 0

    for region in scenario.regions:
        abs_start = t0 + region.start_time
        abs_end = t0 + region.end_time

        # Find indices in this region
        mask = (traj.timestamps >= abs_start) & (traj.timestamps <= abs_end)
        idx = np.where(mask)[0]

        if len(idx) == 0:
            continue

        if region.outage_type == "blackout":
            # Complete removal of GNSS data
            traj.gps_lat[idx] = 0.0
            traj.gps_lon[idx] = 0.0
            traj.gps_velocity[idx] = 0.0
            traj.gps_heading[idx] = 0.0
            traj.gps_accuracy[idx] = float('inf')
            traj.num_satellites[idx] = 0.0
            gnss_available[idx] = False

        elif region.outage_type == "degraded":
            # Keep GPS but add large noise and reduce satellites
            noise_lat = rng.normal(0, 50 / 111320.0, len(idx))
            noise_lon = rng.normal(0, 50 / 111320.0, len(idx))
            traj.gps_lat[idx] += noise_lat
            traj.gps_lon[idx] += noise_lon
            traj.gps_accuracy[idx] = rng.uniform(15, 50, len(idx))
            traj.num_satellites[idx] = rng.choice([2, 3, 4], len(idx)).astype(float)
            # Mark as degraded but still "available" (filter should detect)
            gnss_available[idx] = True

        elif region.outage_type == "intermittent":
            # Random drops within the region
            for j in idx:
                if rng.random() < 0.6:  # 60% of samples dropped
                    traj.gps_lat[j] = 0.0
                    traj.gps_lon[j] = 0.0
                    traj.gps_velocity[j] = 0.0
                    traj.gps_accuracy[j] = float('inf')
                    traj.num_satellites[j] = 0.0
                    gnss_available[j] = False

    return traj, gnss_available


def _deep_copy_trajectory(traj: TrajectoryData) -> TrajectoryData:
    """Create a deep copy of trajectory data (copies all arrays)."""
    return TrajectoryData(
        name=traj.name + "_outage",
        source=traj.source,
        timestamps=traj.timestamps.copy(),
        accel_x=traj.accel_x.copy(),
        accel_y=traj.accel_y.copy(),
        accel_z=traj.accel_z.copy(),
        gyro_x=traj.gyro_x.copy(),
        gyro_y=traj.gyro_y.copy(),
        gyro_z=traj.gyro_z.copy(),
        mag_x=traj.mag_x.copy(),
        mag_y=traj.mag_y.copy(),
        mag_z=traj.mag_z.copy(),
        orient_yaw=traj.orient_yaw.copy(),
        orient_pitch=traj.orient_pitch.copy(),
        orient_roll=traj.orient_roll.copy(),
        gps_lat=traj.gps_lat.copy(),
        gps_lon=traj.gps_lon.copy(),
        gps_alt=traj.gps_alt.copy(),
        gps_velocity=traj.gps_velocity.copy(),
        gps_heading=traj.gps_heading.copy(),
        gps_accuracy=traj.gps_accuracy.copy(),
        num_satellites=traj.num_satellites.copy(),
        indicated_speed=traj.indicated_speed.copy() if len(traj.indicated_speed) > 0 else np.array([]),
        yaw_rate=traj.yaw_rate.copy() if len(traj.yaw_rate) > 0 else np.array([]),
        sample_rate=traj.sample_rate,
        duration=traj.duration,
        total_distance=traj.total_distance,
    )


def get_outage_timeline(
    timestamps: np.ndarray,
    gnss_available: np.ndarray,
) -> list[dict]:
    """
    Convert boolean GNSS availability mask to a list of state transitions.

    Returns list of {time, state: 'available'|'outage'} dicts.
    """
    timeline = []
    current_state = gnss_available[0]
    timeline.append({
        'time': float(timestamps[0]),
        'state': 'available' if current_state else 'outage'
    })

    for i in range(1, len(gnss_available)):
        if gnss_available[i] != current_state:
            current_state = gnss_available[i]
            timeline.append({
                'time': float(timestamps[i]),
                'state': 'available' if current_state else 'outage'
            })

    return timeline
