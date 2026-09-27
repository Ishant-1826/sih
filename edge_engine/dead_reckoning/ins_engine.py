"""
Dead Reckoning Engine for IDR-GNSS-FUSION.

Implements vehicle dead reckoning using:
- IMU integration (acceleration → velocity → position)
- Heading from gyroscope integration
- ZUPT (Zero Velocity Update) when stationary
- Non-holonomic constraints (NHC)
- Optional AI speed correction

This is the core INS propagation engine.
NOT a standalone filter — designed to be used with the fusion module.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from edge_engine.core.data_types import (
    NavigationState, NavigationMode, GNSSStatus,
    Orientation, latlon_to_meters, meters_to_latlon
)


@dataclass
class DRConfig:
    """Dead reckoning configuration."""
    enable_nhc: bool = True
    enable_zupt: bool = True
    nhc_lateral_limit: float = 0.5   # m/s max lateral velocity
    nhc_vertical_limit: float = 0.3  # m/s max vertical velocity
    zupt_accel_threshold: float = 0.3  # m/s^2
    zupt_gyro_threshold: float = 0.05  # rad/s
    max_acceleration: float = 5.0    # m/s^2 (vehicle constraint)
    max_speed: float = 50.0          # m/s (~180 km/h)


class DeadReckoningEngine:
    """
    Vehicle dead reckoning from IMU data.

    State vector:
        [pos_x, pos_y, vel_x, vel_y, heading, accel_bias_x, accel_bias_y, gyro_bias_z]

    Coordinate system:
        x = East, y = North (local tangent plane)
        heading = clockwise from North (degrees)
    """

    def __init__(self, config: Optional[DRConfig] = None):
        self.config = config or DRConfig()
        self.state = NavigationState()
        self.ref_lat = 0.0
        self.ref_lon = 0.0
        self._initialized = False
        self._history: list[NavigationState] = []

    def initialize(
        self,
        latitude: float,
        longitude: float,
        heading: float = 0.0,   # degrees
        speed: float = 0.0,     # m/s
        timestamp: float = 0.0,
    ):
        """Initialize dead reckoning with known position and heading."""
        self.ref_lat = latitude
        self.ref_lon = longitude

        self.state = NavigationState(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            position_x=0.0,
            position_y=0.0,
            velocity_x=speed * np.sin(np.radians(heading)),
            velocity_y=speed * np.cos(np.radians(heading)),
            speed=speed,
            heading=heading,
            orientation=Orientation.from_euler(0, 0, np.radians(heading)),
            mode=NavigationMode.DEAD_RECKONING,
            gnss_status=GNSSStatus.UNAVAILABLE,
            confidence=1.0,
            position_uncertainty=0.0,
            distance_travelled=0.0,
        )
        self._initialized = True
        self._history = [self._copy_state()]

    def propagate(
        self,
        accel_forward: float,   # m/s^2, forward acceleration (vehicle frame)
        accel_lateral: float,   # m/s^2, lateral acceleration
        yaw_rate: float,        # rad/s
        dt: float,              # seconds
        timestamp: float = 0.0,
        is_stationary: bool = False,
        ai_speed: Optional[float] = None,  # AI-estimated speed (m/s)
    ) -> NavigationState:
        """
        Propagate dead reckoning state by one time step.

        Args:
            accel_forward: Forward acceleration in vehicle frame
            accel_lateral: Lateral acceleration in vehicle frame
            yaw_rate: Yaw rate (heading change rate) in rad/s
            dt: Time step in seconds
            timestamp: Current timestamp
            is_stationary: ZUPT detection result
            ai_speed: Optional AI-predicted speed for correction

        Returns:
            Updated NavigationState
        """
        if not self._initialized:
            raise RuntimeError("DeadReckoningEngine not initialized. Call initialize() first.")

        # ---- Update heading ----
        heading_rad = np.radians(self.state.heading)
        heading_rad += yaw_rate * dt
        heading_deg = np.degrees(heading_rad) % 360

        # ---- ZUPT: Zero Velocity Update ----
        if is_stationary and self.config.enable_zupt:
            self.state.velocity_x = 0.0
            self.state.velocity_y = 0.0
            self.state.speed = 0.0
        else:
            # ---- Integrate acceleration to velocity ----
            # Transform vehicle-frame accel to world frame
            ax_world = accel_forward * np.sin(heading_rad) + accel_lateral * np.cos(heading_rad)
            ay_world = accel_forward * np.cos(heading_rad) - accel_lateral * np.sin(heading_rad)

            vx = self.state.velocity_x + ax_world * dt
            vy = self.state.velocity_y + ay_world * dt

            # ---- Apply Non-Holonomic Constraints ----
            if self.config.enable_nhc:
                # Vehicle lateral velocity should be ~0
                # Project velocity onto heading and cross-heading
                v_forward = vx * np.sin(heading_rad) + vy * np.cos(heading_rad)
                v_lateral = vx * np.cos(heading_rad) - vy * np.sin(heading_rad)

                # Constrain lateral velocity
                v_lateral = np.clip(v_lateral,
                                    -self.config.nhc_lateral_limit,
                                    self.config.nhc_lateral_limit)

                # Constrain forward speed
                v_forward = np.clip(v_forward, -2.0, self.config.max_speed)

                # Use AI speed if available (blend)
                if ai_speed is not None and ai_speed >= 0:
                    # Weighted blend: AI speed has 70% weight when available
                    alpha = 0.7
                    v_forward = alpha * ai_speed + (1 - alpha) * v_forward

                # Back to world frame
                vx = v_forward * np.sin(heading_rad) + v_lateral * np.cos(heading_rad)
                vy = v_forward * np.cos(heading_rad) - v_lateral * np.sin(heading_rad)

            self.state.velocity_x = vx
            self.state.velocity_y = vy
            self.state.speed = np.sqrt(vx**2 + vy**2)

        # ---- Integrate velocity to position ----
        dx = self.state.velocity_x * dt
        dy = self.state.velocity_y * dt
        self.state.position_x += dx
        self.state.position_y += dy

        # ---- Update lat/lon ----
        lat, lon = meters_to_latlon(
            self.state.position_x, self.state.position_y,
            self.ref_lat, self.ref_lon
        )
        self.state.latitude = lat
        self.state.longitude = lon

        # ---- Update heading ----
        self.state.heading = heading_deg
        self.state.orientation = Orientation.from_euler(0, 0, heading_rad)

        # ---- Update metadata ----
        self.state.timestamp = timestamp
        self.state.distance_travelled += np.sqrt(dx**2 + dy**2)
        self.state.mode = NavigationMode.DEAD_RECKONING

        # Confidence decays over time in DR mode
        decay_rate = 0.001  # per second
        self.state.confidence = max(0.1, self.state.confidence - decay_rate * dt)
        self.state.position_uncertainty += self.state.speed * dt * 0.05  # ~5% drift rate

        self._history.append(self._copy_state())

        return self._copy_state()

    def get_trajectory(self) -> list[NavigationState]:
        """Return full history of navigation states."""
        return self._history.copy()

    def get_trajectory_arrays(self) -> dict:
        """Return trajectory as numpy arrays for plotting."""
        if not self._history:
            return {}
        return {
            'timestamps': np.array([s.timestamp for s in self._history]),
            'lat': np.array([s.latitude for s in self._history]),
            'lon': np.array([s.longitude for s in self._history]),
            'pos_x': np.array([s.position_x for s in self._history]),
            'pos_y': np.array([s.position_y for s in self._history]),
            'speed': np.array([s.speed for s in self._history]),
            'heading': np.array([s.heading for s in self._history]),
            'confidence': np.array([s.confidence for s in self._history]),
            'uncertainty': np.array([s.position_uncertainty for s in self._history]),
            'distance': np.array([s.distance_travelled for s in self._history]),
        }

    def _copy_state(self) -> NavigationState:
        """Create a copy of current state."""
        import copy
        return copy.deepcopy(self.state)

    def reset(self):
        """Reset engine state."""
        self.state = NavigationState()
        self._initialized = False
        self._history = []


def run_dead_reckoning(
    timestamps: np.ndarray,
    accel: np.ndarray,    # (N, 3) vehicle frame: [forward, lateral, vertical]
    gyro: np.ndarray,     # (N, 3) [roll_rate, pitch_rate, yaw_rate]
    start_lat: float,
    start_lon: float,
    start_heading: float,
    start_speed: float = 0.0,
    stationary_mask: Optional[np.ndarray] = None,
    config: Optional[DRConfig] = None,
) -> dict:
    """
    Convenience function: run dead reckoning over entire trajectory.

    Returns dict with trajectory arrays.
    """
    engine = DeadReckoningEngine(config)
    engine.initialize(start_lat, start_lon, start_heading, start_speed, timestamps[0])

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0 or dt > 1.0:
            dt = 0.1  # fallback

        is_stat = stationary_mask[i] if stationary_mask is not None else False

        engine.propagate(
            accel_forward=accel[i, 0],   # forward
            accel_lateral=accel[i, 1],   # lateral
            yaw_rate=gyro[i, 2],         # yaw rate
            dt=dt,
            timestamp=timestamps[i],
            is_stationary=is_stat,
        )

    return engine.get_trajectory_arrays()
