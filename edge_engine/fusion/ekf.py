"""
Extended Kalman Filter (EKF) for GNSS/INS Fusion.

State vector (9-dimensional):
    [pos_x, pos_y, vel_x, vel_y, heading, accel_bias_x, accel_bias_y, gyro_bias_z, speed_scale]

Measurement models:
    - GNSS position update (lat/lon → x/y)
    - GNSS velocity update
    - GNSS heading update
    - ZUPT (zero velocity) update
    - NHC (non-holonomic constraint) update
    - AI speed update (when available)

The EKF runs at IMU rate, with GNSS updates at 1 Hz.
During GNSS outage, only prediction + NHC + ZUPT + AI updates occur.

Reference:
    Groves, P.D. "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems" (2nd ed.), Artech House, 2013.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from edge_engine.core.data_types import (
    NavigationState, NavigationMode, GNSSStatus,
    Orientation, latlon_to_meters, meters_to_latlon
)


@dataclass
class EKFConfig:
    """EKF tuning parameters."""
    # Process noise
    accel_noise: float = 0.5        # m/s^2 (acceleration process noise)
    gyro_noise: float = 0.01        # rad/s (gyro process noise)
    accel_bias_noise: float = 0.001  # m/s^2/√s (bias random walk)
    gyro_bias_noise: float = 0.0001  # rad/s/√s
    speed_scale_noise: float = 0.0001  # scale factor drift

    # GNSS measurement noise
    gnss_pos_noise: float = 5.0     # meters
    gnss_vel_noise: float = 0.5     # m/s
    gnss_heading_noise: float = 5.0  # degrees

    # Constraint noise
    nhc_lateral_noise: float = 0.1   # m/s
    nhc_vertical_noise: float = 0.1  # m/s
    zupt_noise: float = 0.05        # m/s

    # AI speed measurement noise
    ai_speed_noise: float = 1.0     # m/s

    # Vehicle constraints
    max_speed: float = 50.0         # m/s
    enable_nhc: bool = True
    enable_zupt: bool = True


class EKFNavigationFilter:
    """
    Extended Kalman Filter for GNSS/INS/AI sensor fusion.

    Maintains a 9-state estimate with full covariance tracking.
    Designed for real-time operation at IMU rate (~10-200 Hz).
    """

    # State indices
    PX, PY = 0, 1        # Position (meters, local frame)
    VX, VY = 2, 3        # Velocity (m/s)
    HDG = 4              # Heading (radians)
    ABX, ABY = 5, 6      # Accelerometer bias x, y
    GBZ = 7              # Gyro bias z (yaw)
    SSC = 8              # Speed scale factor

    N_STATES = 9

    def __init__(self, config: Optional[EKFConfig] = None):
        self.config = config or EKFConfig()

        # State vector
        self.x = np.zeros(self.N_STATES)
        self.x[self.SSC] = 1.0  # Initial speed scale = 1

        # Covariance matrix
        self.P = np.eye(self.N_STATES)
        self.P[self.PX, self.PX] = 10.0**2    # 10m initial uncertainty
        self.P[self.PY, self.PY] = 10.0**2
        self.P[self.VX, self.VX] = 5.0**2     # 5 m/s
        self.P[self.VY, self.VY] = 5.0**2
        self.P[self.HDG, self.HDG] = np.radians(30)**2  # 30 deg
        self.P[self.ABX, self.ABX] = 0.5**2   # 0.5 m/s^2
        self.P[self.ABY, self.ABY] = 0.5**2
        self.P[self.GBZ, self.GBZ] = 0.05**2  # 0.05 rad/s
        self.P[self.SSC, self.SSC] = 0.1**2   # 10% scale uncertainty

        # Reference coordinates
        self.ref_lat = 0.0
        self.ref_lon = 0.0

        # Tracking
        self._initialized = False
        self._history: list[dict] = []
        self._gnss_status = GNSSStatus.UNAVAILABLE
        self._mode = NavigationMode.GNSS_AIDED
        self._distance_travelled = 0.0
        self._last_gnss_time = -float('inf')

    def initialize(
        self,
        latitude: float,
        longitude: float,
        heading_deg: float = 0.0,
        speed: float = 0.0,
        timestamp: float = 0.0,
    ):
        """Initialize the filter with known position and heading."""
        self.ref_lat = latitude
        self.ref_lon = longitude

        self.x[self.PX] = 0.0
        self.x[self.PY] = 0.0
        self.x[self.VX] = speed * np.sin(np.radians(heading_deg))
        self.x[self.VY] = speed * np.cos(np.radians(heading_deg))
        self.x[self.HDG] = np.radians(heading_deg)
        self.x[self.ABX] = 0.0
        self.x[self.ABY] = 0.0
        self.x[self.GBZ] = 0.0
        self.x[self.SSC] = 1.0

        # Reset covariance
        self.P = np.eye(self.N_STATES) * 0.01
        self.P[self.PX, self.PX] = 3.0**2
        self.P[self.PY, self.PY] = 3.0**2
        self.P[self.VX, self.VX] = 1.0**2
        self.P[self.VY, self.VY] = 1.0**2
        self.P[self.HDG, self.HDG] = np.radians(10)**2
        self.P[self.ABX, self.ABX] = 0.3**2
        self.P[self.ABY, self.ABY] = 0.3**2
        self.P[self.GBZ, self.GBZ] = 0.02**2
        self.P[self.SSC, self.SSC] = 0.05**2

        self._initialized = True
        self._last_timestamp = timestamp
        self._distance_travelled = 0.0
        self._gnss_status = GNSSStatus.AVAILABLE
        self._mode = NavigationMode.GNSS_AIDED

        self._record_state(timestamp)

    def predict(
        self,
        accel_forward: float,
        accel_lateral: float,
        yaw_rate: float,
        dt: float,
        timestamp: float,
    ):
        """
        EKF prediction step using IMU measurements.

        Propagates state using the nonlinear vehicle motion model
        and updates covariance with linearized state transition.
        """
        if not self._initialized:
            return

        if dt <= 0 or dt > 2.0:
            return

        # Extract current state
        heading = self.x[self.HDG]
        vx = self.x[self.VX]
        vy = self.x[self.VY]
        ab_x = self.x[self.ABX]
        ab_y = self.x[self.ABY]
        gb_z = self.x[self.GBZ]
        ssc = self.x[self.SSC]

        # Correct measurements with estimated biases
        af_corrected = (accel_forward - ab_x) * ssc
        al_corrected = (accel_lateral - ab_y) * ssc
        yr_corrected = yaw_rate - gb_z

        # ---- Nonlinear state prediction ----
        sin_h = np.sin(heading)
        cos_h = np.cos(heading)

        # Heading update
        new_heading = heading + yr_corrected * dt

        # Velocity update (vehicle→world frame transform)
        ax_world = af_corrected * sin_h + al_corrected * cos_h
        ay_world = af_corrected * cos_h - al_corrected * sin_h

        new_vx = vx + ax_world * dt
        new_vy = vy + ay_world * dt

        # Position update
        new_px = self.x[self.PX] + vx * dt + 0.5 * ax_world * dt**2
        new_py = self.x[self.PY] + vy * dt + 0.5 * ay_world * dt**2

        # Update state vector
        self.x[self.PX] = new_px
        self.x[self.PY] = new_py
        self.x[self.VX] = new_vx
        self.x[self.VY] = new_vy
        self.x[self.HDG] = new_heading

        # Track distance
        self._distance_travelled += np.sqrt((new_vx*dt)**2 + (new_vy*dt)**2)

        # ---- Jacobian of state transition (F matrix) ----
        F = np.eye(self.N_STATES)

        # ∂pos/∂vel
        F[self.PX, self.VX] = dt
        F[self.PY, self.VY] = dt

        # ∂pos/∂heading
        F[self.PX, self.HDG] = (af_corrected * cos_h - al_corrected * sin_h) * 0.5 * dt**2
        F[self.PY, self.HDG] = (-af_corrected * sin_h - al_corrected * cos_h) * 0.5 * dt**2

        # ∂vel/∂heading
        F[self.VX, self.HDG] = (af_corrected * cos_h - al_corrected * sin_h) * dt
        F[self.VY, self.HDG] = (-af_corrected * sin_h - al_corrected * cos_h) * dt

        # ∂vel/∂accel_bias
        F[self.VX, self.ABX] = -ssc * sin_h * dt
        F[self.VX, self.ABY] = -ssc * cos_h * dt
        F[self.VY, self.ABX] = -ssc * cos_h * dt
        F[self.VY, self.ABY] = ssc * sin_h * dt

        # ∂heading/∂gyro_bias
        F[self.HDG, self.GBZ] = -dt

        # ∂vel/∂speed_scale
        F[self.VX, self.SSC] = ((accel_forward - ab_x) * sin_h + 
                                 (accel_lateral - ab_y) * cos_h) * dt
        F[self.VY, self.SSC] = ((accel_forward - ab_x) * cos_h - 
                                 (accel_lateral - ab_y) * sin_h) * dt

        # ---- Process noise ----
        Q = np.zeros((self.N_STATES, self.N_STATES))
        Q[self.PX, self.PX] = (self.config.accel_noise * dt**2 / 2)**2
        Q[self.PY, self.PY] = (self.config.accel_noise * dt**2 / 2)**2
        Q[self.VX, self.VX] = (self.config.accel_noise * dt)**2
        Q[self.VY, self.VY] = (self.config.accel_noise * dt)**2
        Q[self.HDG, self.HDG] = (self.config.gyro_noise * dt)**2
        Q[self.ABX, self.ABX] = (self.config.accel_bias_noise * dt)**2
        Q[self.ABY, self.ABY] = (self.config.accel_bias_noise * dt)**2
        Q[self.GBZ, self.GBZ] = (self.config.gyro_bias_noise * dt)**2
        Q[self.SSC, self.SSC] = (self.config.speed_scale_noise * dt)**2

        # Covariance prediction
        self.P = F @ self.P @ F.T + Q

        # Ensure symmetry
        self.P = 0.5 * (self.P + self.P.T)

        self._last_timestamp = timestamp

    def update_gnss_position(
        self,
        latitude: float,
        longitude: float,
        accuracy: float = 5.0,
        timestamp: float = 0.0,
    ):
        """GNSS position measurement update."""
        if not self._initialized or latitude == 0 or longitude == 0:
            return

        # Convert measurement to local frame
        mx, my = latlon_to_meters(latitude, longitude, self.ref_lat, self.ref_lon)

        # Measurement vector: [x, y]
        z = np.array([mx, my])

        # Predicted measurement
        z_pred = np.array([self.x[self.PX], self.x[self.PY]])

        # Innovation
        y = z - z_pred

        # Measurement Jacobian (H)
        H = np.zeros((2, self.N_STATES))
        H[0, self.PX] = 1.0
        H[1, self.PY] = 1.0

        # Measurement noise (use reported accuracy)
        R = np.eye(2) * max(accuracy, self.config.gnss_pos_noise)**2

        # Kalman gain
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return  # Skip update if singular

        # State update
        self.x += K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

        self._gnss_status = GNSSStatus.AVAILABLE
        self._mode = NavigationMode.GNSS_AIDED
        self._last_gnss_time = timestamp

    def update_gnss_velocity(
        self,
        speed: float,       # m/s
        heading_deg: float,  # degrees
        timestamp: float = 0.0,
    ):
        """GNSS velocity and heading measurement update."""
        if not self._initialized:
            return

        heading_rad = np.radians(heading_deg)
        vx_meas = speed * np.sin(heading_rad)
        vy_meas = speed * np.cos(heading_rad)

        # Measurement: [vx, vy]
        z = np.array([vx_meas, vy_meas])
        z_pred = np.array([self.x[self.VX], self.x[self.VY]])
        y = z - z_pred

        H = np.zeros((2, self.N_STATES))
        H[0, self.VX] = 1.0
        H[1, self.VY] = 1.0

        R = np.eye(2) * self.config.gnss_vel_noise**2

        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.x += K @ y
        I_KH = np.eye(self.N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_gnss_heading(self, heading_deg: float):
        """GNSS heading measurement update."""
        if not self._initialized:
            return

        heading_rad = np.radians(heading_deg)

        # Innovation with angle wrapping
        z = heading_rad
        z_pred = self.x[self.HDG]
        y = z - z_pred
        # Wrap to [-pi, pi]
        y = (y + np.pi) % (2 * np.pi) - np.pi

        H = np.zeros((1, self.N_STATES))
        H[0, self.HDG] = 1.0

        R = np.array([[np.radians(self.config.gnss_heading_noise)**2]])

        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.x += (K @ np.array([y])).flatten()
        I_KH = np.eye(self.N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_nhc(self):
        """
        Non-Holonomic Constraint update.

        Constrains lateral velocity to ~0 (vehicles don't slide sideways).
        This is the single most important constraint for vehicle DR.
        """
        if not self._initialized or not self.config.enable_nhc:
            return

        heading = self.x[self.HDG]
        sin_h = np.sin(heading)
        cos_h = np.cos(heading)

        # Lateral velocity in vehicle frame
        v_lat = self.x[self.VX] * cos_h - self.x[self.VY] * sin_h

        # Measurement: lateral velocity should be 0
        z = np.array([0.0])
        z_pred = np.array([v_lat])
        y = z - z_pred

        # Jacobian
        H = np.zeros((1, self.N_STATES))
        H[0, self.VX] = cos_h
        H[0, self.VY] = -sin_h
        # Heading affects the lateral velocity projection
        H[0, self.HDG] = -(self.x[self.VX] * sin_h + self.x[self.VY] * cos_h)

        R = np.array([[self.config.nhc_lateral_noise**2]])

        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.x += (K @ y).flatten()
        I_KH = np.eye(self.N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_zupt(self):
        """
        Zero Velocity Update.

        When vehicle is detected as stationary, constrain velocity to 0.
        """
        if not self._initialized or not self.config.enable_zupt:
            return

        z = np.array([0.0, 0.0])
        z_pred = np.array([self.x[self.VX], self.x[self.VY]])
        y = z - z_pred

        H = np.zeros((2, self.N_STATES))
        H[0, self.VX] = 1.0
        H[1, self.VY] = 1.0

        R = np.eye(2) * self.config.zupt_noise**2

        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.x += (K @ y).flatten()
        I_KH = np.eye(self.N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def update_ai_speed(self, speed: float):
        """
        AI-predicted speed measurement update.

        Used during GNSS outage to constrain velocity magnitude.
        """
        if not self._initialized or speed < 0:
            return

        # Current speed estimate
        current_speed = np.sqrt(self.x[self.VX]**2 + self.x[self.VY]**2)

        if current_speed < 0.01:
            return  # Avoid division by zero

        # Measurement: speed magnitude
        z = np.array([speed])
        z_pred = np.array([current_speed])
        y = z - z_pred

        # Jacobian of speed w.r.t. state
        H = np.zeros((1, self.N_STATES))
        H[0, self.VX] = self.x[self.VX] / current_speed
        H[0, self.VY] = self.x[self.VY] / current_speed

        R = np.array([[self.config.ai_speed_noise**2]])

        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        self.x += (K @ y).flatten()
        I_KH = np.eye(self.N_STATES) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T

    def get_state(self, timestamp: float = 0.0) -> dict:
        """Get current navigation state as dict."""
        lat, lon = meters_to_latlon(
            self.x[self.PX], self.x[self.PY],
            self.ref_lat, self.ref_lon
        )
        speed = np.sqrt(self.x[self.VX]**2 + self.x[self.VY]**2)
        heading = np.degrees(self.x[self.HDG]) % 360

        pos_uncertainty = np.sqrt(self.P[self.PX, self.PX] + self.P[self.PY, self.PY])

        return {
            'timestamp': timestamp,
            'latitude': lat,
            'longitude': lon,
            'position_x': self.x[self.PX],
            'position_y': self.x[self.PY],
            'velocity_x': self.x[self.VX],
            'velocity_y': self.x[self.VY],
            'speed': speed,
            'heading': heading,
            'accel_bias_x': self.x[self.ABX],
            'accel_bias_y': self.x[self.ABY],
            'gyro_bias_z': self.x[self.GBZ],
            'speed_scale': self.x[self.SSC],
            'position_uncertainty': pos_uncertainty,
            'mode': self._mode.name,
            'gnss_status': self._gnss_status.name,
            'distance_travelled': self._distance_travelled,
        }

    def _record_state(self, timestamp: float):
        """Record current state in history."""
        self._history.append(self.get_state(timestamp))

    def get_trajectory_arrays(self) -> dict:
        """Return trajectory history as numpy arrays."""
        if not self._history:
            return {}
        return {
            'timestamps': np.array([s['timestamp'] for s in self._history]),
            'lat': np.array([s['latitude'] for s in self._history]),
            'lon': np.array([s['longitude'] for s in self._history]),
            'speed': np.array([s['speed'] for s in self._history]),
            'heading': np.array([s['heading'] for s in self._history]),
            'uncertainty': np.array([s['position_uncertainty'] for s in self._history]),
            'distance': np.array([s['distance_travelled'] for s in self._history]),
        }

    def step(
        self,
        accel_forward: float,
        accel_lateral: float,
        yaw_rate: float,
        dt: float,
        timestamp: float,
        gnss_lat: Optional[float] = None,
        gnss_lon: Optional[float] = None,
        gnss_speed: Optional[float] = None,
        gnss_heading: Optional[float] = None,
        gnss_accuracy: Optional[float] = None,
        is_stationary: bool = False,
        ai_speed: Optional[float] = None,
    ) -> dict:
        """
        Complete EKF step: predict + all applicable updates.

        This is the main interface for running the filter.
        """
        # ---- Prediction ----
        self.predict(accel_forward, accel_lateral, yaw_rate, dt, timestamp)

        # ---- GNSS Updates ----
        gnss_valid = (gnss_lat is not None and gnss_lon is not None and
                      gnss_lat != 0 and gnss_lon != 0)

        if gnss_valid:
            accuracy = gnss_accuracy if gnss_accuracy is not None else self.config.gnss_pos_noise
            if accuracy < 50:  # Only trust reasonable accuracy
                self.update_gnss_position(gnss_lat, gnss_lon, accuracy, timestamp)

                if gnss_speed is not None and gnss_speed >= 0:
                    if gnss_heading is not None:
                        self.update_gnss_velocity(gnss_speed, gnss_heading, timestamp)
                    if gnss_heading is not None:
                        self.update_gnss_heading(gnss_heading)

                self._gnss_status = GNSSStatus.AVAILABLE
                self._mode = NavigationMode.GNSS_AIDED
        else:
            self._gnss_status = GNSSStatus.UNAVAILABLE
            self._mode = NavigationMode.DEAD_RECKONING

        # ---- Constraint Updates ----
        if is_stationary:
            self.update_zupt()
        else:
            self.update_nhc()

        # ---- AI Speed Update ----
        if ai_speed is not None and ai_speed >= 0:
            self.update_ai_speed(ai_speed)

        # Speed limit enforcement
        speed = np.sqrt(self.x[self.VX]**2 + self.x[self.VY]**2)
        if speed > self.config.max_speed:
            scale = self.config.max_speed / speed
            self.x[self.VX] *= scale
            self.x[self.VY] *= scale

        state = self.get_state(timestamp)
        self._history.append(state)
        return state
