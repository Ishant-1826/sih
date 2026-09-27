"""
Synthetic Trajectory Generator for IDR-GNSS-FUSION.

Generates realistic vehicle trajectories with:
- GPS coordinates following roads (simulated curves, straights, turns)
- IMU data (accel + gyro + magnetometer) with realistic noise
- Multiple driving scenarios (urban, highway, parking, etc.)

This allows full pipeline development and testing WITHOUT needing the
actual IO-VNBD dataset, which requires Git LFS download.

Usage:
    py ml/datasets/generate_sample.py
"""

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.datasets.io_vnbd import TrajectoryData


# ---- Constants ----
GRAVITY = 9.81  # m/s^2
EARTH_RADIUS = 6371000.0  # meters


def generate_trajectory(
    name: str = "synthetic_urban",
    duration: float = 300.0,        # seconds
    sample_rate: float = 10.0,      # Hz
    start_lat: float = 28.6139,     # New Delhi (India-relevant)
    start_lon: float = 77.2090,
    start_heading: float = 45.0,    # degrees
    max_speed_kmh: float = 60.0,
    scenario: str = "urban",        # urban | highway | parking
    accel_noise_std: float = 0.3,   # m/s^2
    gyro_noise_std: float = 0.01,   # rad/s
    gps_noise_std: float = 3.0,     # meters
    accel_bias: Optional[np.ndarray] = None,
    gyro_bias: Optional[np.ndarray] = None,
    seed: int = 42,
) -> TrajectoryData:
    """
    Generate a synthetic vehicle trajectory with realistic sensor data.

    The trajectory consists of segments: straight → turn → straight → turn ...
    with speed variations simulating real driving.
    """
    rng = np.random.RandomState(seed)
    dt = 1.0 / sample_rate
    n_samples = int(duration * sample_rate)

    if accel_bias is None:
        accel_bias = rng.uniform(-0.1, 0.1, size=3)
    if gyro_bias is None:
        gyro_bias = rng.uniform(-0.005, 0.005, size=3)

    # ---- Generate Speed Profile ----
    max_speed = max_speed_kmh / 3.6  # m/s

    if scenario == "highway":
        speed_profile = _gen_highway_speed(n_samples, max_speed, dt, rng)
    elif scenario == "parking":
        speed_profile = _gen_parking_speed(n_samples, max_speed * 0.3, dt, rng)
    else:  # urban
        speed_profile = _gen_urban_speed(n_samples, max_speed, dt, rng)

    # ---- Generate Heading Profile ----
    heading_profile = _gen_heading_profile(n_samples, start_heading, scenario, dt, rng)

    # ---- Compute True Position (lat/lon) ----
    heading_rad = np.radians(heading_profile)
    vx = speed_profile * np.sin(heading_rad)  # east component
    vy = speed_profile * np.cos(heading_rad)  # north component

    # Integrate velocity to get displacement in meters
    dx = np.cumsum(vx * dt)
    dy = np.cumsum(vy * dt)

    # Convert to lat/lon
    true_lat = start_lat + np.degrees(dy / EARTH_RADIUS)
    true_lon = start_lon + np.degrees(dx / (EARTH_RADIUS * np.cos(np.radians(start_lat))))

    # ---- Compute True Acceleration (vehicle frame) ----
    # Forward acceleration = d(speed)/dt
    fwd_accel = np.gradient(speed_profile, dt)
    # Lateral acceleration = v * d(heading)/dt
    heading_rate_rad = np.gradient(heading_rad, dt)
    lat_accel = speed_profile * heading_rate_rad

    # ---- Transform to phone frame (assume phone roughly aligned) ----
    # Simulate slight phone misalignment
    phone_roll = np.radians(rng.uniform(-5, 5))
    phone_pitch = np.radians(rng.uniform(-3, 3))
    phone_yaw = np.radians(rng.uniform(-10, 10))

    # Simplified: vehicle frame → phone frame with small rotation
    # x=forward, y=right, z=down in vehicle frame
    # Phone adds gravity component
    true_ax = fwd_accel * np.cos(phone_yaw) + lat_accel * np.sin(phone_yaw)
    true_ay = -fwd_accel * np.sin(phone_yaw) + lat_accel * np.cos(phone_yaw)
    true_az = np.full(n_samples, -GRAVITY)  # gravity in phone z-axis

    # Add phone tilt effects
    true_ax += GRAVITY * np.sin(phone_pitch)
    true_ay += GRAVITY * np.sin(phone_roll) * np.cos(phone_pitch)

    # ---- Generate Noisy IMU ----
    accel_x = true_ax + accel_bias[0] + rng.normal(0, accel_noise_std, n_samples)
    accel_y = true_ay + accel_bias[1] + rng.normal(0, accel_noise_std, n_samples)
    accel_z = true_az + accel_bias[2] + rng.normal(0, accel_noise_std, n_samples)

    # Gyroscope
    gyro_x = np.zeros(n_samples) + gyro_bias[0] + rng.normal(0, gyro_noise_std, n_samples)
    gyro_y = np.zeros(n_samples) + gyro_bias[1] + rng.normal(0, gyro_noise_std, n_samples)
    gyro_z = heading_rate_rad + gyro_bias[2] + rng.normal(0, gyro_noise_std, n_samples)

    # Magnetometer (simplified: heading-dependent + noise)
    mag_field_strength = 45.0  # μT (typical for India)
    mag_inclination = np.radians(39.0)  # magnetic inclination at Delhi
    mag_x = mag_field_strength * np.cos(mag_inclination) * np.cos(heading_rad) + \
            rng.normal(0, 2.0, n_samples)
    mag_y = mag_field_strength * np.cos(mag_inclination) * np.sin(heading_rad) + \
            rng.normal(0, 2.0, n_samples)
    mag_z = mag_field_strength * np.sin(mag_inclination) + \
            rng.normal(0, 2.0, n_samples)

    # ---- Generate Noisy GPS ----
    gps_noise_lat = rng.normal(0, gps_noise_std / 111320.0, n_samples)
    gps_noise_lon = rng.normal(0, gps_noise_std / (111320.0 * np.cos(np.radians(start_lat))),
                                n_samples)
    gps_lat = true_lat + gps_noise_lat
    gps_lon = true_lon + gps_noise_lon
    gps_velocity = speed_profile * 3.6 + rng.normal(0, 0.5, n_samples)  # km/h
    gps_velocity = np.maximum(gps_velocity, 0)
    gps_heading = heading_profile + rng.normal(0, 2.0, n_samples)
    gps_accuracy = np.abs(rng.normal(3.0, 1.5, n_samples))
    num_satellites = rng.randint(6, 14, n_samples)

    # ---- Build TrajectoryData ----
    timestamps = np.arange(n_samples) * dt

    traj = TrajectoryData(
        name=name,
        source='synthetic',
        timestamps=timestamps,
        accel_x=accel_x,
        accel_y=accel_y,
        accel_z=accel_z,
        gyro_x=gyro_x,
        gyro_y=gyro_y,
        gyro_z=gyro_z,
        mag_x=mag_x,
        mag_y=mag_y,
        mag_z=mag_z,
        orient_yaw=heading_profile,
        orient_pitch=np.full(n_samples, np.degrees(phone_pitch)),
        orient_roll=np.full(n_samples, np.degrees(phone_roll)),
        gps_lat=gps_lat,
        gps_lon=gps_lon,
        gps_alt=np.zeros(n_samples),
        gps_velocity=gps_velocity,
        gps_heading=gps_heading,
        gps_accuracy=gps_accuracy,
        num_satellites=num_satellites.astype(float),
        indicated_speed=speed_profile * 3.6,  # km/h
        yaw_rate=np.degrees(heading_rate_rad),
        sample_rate=sample_rate,
        duration=duration,
    )

    # Store ground truth as hidden attributes for evaluation
    traj._true_lat = true_lat
    traj._true_lon = true_lon
    traj._true_speed = speed_profile
    traj._true_heading = heading_profile

    traj.compute_gps_distance()

    return traj


def _gen_urban_speed(n: int, max_speed: float, dt: float, rng) -> np.ndarray:
    """Generate urban speed profile: stops, accelerations, cruising, braking."""
    speed = np.zeros(n)
    v = 0.0
    target_v = max_speed * 0.6
    state = 'accel'
    state_timer = 0

    for i in range(n):
        state_timer += 1

        if state == 'accel':
            v += rng.uniform(0.8, 2.0) * dt
            if v >= target_v:
                v = target_v
                state = 'cruise'
                state_timer = 0
        elif state == 'cruise':
            v += rng.normal(0, 0.1) * dt
            if state_timer > rng.randint(50, 200):
                decision = rng.random()
                if decision < 0.3:
                    state = 'brake'
                    state_timer = 0
                elif decision < 0.5:
                    target_v = rng.uniform(max_speed * 0.3, max_speed * 0.8)
                    state = 'accel' if target_v > v else 'brake'
                    state_timer = 0
        elif state == 'brake':
            v -= rng.uniform(1.0, 3.0) * dt
            if v <= 0:
                v = 0
                state = 'stop'
                state_timer = 0
        elif state == 'stop':
            v = 0
            if state_timer > rng.randint(20, 80):
                target_v = rng.uniform(max_speed * 0.3, max_speed * 0.8)
                state = 'accel'
                state_timer = 0

        speed[i] = max(0, min(v, max_speed))

    return speed


def _gen_highway_speed(n: int, max_speed: float, dt: float, rng) -> np.ndarray:
    """Highway profile: ramp up, mostly cruising at high speed."""
    speed = np.zeros(n)
    v = 0.0
    cruise_speed = max_speed * 0.85

    for i in range(n):
        if i < n * 0.1:
            v += 1.5 * dt
        elif i < n * 0.9:
            v = cruise_speed + rng.normal(0, 0.3) * dt
        else:
            v -= 1.0 * dt
        speed[i] = max(0, min(v, max_speed))

    return speed


def _gen_parking_speed(n: int, max_speed: float, dt: float, rng) -> np.ndarray:
    """Parking lot: very slow, frequent stops."""
    speed = np.zeros(n)
    v = 0.0

    for i in range(n):
        if rng.random() < 0.02:
            v = 0
        elif v == 0 and rng.random() < 0.05:
            v = rng.uniform(1, max_speed * 0.5)
        else:
            v += rng.normal(0, 0.3) * dt
        speed[i] = max(0, min(v, max_speed))

    return speed


def _gen_heading_profile(n: int, start_heading: float, scenario: str,
                         dt: float, rng) -> np.ndarray:
    """Generate heading profile with turns appropriate for scenario."""
    heading = np.zeros(n)
    h = start_heading

    if scenario == "highway":
        turn_prob = 0.005
        max_turn_rate = 5.0  # deg/s
    elif scenario == "parking":
        turn_prob = 0.03
        max_turn_rate = 30.0
    else:  # urban
        turn_prob = 0.01
        max_turn_rate = 15.0

    turn_rate = 0.0
    turn_timer = 0

    for i in range(n):
        if turn_timer > 0:
            turn_timer -= 1
            h += turn_rate * dt
        elif rng.random() < turn_prob:
            # Start a turn
            turn_rate = rng.uniform(-max_turn_rate, max_turn_rate)
            turn_timer = rng.randint(10, 40)  # samples

        # Small drift
        h += rng.normal(0, 0.1) * dt

        heading[i] = h % 360

    return heading


def save_trajectory_as_csv(traj: TrajectoryData, output_dir: str) -> str:
    """Save a TrajectoryData as CSV in IO-VNBD-compatible format."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"S-{traj.name}.csv")

    df = pd.DataFrame({
        'Time (ms)': (traj.timestamps * 1000).astype(int),
        'Accelerometer x (m/s2)': traj.accel_x,
        'Accelerometer y (m/s2)': traj.accel_y,
        'Accelerometer z (m/s2)': traj.accel_z,
        'Gyroscope x (rad/s)': traj.gyro_x,
        'Gyroscope y (rad/s)': traj.gyro_y,
        'Gyroscope z (rad/s)': traj.gyro_z,
        'Magnetic field x (uT)': traj.mag_x,
        'Magnetic field y (uT)': traj.mag_y,
        'Magnetic field z (uT)': traj.mag_z,
        'Orientation Yaw (deg)': traj.orient_yaw,
        'Orientation Pitch (deg)': traj.orient_pitch,
        'Orientation Roll (deg)': traj.orient_roll,
        'GPS Latitude (deg)': traj.gps_lat,
        'GPS Longitude (deg)': traj.gps_lon,
        'GPS Altitude (km)': traj.gps_alt,
        'GPS Velocity (km/hr)': traj.gps_velocity,
        'GPS Heading (deg)': traj.gps_heading,
        'GPS Accuracy (m)': traj.gps_accuracy,
        'Satellites': traj.num_satellites,
    })

    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df)} samples, {traj.duration:.0f}s)")
    return path


# ---- Main: Generate Sample Dataset ----

if __name__ == "__main__":
    print("=" * 60)
    print("IDR-GNSS-FUSION — Synthetic Dataset Generator")
    print("=" * 60)

    output_dir = str(Path(__file__).resolve().parent.parent.parent / "datasets" / "synthetic")

    scenarios = [
        ("urban_delhi_01",   300, "urban",   60, 28.6139, 77.2090, 42),
        ("urban_delhi_02",   400, "urban",   50, 28.6280, 77.2190, 123),
        ("highway_nh44_01",  600, "highway", 100, 28.7041, 77.1025, 456),
        ("parking_mall_01",  180, "parking", 15, 28.5563, 77.2400, 789),
        ("urban_mumbai_01",  350, "urban",   45, 19.0760, 72.8777, 101),
        ("tunnel_sim_01",    200, "urban",   60, 28.6350, 77.2280, 202),
    ]

    print(f"\nGenerating {len(scenarios)} synthetic trajectories...\n")

    for name, duration, scenario, max_speed, lat, lon, seed in scenarios:
        traj = generate_trajectory(
            name=name,
            duration=duration,
            scenario=scenario,
            max_speed_kmh=max_speed,
            start_lat=lat,
            start_lon=lon,
            seed=seed,
        )
        save_trajectory_as_csv(traj, output_dir)

    print(f"\n✓ All trajectories saved to: {output_dir}")
    print(f"  These files are in IO-VNBD-compatible CSV format.")
    print(f"  Use them for development. Replace with real IO-VNBD data later.")
