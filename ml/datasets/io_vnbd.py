"""
IO-VNBD Dataset Loader.

Loads synchronized smartphone (S-Dataset) and vehicle (V-Dataset) CSV files
from the IO-VNBD benchmark dataset. Provides a uniform interface for accessing
IMU, GNSS, and ground truth data across trajectories.

Dataset structure (expected after download/extraction):
    datasets/
        io_vnbd/
            Synchronised V abd S datasets/
                Uncategorised IOVNB Dataset/
                    S-Dataset/  ← smartphone sensor data
                    V-Dataset/  ← vehicle ECU data

Reference:
    IO-VNBD: Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning
    Data in Brief, Volume 35, 106885 (2021)
    DOI: 10.1016/j.dib.2021.106885
    GitHub: https://github.com/onyekpeu/IO-VNBD
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---- Expected Column Mappings ----

# Smartphone dataset columns (may vary slightly between files)
S_COLUMN_PATTERNS = {
    'timestamp_ms': r'time.*\(ms\)|time.*ms',
    'date': r'date',
    'accel_x': r'accelerometer.*x|accel.*x|ax',
    'accel_y': r'accelerometer.*y|accel.*y|ay',
    'accel_z': r'accelerometer.*z|accel.*z|az',
    'gravity_x': r'gravity.*x|grav.*x',
    'gravity_y': r'gravity.*y|grav.*y',
    'gravity_z': r'gravity.*z|grav.*z',
    'gyro_yaw': r'gyroscope.*yaw|gyro.*yaw|gyro.*z',
    'gyro_pitch': r'gyroscope.*pitch|gyro.*pitch|gyro.*y',
    'gyro_roll': r'gyroscope.*roll|gyro.*roll|gyro.*x',
    'mag_x': r'magnetic.*x|mag.*x',
    'mag_y': r'magnetic.*y|mag.*y',
    'mag_z': r'magnetic.*z|mag.*z',
    'orient_yaw': r'orientation.*yaw|orient.*yaw|heading.*deg',
    'orient_pitch': r'orientation.*pitch|orient.*pitch',
    'orient_roll': r'orientation.*roll|orient.*roll',
    'gps_lat': r'gps.*lat|latitude',
    'gps_lon': r'gps.*lon|longitude',
    'gps_alt': r'gps.*alt|altitude',
    'gps_velocity': r'gps.*velocity|gps.*speed',
    'gps_heading': r'gps.*heading|gps.*bearing',
    'gps_vert_vel': r'gps.*vertical.*vel',
    'gps_accuracy': r'gps.*accuracy|accuracy',
    'num_satellites': r'satellite|num.*sat',
}

# Vehicle dataset columns
V_COLUMN_PATTERNS = {
    'timestamp_s': r'time.*\(s\)|time.*sec|time$',
    'long_accel': r'long.*accel|longitudinal.*accel',
    'lat_accel': r'lat.*accel|lateral.*accel',
    'yaw_rate': r'yaw.*rate',
    'gps_heading': r'gps.*heading|heading',
    'gps_lat': r'gps.*lat|latitude',
    'gps_lon': r'gps.*lon|longitude',
    'gps_velocity': r'gps.*velocity|gps.*speed',
    'indicated_speed': r'indicated.*speed|vehicle.*speed',
    'wheel_fl': r'wheel.*fl|wheel.*front.*left',
    'wheel_fr': r'wheel.*fr|wheel.*front.*right',
    'wheel_rl': r'wheel.*rl|wheel.*rear.*left',
    'wheel_rr': r'wheel.*rr|wheel.*rear.*right',
    'steering_angle': r'steering.*angle',
    'gear': r'gear',
    'handbrake': r'handbrake|parking.*brake',
}


@dataclass
class TrajectoryData:
    """Unified trajectory data from IO-VNBD or any compatible dataset."""
    name: str
    source: str  # 'io_vnbd', 'synthetic', 'custom'

    # Time
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))  # seconds

    # IMU (smartphone frame)
    accel_x: np.ndarray = field(default_factory=lambda: np.array([]))  # m/s^2
    accel_y: np.ndarray = field(default_factory=lambda: np.array([]))
    accel_z: np.ndarray = field(default_factory=lambda: np.array([]))
    gyro_x: np.ndarray = field(default_factory=lambda: np.array([]))   # rad/s
    gyro_y: np.ndarray = field(default_factory=lambda: np.array([]))
    gyro_z: np.ndarray = field(default_factory=lambda: np.array([]))
    mag_x: np.ndarray = field(default_factory=lambda: np.array([]))    # μT
    mag_y: np.ndarray = field(default_factory=lambda: np.array([]))
    mag_z: np.ndarray = field(default_factory=lambda: np.array([]))

    # Orientation (from device)
    orient_yaw: np.ndarray = field(default_factory=lambda: np.array([]))    # degrees
    orient_pitch: np.ndarray = field(default_factory=lambda: np.array([]))
    orient_roll: np.ndarray = field(default_factory=lambda: np.array([]))

    # GNSS (ground truth when available)
    gps_lat: np.ndarray = field(default_factory=lambda: np.array([]))
    gps_lon: np.ndarray = field(default_factory=lambda: np.array([]))
    gps_alt: np.ndarray = field(default_factory=lambda: np.array([]))
    gps_velocity: np.ndarray = field(default_factory=lambda: np.array([]))  # km/h
    gps_heading: np.ndarray = field(default_factory=lambda: np.array([]))   # degrees
    gps_accuracy: np.ndarray = field(default_factory=lambda: np.array([]))  # meters
    num_satellites: np.ndarray = field(default_factory=lambda: np.array([]))

    # Vehicle data (from ECU, optional)
    indicated_speed: np.ndarray = field(default_factory=lambda: np.array([]))  # km/h
    yaw_rate: np.ndarray = field(default_factory=lambda: np.array([]))         # deg/s
    wheel_speed_avg: np.ndarray = field(default_factory=lambda: np.array([]))  # rad/s

    # Derived
    sample_rate: float = 10.0  # Hz
    duration: float = 0.0      # seconds
    total_distance: float = 0.0  # meters (GPS-derived)

    @property
    def num_samples(self) -> int:
        return len(self.timestamps)

    def compute_gps_distance(self) -> float:
        """Compute total GPS trajectory distance in meters."""
        if len(self.gps_lat) < 2:
            return 0.0
        R = 6371000.0
        lat_rad = np.radians(self.gps_lat)
        lon_rad = np.radians(self.gps_lon)
        dlat = np.diff(lat_rad)
        dlon = np.diff(lon_rad)
        a = np.sin(dlat/2)**2 + np.cos(lat_rad[:-1]) * np.cos(lat_rad[1:]) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        dists = R * c
        # Filter out GPS jumps (>100m between consecutive points at 10Hz is suspicious)
        dists = dists[dists < 100.0]
        self.total_distance = float(np.sum(dists))
        return self.total_distance


def _match_columns(df_columns: list, patterns: dict) -> dict:
    """Match actual CSV column names to our canonical names using regex patterns."""
    mapping = {}
    for canonical, pattern in patterns.items():
        for col in df_columns:
            if re.search(pattern, col, re.IGNORECASE):
                mapping[canonical] = col
                break
    return mapping


def load_io_vnbd_smartphone(csv_path: str) -> Optional[TrajectoryData]:
    """
    Load a single IO-VNBD smartphone CSV file.

    Args:
        csv_path: Path to the S-*.csv file.

    Returns:
        TrajectoryData with all available sensor streams, or None if load fails.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"[WARN] File not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"[WARN] Failed to load {csv_path}: {e}")
        return None

    if len(df) < 10:
        print(f"[WARN] Too few rows in {csv_path}: {len(df)}")
        return None

    col_map = _match_columns(list(df.columns), S_COLUMN_PATTERNS)

    traj = TrajectoryData(
        name=path.stem,
        source='io_vnbd'
    )

    # Timestamp
    if 'timestamp_ms' in col_map:
        ts = pd.to_numeric(df[col_map['timestamp_ms']], errors='coerce').values
        traj.timestamps = ts / 1000.0  # ms → seconds
    else:
        # Fallback: generate timestamps assuming 10 Hz
        traj.timestamps = np.arange(len(df)) * 0.1

    # Helper to safely extract numeric column
    def _get(key: str) -> np.ndarray:
        if key in col_map and col_map[key] in df.columns:
            return pd.to_numeric(df[col_map[key]], errors='coerce').fillna(0).values
        return np.zeros(len(df))

    # IMU
    traj.accel_x = _get('accel_x')
    traj.accel_y = _get('accel_y')
    traj.accel_z = _get('accel_z')
    traj.gyro_x = _get('gyro_roll')   # roll ↔ x-axis
    traj.gyro_y = _get('gyro_pitch')  # pitch ↔ y-axis
    traj.gyro_z = _get('gyro_yaw')    # yaw ↔ z-axis
    traj.mag_x = _get('mag_x')
    traj.mag_y = _get('mag_y')
    traj.mag_z = _get('mag_z')

    # Orientation
    traj.orient_yaw = _get('orient_yaw')
    traj.orient_pitch = _get('orient_pitch')
    traj.orient_roll = _get('orient_roll')

    # GPS
    traj.gps_lat = _get('gps_lat')
    traj.gps_lon = _get('gps_lon')
    traj.gps_alt = _get('gps_alt')
    traj.gps_velocity = _get('gps_velocity')
    traj.gps_heading = _get('gps_heading')
    traj.gps_accuracy = _get('gps_accuracy')
    traj.num_satellites = _get('num_satellites')

    # Derived
    if len(traj.timestamps) > 1:
        traj.duration = traj.timestamps[-1] - traj.timestamps[0]
        dt = np.median(np.diff(traj.timestamps))
        traj.sample_rate = 1.0 / dt if dt > 0 else 10.0

    traj.compute_gps_distance()

    return traj


def load_io_vnbd_vehicle(csv_path: str) -> Optional[TrajectoryData]:
    """Load a single IO-VNBD vehicle ECU CSV file."""
    path = Path(csv_path)
    if not path.exists():
        return None

    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None

    if len(df) < 10:
        return None

    col_map = _match_columns(list(df.columns), V_COLUMN_PATTERNS)

    traj = TrajectoryData(name=path.stem, source='io_vnbd')

    if 'timestamp_s' in col_map:
        traj.timestamps = pd.to_numeric(df[col_map['timestamp_s']], errors='coerce').values
    else:
        traj.timestamps = np.arange(len(df)) * 0.1

    def _get(key: str) -> np.ndarray:
        if key in col_map and col_map[key] in df.columns:
            return pd.to_numeric(df[col_map[key]], errors='coerce').fillna(0).values
        return np.zeros(len(df))

    traj.gps_lat = _get('gps_lat')
    traj.gps_lon = _get('gps_lon')
    traj.gps_velocity = _get('gps_velocity')
    traj.gps_heading = _get('gps_heading')
    traj.indicated_speed = _get('indicated_speed')
    traj.yaw_rate = _get('yaw_rate')

    # Compute average wheel speed if available
    wfl = _get('wheel_fl')
    wfr = _get('wheel_fr')
    wrl = _get('wheel_rl')
    wrr = _get('wheel_rr')
    wheel_data = np.stack([wfl, wfr, wrl, wrr])
    nonzero_mask = np.any(wheel_data != 0, axis=0)
    if np.any(nonzero_mask):
        traj.wheel_speed_avg = np.mean(wheel_data, axis=0)
    
    if len(traj.timestamps) > 1:
        traj.duration = traj.timestamps[-1] - traj.timestamps[0]
        dt = np.median(np.diff(traj.timestamps))
        traj.sample_rate = 1.0 / dt if dt > 0 else 10.0

    traj.compute_gps_distance()

    return traj


def discover_io_vnbd_trajectories(dataset_root: str) -> list[dict]:
    """
    Discover all available trajectory pairs (smartphone + vehicle) in IO-VNBD.

    Returns list of dicts with keys: name, smartphone_path, vehicle_path
    """
    root = Path(dataset_root)
    trajectories = []

    # Try standard IO-VNBD structure
    s_dir = root / "Synchronised V abd S datasets" / "Uncategorised IOVNB Dataset" / "S-Dataset"
    v_dir = root / "Synchronised V abd S datasets" / "Uncategorised IOVNB Dataset" / "V-Dataset"

    if not s_dir.exists():
        # Try flat structure
        s_dir = root / "S-Dataset"
        v_dir = root / "V-Dataset"

    if not s_dir.exists():
        # Try looking for any CSV files
        csv_files = list(root.rglob("S-*.csv"))
        if csv_files:
            s_dir = csv_files[0].parent
            v_dir = s_dir.parent / "V-Dataset"

    if s_dir.exists():
        for s_file in sorted(s_dir.glob("S-*.csv")):
            name = s_file.stem.replace("S-", "")
            v_file = v_dir / f"V-{name}.csv" if v_dir.exists() else None
            trajectories.append({
                'name': name,
                'smartphone_path': str(s_file),
                'vehicle_path': str(v_file) if v_file and v_file.exists() else None
            })

    return trajectories
