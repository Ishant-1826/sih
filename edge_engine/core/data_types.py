"""
Core data types for the IDR-GNSS-FUSION system.

These dataclasses define the canonical representations used throughout
the pipeline — from raw sensor ingestion through to navigation output.
All modules communicate using these types.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import numpy as np


class NavigationMode(Enum):
    """Current navigation operating mode."""
    GNSS_AIDED = auto()
    GNSS_DEGRADED = auto()
    DEAD_RECKONING = auto()
    MAP_MATCHED = auto()
    RECOVERING = auto()


class GNSSStatus(Enum):
    """GNSS signal quality classification."""
    AVAILABLE = auto()
    DEGRADED = auto()
    UNRELIABLE = auto()
    UNAVAILABLE = auto()
    RECOVERED = auto()


class MotionClass(Enum):
    """Vehicle/phone motion classification."""
    STATIONARY = auto()
    NORMAL_DRIVING = auto()
    ACCELERATION = auto()
    BRAKING = auto()
    TURNING = auto()
    POTHOLE = auto()
    SPEED_BREAKER = auto()
    ENGINE_VIBRATION = auto()
    PHONE_MOVEMENT = auto()
    PHONE_REMOVAL = auto()
    PHONE_MISALIGNMENT = auto()
    SHARP_SHOCK = auto()


@dataclass
class IMUSample:
    """Single IMU measurement at a point in time."""
    timestamp: float  # seconds since epoch
    ax: float = 0.0   # accelerometer x (m/s^2)
    ay: float = 0.0   # accelerometer y
    az: float = 0.0   # accelerometer z
    gx: float = 0.0   # gyroscope x (rad/s)
    gy: float = 0.0   # gyroscope y
    gz: float = 0.0   # gyroscope z
    mx: float = 0.0   # magnetometer x (μT)
    my: float = 0.0   # magnetometer y
    mz: float = 0.0   # magnetometer z


@dataclass
class GNSSMeasurement:
    """Single GNSS/GPS fix."""
    timestamp: float
    latitude: float = 0.0
    longitude: float = 0.0
    altitude: float = 0.0
    velocity: float = 0.0          # m/s
    heading: float = 0.0           # degrees
    vertical_velocity: float = 0.0 # m/s
    accuracy: float = float('inf') # meters
    num_satellites: int = 0
    hdop: float = float('inf')
    fix_valid: bool = False


@dataclass
class Orientation:
    """Orientation represented as quaternion (internally) with Euler convenience."""
    qw: float = 1.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0

    @property
    def roll(self) -> float:
        """Roll angle in radians."""
        sinr = 2.0 * (self.qw * self.qx + self.qy * self.qz)
        cosr = 1.0 - 2.0 * (self.qx**2 + self.qy**2)
        return np.arctan2(sinr, cosr)

    @property
    def pitch(self) -> float:
        """Pitch angle in radians."""
        sinp = 2.0 * (self.qw * self.qy - self.qz * self.qx)
        sinp = np.clip(sinp, -1.0, 1.0)
        return np.arcsin(sinp)

    @property
    def yaw(self) -> float:
        """Yaw angle in radians."""
        siny = 2.0 * (self.qw * self.qz + self.qx * self.qy)
        cosy = 1.0 - 2.0 * (self.qy**2 + self.qz**2)
        return np.arctan2(siny, cosy)

    def to_rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        w, x, y, z = self.qw, self.qx, self.qy, self.qz
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
        ])

    @staticmethod
    def from_euler(roll: float, pitch: float, yaw: float) -> 'Orientation':
        """Create Orientation from Euler angles (radians)."""
        cr, sr = np.cos(roll/2), np.sin(roll/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        return Orientation(
            qw=cr*cp*cy + sr*sp*sy,
            qx=sr*cp*cy - cr*sp*sy,
            qy=cr*sp*cy + sr*cp*sy,
            qz=cr*cp*sy - sr*sp*cy
        )

    def normalize(self) -> 'Orientation':
        """Return normalized quaternion."""
        n = np.sqrt(self.qw**2 + self.qx**2 + self.qy**2 + self.qz**2)
        if n < 1e-10:
            return Orientation()
        return Orientation(self.qw/n, self.qx/n, self.qy/n, self.qz/n)


@dataclass
class NavigationState:
    """Complete navigation state estimate at a single epoch."""
    timestamp: float = 0.0

    # Position (meters in local tangent plane, or lat/lon)
    latitude: float = 0.0
    longitude: float = 0.0
    position_x: float = 0.0   # meters, local frame
    position_y: float = 0.0

    # Velocity
    velocity_x: float = 0.0   # m/s
    velocity_y: float = 0.0
    speed: float = 0.0        # m/s, scalar

    # Heading & Orientation
    heading: float = 0.0      # degrees, 0=North, clockwise
    orientation: Orientation = field(default_factory=Orientation)

    # Biases (estimated)
    accel_bias_x: float = 0.0
    accel_bias_y: float = 0.0
    accel_bias_z: float = 0.0
    gyro_bias_x: float = 0.0
    gyro_bias_y: float = 0.0
    gyro_bias_z: float = 0.0

    # Status
    mode: NavigationMode = NavigationMode.GNSS_AIDED
    gnss_status: GNSSStatus = GNSSStatus.UNAVAILABLE
    confidence: float = 0.0         # 0.0 to 1.0
    position_uncertainty: float = float('inf')  # meters

    # Distance tracking
    distance_travelled: float = 0.0  # meters since start


@dataclass
class TrajectoryPoint:
    """A single point in a trajectory (for ground truth / comparison)."""
    timestamp: float
    latitude: float
    longitude: float
    speed: float = 0.0      # m/s
    heading: float = 0.0    # degrees
    position_x: float = 0.0 # meters (local)
    position_y: float = 0.0


# ---- Coordinate Utilities ----

def latlon_to_meters(lat: float, lon: float,
                     ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """
    Convert lat/lon to local tangent plane meters relative to reference point.
    Uses equirectangular approximation (accurate for small regions ~50km).
    """
    R = 6371000.0  # Earth radius in meters
    dlat = np.radians(lat - ref_lat)
    dlon = np.radians(lon - ref_lon)
    x = dlon * R * np.cos(np.radians(ref_lat))
    y = dlat * R
    return x, y


def meters_to_latlon(x: float, y: float,
                     ref_lat: float, ref_lon: float) -> tuple[float, float]:
    """Convert local tangent plane meters back to lat/lon."""
    R = 6371000.0
    lat = ref_lat + np.degrees(y / R)
    lon = ref_lon + np.degrees(x / (R * np.cos(np.radians(ref_lat))))
    return lat, lon


def haversine_distance(lat1: float, lon1: float,
                       lat2: float, lon2: float) -> float:
    """Distance in meters between two lat/lon points (Haversine formula)."""
    R = 6371000.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * \
        np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
