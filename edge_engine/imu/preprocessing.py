"""
IMU Preprocessing Pipeline for IDR-GNSS-FUSION.

Performs:
- Static bias estimation (from stationary periods)
- Noise filtering (Butterworth low-pass, median)
- Outlier rejection
- Gravity removal from accelerometer
- Sensor validation
"""

import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import median_filter
from dataclasses import dataclass


@dataclass
class IMUCalibration:
    """IMU calibration parameters estimated from data."""
    accel_bias: np.ndarray  # shape (3,) m/s^2
    gyro_bias: np.ndarray   # shape (3,) rad/s
    accel_scale: np.ndarray  # shape (3,) scale factors
    gravity_magnitude: float  # estimated local gravity
    is_calibrated: bool = False


def estimate_static_bias(
    accel: np.ndarray,  # shape (N, 3)
    gyro: np.ndarray,   # shape (N, 3)
    window: int = 100,
    gravity: float = 9.81,
) -> IMUCalibration:
    """
    Estimate IMU biases from initial stationary period.

    Assumes the first `window` samples are approximately stationary.
    Accelerometer bias = mean - gravity direction.
    Gyroscope bias = mean (should be near zero when stationary).
    """
    n = min(window, len(accel))

    # Gyro bias: mean of stationary period (should be ~0)
    gyro_bias = np.mean(gyro[:n], axis=0)

    # Accel: mean should equal gravity vector
    accel_mean = np.mean(accel[:n], axis=0)
    gravity_magnitude = np.linalg.norm(accel_mean)

    # Gravity direction in device frame
    gravity_dir = accel_mean / gravity_magnitude if gravity_magnitude > 0 else np.array([0, 0, -1])

    # Bias = measured - true (gravity only when stationary)
    accel_bias = accel_mean - gravity_dir * gravity

    return IMUCalibration(
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
        accel_scale=np.ones(3),
        gravity_magnitude=gravity_magnitude,
        is_calibrated=True,
    )


def apply_bias_correction(
    accel: np.ndarray,
    gyro: np.ndarray,
    calibration: IMUCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove estimated biases from IMU data."""
    accel_corrected = accel - calibration.accel_bias
    gyro_corrected = gyro - calibration.gyro_bias
    return accel_corrected, gyro_corrected


def butterworth_lowpass(
    data: np.ndarray,
    cutoff_hz: float = 5.0,
    sample_rate: float = 10.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth low-pass filter to remove high-frequency noise.

    Args:
        data: Input signal, shape (N,) or (N, channels)
        cutoff_hz: Cutoff frequency
        sample_rate: Sampling rate in Hz
        order: Filter order

    Returns:
        Filtered signal, same shape as input.
    """
    nyq = 0.5 * sample_rate
    if cutoff_hz >= nyq:
        return data  # Can't filter above Nyquist

    b, a = butter(order, cutoff_hz / nyq, btype='low')

    if data.ndim == 1:
        if len(data) < 3 * max(len(a), len(b)):
            return data
        return filtfilt(b, a, data)
    else:
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            if len(data[:, i]) < 3 * max(len(a), len(b)):
                result[:, i] = data[:, i]
            else:
                result[:, i] = filtfilt(b, a, data[:, i])
        return result


def median_filter_signal(
    data: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Apply median filter for impulse noise removal (potholes, spikes)."""
    if data.ndim == 1:
        return medfilt(data, kernel_size=kernel_size)
    else:
        result = np.zeros_like(data)
        for i in range(data.shape[1]):
            result[:, i] = medfilt(data[:, i], kernel_size=kernel_size)
        return result


def reject_outliers(
    data: np.ndarray,
    threshold_std: float = 5.0,
) -> np.ndarray:
    """
    Replace outliers (> threshold_std from mean) with interpolated values.
    Uses per-column statistics for multi-dimensional data.
    """
    result = data.copy()

    if data.ndim == 1:
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return result
        outliers = np.abs(data - mean) > threshold_std * std
        if np.any(outliers):
            valid = ~outliers
            if np.any(valid):
                result[outliers] = np.interp(
                    np.where(outliers)[0],
                    np.where(valid)[0],
                    data[valid]
                )
    else:
        for i in range(data.shape[1]):
            result[:, i] = reject_outliers(data[:, i], threshold_std)

    return result


def remove_gravity(
    accel: np.ndarray,  # shape (N, 3)
    gravity_magnitude: float = 9.81,
    method: str = "highpass",
    sample_rate: float = 10.0,
) -> np.ndarray:
    """
    Remove gravity component from accelerometer readings.

    Methods:
        'highpass': High-pass filter (simple, works for phone-on-dashboard)
        'orientation': Use known orientation to subtract gravity vector
    """
    if method == "highpass":
        # Gravity is a DC component; high-pass filter removes it
        nyq = 0.5 * sample_rate
        cutoff = 0.5  # Hz
        if cutoff >= nyq:
            # Can't filter, just subtract mean
            return accel - np.mean(accel, axis=0)
        b, a = butter(2, cutoff / nyq, btype='high')
        result = np.zeros_like(accel)
        for i in range(3):
            if len(accel[:, i]) >= 12:
                result[:, i] = filtfilt(b, a, accel[:, i])
            else:
                result[:, i] = accel[:, i] - np.mean(accel[:, i])
        return result
    else:
        # Simple: subtract column means (approximation)
        return accel - np.mean(accel[:min(100, len(accel))], axis=0)


def detect_stationary(
    accel: np.ndarray,
    gyro: np.ndarray,
    accel_threshold: float = 0.3,  # m/s^2 deviation from gravity
    gyro_threshold: float = 0.05,  # rad/s
    window: int = 10,
) -> np.ndarray:
    """
    Detect stationary periods using Zero Velocity Update (ZUPT) criteria.

    Returns boolean mask: True where vehicle is likely stationary.
    """
    n = len(accel)
    stationary = np.zeros(n, dtype=bool)

    # Compute running variance of accel magnitude
    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    for i in range(window, n):
        accel_var = np.var(accel_mag[i-window:i])
        gyro_mean = np.mean(gyro_mag[i-window:i])
        if accel_var < accel_threshold**2 and gyro_mean < gyro_threshold:
            stationary[i-window:i] = True

    return stationary


def preprocess_imu(
    accel: np.ndarray,  # (N, 3)
    gyro: np.ndarray,   # (N, 3)
    sample_rate: float = 10.0,
    filter_cutoff: float = 5.0,
    outlier_threshold: float = 5.0,
    estimate_bias: bool = True,
    remove_gravity_component: bool = True,
) -> dict:
    """
    Full IMU preprocessing pipeline.

    Returns dict with:
        - accel: preprocessed accelerometer (N, 3)
        - gyro: preprocessed gyroscope (N, 3)
        - calibration: IMUCalibration object
        - stationary: boolean mask
    """
    # Step 1: Outlier rejection
    accel = reject_outliers(accel, outlier_threshold)
    gyro = reject_outliers(gyro, outlier_threshold)

    # Step 2: Bias estimation
    calibration = IMUCalibration(
        accel_bias=np.zeros(3),
        gyro_bias=np.zeros(3),
        accel_scale=np.ones(3),
        gravity_magnitude=9.81,
    )
    if estimate_bias:
        calibration = estimate_static_bias(accel, gyro)
        accel, gyro = apply_bias_correction(accel, gyro, calibration)

    # Step 3: Low-pass filter
    accel = butterworth_lowpass(accel, filter_cutoff, sample_rate)
    gyro = butterworth_lowpass(gyro, filter_cutoff, sample_rate)

    # Step 4: Gravity removal
    if remove_gravity_component:
        accel_linear = remove_gravity(accel, calibration.gravity_magnitude,
                                       sample_rate=sample_rate)
    else:
        accel_linear = accel

    # Step 5: Stationary detection
    stationary = detect_stationary(accel, gyro)

    return {
        'accel': accel_linear,
        'accel_with_gravity': accel,
        'gyro': gyro,
        'calibration': calibration,
        'stationary': stationary,
    }
