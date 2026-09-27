# IDR-GNSS-FUSION

## AI-Enhanced Intelligent Dead Reckoning and GNSS/INS Fusion for Smartphone-Based Vehicle Navigation

**Smart India Hackathon (SIH) Project**

A hybrid physics + AI navigation system that transforms a commodity smartphone into an Intelligent Dead Reckoning (IDR) system capable of maintaining continuous vehicle localization during GNSS/GPS outages.

---

### Architecture

```
GNSS + IMU → Sensor Fusion (EKF/UKF) → AI Correction → Dead Reckoning
                                              ↓
                        NHC + Vehicle Constraints → Map Matching
                                              ↓
                              Position / Velocity / Heading / Confidence
```

### Project Structure

```
IDR-GNSS-FUSION/
├── edge_engine/           # Hardware-independent navigation engine
│   ├── core/              # Data types, configuration
│   ├── imu/               # IMU preprocessing, bias estimation
│   ├── gnss/              # GNSS status, outage simulation
│   ├── fusion/            # EKF/UKF sensor fusion
│   ├── dead_reckoning/    # INS dead reckoning engine
│   ├── map_matching/      # Offline map matching
│   ├── calibration/       # Phone/vehicle alignment
│   └── api/               # REST/WebSocket API
├── ml/                    # Machine learning pipeline
│   ├── datasets/          # IO-VNBD loader, synthetic data
│   ├── preprocessing/     # Feature engineering
│   ├── models/            # Speed estimation, motion classifier
│   ├── training/          # Training scripts
│   ├── evaluation/        # Metrics, benchmarking
│   └── export/            # ONNX/TFLite export
├── mobile/                # Android application (Phase 8)
├── experiments/           # Experiment scripts and results
├── configs/               # YAML configuration
├── datasets/              # Dataset storage
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

### Quick Start

```bash
# Install dependencies
pip install numpy pandas scipy matplotlib pyyaml

# Generate synthetic dataset
py ml/datasets/generate_sample.py

# Run baseline experiment
py experiments/run_baseline.py
```

### Development Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | Dataset pipeline + GNSS outage simulation + INS baseline |
| 2 | 🔲 | Classical EKF/UKF sensor fusion |
| 3 | 🔲 | AI speed estimation model |
| 4 | 🔲 | AI sensor correction + motion classifier |
| 5 | 🔲 | Offline map matching |
| 6 | 🔲 | Hybrid fusion system |
| 7 | 🔲 | Edge engine API |
| 8 | 🔲 | Android application |
| 9 | 🔲 | Full evaluation + SIH demo |
| 10 | 🔲 | Documentation + presentation |

### Dataset

Uses **IO-VNBD** (Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning).
- 58+ hours, 4,400+ km of driving data
- Smartphone IMU + GPS at 10 Hz
- Vehicle ECU data (wheel speed, steering, yaw rate)
- GitHub: https://github.com/onyekpeu/IO-VNBD

### Performance Target

**< 10% position drift** during GNSS-denied operation (e.g., <5m error over 50m travel at 60 km/h).

### License

Research prototype — not for safety-critical applications.
