"""
Phase 1 Experiment: Baseline INS Dead Reckoning with GNSS Outage Simulation.

This is the FIRST measurable milestone of the IDR-GNSS-FUSION project.

Pipeline:
    Synthetic Data → GNSS Blackout Simulation → INS Baseline →
    Ground Truth vs INS Trajectory → Drift Calculation → Plots

Outputs:
    1. Ground truth vs INS trajectory plot
    2. GNSS outage timeline
    3. Position error vs time
    4. Velocity comparison
    5. Heading error
    6. Metrics report

Usage:
    py experiments/run_baseline.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec

from ml.datasets.generate_sample import generate_trajectory
from edge_engine.gnss.outage_simulator import (
    apply_outage, create_progressive_scenario, create_tunnel_scenario,
    get_outage_timeline
)
from edge_engine.imu.preprocessing import preprocess_imu
from edge_engine.dead_reckoning.ins_engine import (
    DeadReckoningEngine, DRConfig, run_dead_reckoning
)
from ml.evaluation.metrics import evaluate_navigation, compute_position_errors
from edge_engine.core.data_types import haversine_distance


def run_experiment(scenario_name: str = "progressive"):
    """Run the complete Phase 1 baseline experiment."""

    print("=" * 70)
    print("IDR-GNSS-FUSION — Phase 1: Baseline Dead Reckoning Experiment")
    print("=" * 70)

    # ---- Step 1: Generate synthetic trajectory ----
    print("\n[1/6] Generating synthetic trajectory...")
    traj = generate_trajectory(
        name="experiment_urban",
        duration=300.0,      # 5 minutes
        sample_rate=10.0,
        start_lat=28.6139,   # New Delhi
        start_lon=77.2090,
        start_heading=45.0,
        max_speed_kmh=60.0,
        scenario="urban",
        accel_noise_std=0.3,
        gyro_noise_std=0.01,
        gps_noise_std=3.0,
        seed=42,
    )
    print(f"  Duration: {traj.duration:.0f}s | Samples: {traj.num_samples} | "
          f"Rate: {traj.sample_rate:.0f} Hz | Distance: {traj.total_distance:.0f}m")

    # Store ground truth
    gt_lat = traj._true_lat.copy()
    gt_lon = traj._true_lon.copy()
    gt_speed = traj._true_speed.copy()
    gt_heading = traj._true_heading.copy()

    # ---- Step 2: Apply GNSS outage ----
    print("\n[2/6] Applying GNSS outage scenario...")
    if scenario_name == "progressive":
        scenario = create_progressive_scenario(traj.duration)
    else:
        scenario = create_tunnel_scenario(traj.duration, tunnel_duration=60.0)

    traj_outage, gnss_available = apply_outage(traj, scenario, seed=42)

    total_outage = np.sum(~gnss_available) / traj.sample_rate
    print(f"  Scenario: {scenario.name}")
    print(f"  Outage regions: {len(scenario.regions)}")
    print(f"  Total outage time: {total_outage:.1f}s "
          f"({total_outage/traj.duration*100:.1f}% of trajectory)")
    for region in scenario.regions:
        print(f"    [{region.start_time:.0f}s - {region.end_time:.0f}s] "
              f"{region.outage_type}: {region.description}")

    # ---- Step 3: IMU Preprocessing ----
    print("\n[3/6] Preprocessing IMU data...")
    accel = np.stack([traj_outage.accel_x, traj_outage.accel_y, traj_outage.accel_z], axis=1)
    gyro = np.stack([traj_outage.gyro_x, traj_outage.gyro_y, traj_outage.gyro_z], axis=1)

    imu_result = preprocess_imu(
        accel, gyro,
        sample_rate=traj.sample_rate,
        filter_cutoff=5.0,
        outlier_threshold=5.0,
    )
    print(f"  Calibration: bias_accel={imu_result['calibration'].accel_bias}")
    print(f"               bias_gyro={imu_result['calibration'].gyro_bias}")
    stationary_pct = np.mean(imu_result['stationary']) * 100
    print(f"  Stationary: {stationary_pct:.1f}% of samples")

    # ---- Step 4: Run Dead Reckoning (INS Baseline) ----
    print("\n[4/6] Running dead reckoning...")

    # Initialize from first valid GPS point
    start_idx = 0
    for i in range(len(gnss_available)):
        if gnss_available[i]:
            start_idx = i
            break

    config = DRConfig(enable_nhc=True, enable_zupt=True)
    accel_proc = imu_result['accel']
    gyro_proc = imu_result['gyro']

    # Track full estimated trajectory
    dr_lat_full = gt_lat.copy()  # Start with ground truth
    dr_lon_full = gt_lon.copy()
    dr_speed_full = gt_speed.copy()
    dr_heading_full = gt_heading.copy()

    # For each outage region, run dead reckoning
    in_dr = False
    engine = DeadReckoningEngine(config)

    for i in range(start_idx + 1, traj.num_samples):
        dt = traj.timestamps[i] - traj.timestamps[i-1]
        if dt <= 0 or dt > 1.0:
            dt = 1.0 / traj.sample_rate

        if gnss_available[i]:
            # GNSS available — use ground truth (simulating perfect fusion)
            dr_lat_full[i] = gt_lat[i]
            dr_lon_full[i] = gt_lon[i]
            dr_speed_full[i] = gt_speed[i]
            dr_heading_full[i] = gt_heading[i]
            in_dr = False
        else:
            if not in_dr:
                # Entering outage — initialize DR from last known position
                engine = DeadReckoningEngine(config)
                engine.initialize(
                    latitude=dr_lat_full[i-1],
                    longitude=dr_lon_full[i-1],
                    heading=dr_heading_full[i-1],
                    speed=dr_speed_full[i-1],
                    timestamp=traj.timestamps[i-1],
                )
                in_dr = True

            state = engine.propagate(
                accel_forward=accel_proc[i, 0],
                accel_lateral=accel_proc[i, 1],
                yaw_rate=gyro_proc[i, 2],
                dt=dt,
                timestamp=traj.timestamps[i],
                is_stationary=imu_result['stationary'][i],
            )
            dr_lat_full[i] = state.latitude
            dr_lon_full[i] = state.longitude
            dr_speed_full[i] = state.speed
            dr_heading_full[i] = state.heading

    # Package results
    dr_result = {
        'lat': dr_lat_full,
        'lon': dr_lon_full,
        'speed': dr_speed_full,
        'heading': dr_heading_full,
    }
    print(f"  DR samples: {len(dr_result['lat'])}")
    print(f"  Final DR position: ({dr_result['lat'][-1]:.6f}, {dr_result['lon'][-1]:.6f})")

    # ---- Step 5: Evaluate ----
    print("\n[5/6] Computing metrics...")

    dr_lat = dr_result['lat']
    dr_lon = dr_result['lon']
    dr_speed_padded = dr_result['speed']
    dr_heading_padded = dr_result['heading']

    metrics = evaluate_navigation(
        true_lat=gt_lat,
        true_lon=gt_lon,
        est_lat=dr_lat,
        est_lon=dr_lon,
        true_speed=gt_speed,
        est_speed=dr_speed_padded,
        true_heading=gt_heading,
        est_heading=dr_heading_padded,
        gnss_available=gnss_available,
        timestamps=traj.timestamps,
    )
    print(metrics.summary())

    # ---- Step 6: Generate Plots ----
    print("\n[6/6] Generating plots...")
    output_dir = os.path.join(PROJECT_ROOT, "experiments", "results", "baseline")
    os.makedirs(output_dir, exist_ok=True)

    _generate_plots(
        timestamps=traj.timestamps,
        gt_lat=gt_lat,
        gt_lon=gt_lon,
        gt_speed=gt_speed,
        gt_heading=gt_heading,
        dr_lat=dr_lat,
        dr_lon=dr_lon,
        dr_speed=dr_speed_padded,
        dr_heading=dr_heading_padded,
        gnss_available=gnss_available,
        scenario=scenario,
        metrics=metrics,
        output_dir=output_dir,
    )

    print(f"\n✓ All plots saved to: {output_dir}")
    print(f"\n{'='*70}")
    print(f"Phase 1 Baseline Complete")
    print(f"  Drift: {metrics.drift_percentage:.2f}%")
    print(f"  RMSE:  {metrics.rmse_position:.2f}m")
    target = "PASS ✓" if metrics.drift_percentage < 10 else "FAIL ✗ (target: <10%)"
    print(f"  Target: {target}")
    print(f"{'='*70}")

    return metrics


def _generate_plots(
    timestamps, gt_lat, gt_lon, gt_speed, gt_heading,
    dr_lat, dr_lon, dr_speed, dr_heading,
    gnss_available, scenario, metrics, output_dir
):
    """Generate all Phase 1 visualization plots."""

    # Color scheme
    COLOR_GT = '#00E676'       # Green - ground truth
    COLOR_DR = '#FF5252'       # Red - dead reckoning
    COLOR_GNSS = '#448AFF'     # Blue - GNSS available
    COLOR_OUTAGE = '#FF6E40'   # Orange - outage region
    COLOR_BG = '#1A1A2E'       # Dark background
    COLOR_PANEL = '#16213E'    # Panel background
    COLOR_TEXT = '#E0E0E0'     # Light text

    plt.rcParams.update({
        'figure.facecolor': COLOR_BG,
        'axes.facecolor': COLOR_PANEL,
        'axes.edgecolor': '#333',
        'text.color': COLOR_TEXT,
        'axes.labelcolor': COLOR_TEXT,
        'xtick.color': COLOR_TEXT,
        'ytick.color': COLOR_TEXT,
        'grid.color': '#333',
        'grid.alpha': 0.5,
        'font.size': 10,
        'font.family': 'sans-serif',
    })

    # ---- Plot 1: Trajectory Map ----
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.suptitle('IDR-GNSS-FUSION — Trajectory Comparison', fontsize=16, fontweight='bold',
                 color='white')

    # Plot ground truth
    ax.plot(gt_lon, gt_lat, color=COLOR_GT, linewidth=2, label='Ground Truth', alpha=0.8)

    # Plot DR estimate
    ax.plot(dr_lon, dr_lat, color=COLOR_DR, linewidth=1.5, label='Dead Reckoning (INS)',
            linestyle='--', alpha=0.8)

    # Highlight outage regions on trajectory
    for region in scenario.regions:
        start_idx = np.argmin(np.abs(timestamps - region.start_time))
        end_idx = np.argmin(np.abs(timestamps - region.end_time))
        if end_idx > start_idx:
            ax.plot(gt_lon[start_idx:end_idx], gt_lat[start_idx:end_idx],
                    color=COLOR_OUTAGE, linewidth=4, alpha=0.4)

    # Start/end markers
    ax.plot(gt_lon[0], gt_lat[0], 'o', color='#76FF03', markersize=12, label='Start',
            zorder=5)
    ax.plot(gt_lon[-1], gt_lat[-1], 's', color='#FF1744', markersize=12, label='End',
            zorder=5)

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='upper right', facecolor=COLOR_PANEL, edgecolor='#555')
    ax.grid(True)

    # Add metrics text box
    text = (f"RMSE: {metrics.rmse_position:.2f}m\n"
            f"Drift: {metrics.drift_percentage:.2f}%\n"
            f"Max Error: {metrics.max_position_error:.2f}m")
    ax.text(0.02, 0.98, text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor=COLOR_BG, alpha=0.8,
                      edgecolor=COLOR_OUTAGE))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot1_trajectory.png'), dpi=150)
    plt.close()

    # ---- Plot 2: Multi-panel Dashboard ----
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('IDR-GNSS-FUSION — Dead Reckoning Performance Dashboard',
                 fontsize=16, fontweight='bold', color='white')

    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3)

    # -- Panel A: Position Error vs Time --
    ax1 = fig.add_subplot(gs[0, :])
    pos_errors = compute_position_errors(gt_lat, gt_lon, dr_lat, dr_lon)
    ax1.plot(timestamps, pos_errors, color=COLOR_DR, linewidth=1)
    ax1.fill_between(timestamps, 0, pos_errors, alpha=0.2, color=COLOR_DR)

    # Shade outage regions
    for region in scenario.regions:
        ax1.axvspan(region.start_time, region.end_time, alpha=0.2, color=COLOR_OUTAGE)

    ax1.set_ylabel('Position Error (m)')
    ax1.set_title('Position Error Over Time', fontweight='bold')
    ax1.grid(True)
    ax1.axhline(y=metrics.rmse_position, color='#FFD740', linestyle='--',
                label=f'RMSE = {metrics.rmse_position:.1f}m', alpha=0.7)
    ax1.legend(facecolor=COLOR_PANEL)

    # -- Panel B: GNSS Availability Timeline --
    ax2 = fig.add_subplot(gs[1, :])
    gnss_color = np.where(gnss_available, 1.0, 0.0)
    ax2.fill_between(timestamps, 0, 1, where=gnss_available,
                     color=COLOR_GNSS, alpha=0.5, label='GNSS Available')
    ax2.fill_between(timestamps, 0, 1, where=~gnss_available,
                     color=COLOR_OUTAGE, alpha=0.5, label='GNSS Outage')
    ax2.set_ylabel('Status')
    ax2.set_title('GNSS Availability & Navigation Mode', fontweight='bold')
    ax2.set_yticks([0.25, 0.75])
    ax2.set_yticklabels(['Outage', 'Available'])
    ax2.legend(loc='upper right', facecolor=COLOR_PANEL)
    ax2.grid(True, axis='x')

    # -- Panel C: Speed Comparison --
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(timestamps, gt_speed * 3.6, color=COLOR_GT, linewidth=1.5,
             label='Ground Truth', alpha=0.8)
    ax3.plot(timestamps, dr_speed * 3.6, color=COLOR_DR, linewidth=1,
             label='DR Estimate', alpha=0.7, linestyle='--')
    for region in scenario.regions:
        ax3.axvspan(region.start_time, region.end_time, alpha=0.15, color=COLOR_OUTAGE)
    ax3.set_ylabel('Speed (km/h)')
    ax3.set_xlabel('Time (s)')
    ax3.set_title('Speed Comparison', fontweight='bold')
    ax3.legend(facecolor=COLOR_PANEL)
    ax3.grid(True)

    # -- Panel D: Heading Comparison --
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(timestamps, gt_heading, color=COLOR_GT, linewidth=1.5,
             label='Ground Truth', alpha=0.8)
    ax4.plot(timestamps, dr_heading, color=COLOR_DR, linewidth=1,
             label='DR Estimate', alpha=0.7, linestyle='--')
    for region in scenario.regions:
        ax4.axvspan(region.start_time, region.end_time, alpha=0.15, color=COLOR_OUTAGE)
    ax4.set_ylabel('Heading (°)')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Heading Comparison', fontweight='bold')
    ax4.legend(facecolor=COLOR_PANEL)
    ax4.grid(True)

    # -- Panel E: Drift vs Distance --
    ax5 = fig.add_subplot(gs[3, 0])
    # Compute cumulative distance
    cum_dist = np.zeros(len(gt_lat))
    for i in range(1, len(gt_lat)):
        if gt_lat[i] != 0 and gt_lon[i] != 0:
            cum_dist[i] = cum_dist[i-1] + haversine_distance(
                gt_lat[i-1], gt_lon[i-1], gt_lat[i], gt_lon[i])
        else:
            cum_dist[i] = cum_dist[i-1]

    ax5.plot(cum_dist, pos_errors, color=COLOR_DR, linewidth=1)
    ax5.fill_between(cum_dist, 0, pos_errors, alpha=0.2, color=COLOR_DR)
    ax5.set_xlabel('Distance Travelled (m)')
    ax5.set_ylabel('Position Error (m)')
    ax5.set_title('Position Error vs Distance', fontweight='bold')
    ax5.grid(True)

    # -- Panel F: Metrics Summary --
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')
    summary_text = (
        f"━━━ BASELINE METRICS ━━━\n\n"
        f"Position RMSE:     {metrics.rmse_position:>8.2f} m\n"
        f"Position MAE:      {metrics.mae_position:>8.2f} m\n"
        f"Max Error:         {metrics.max_position_error:>8.2f} m\n"
        f"95th Percentile:   {metrics.p95_position_error:>8.2f} m\n"
        f"Drift:             {metrics.drift_percentage:>8.2f} %\n\n"
        f"Heading RMSE:      {metrics.rmse_heading:>8.2f}°\n"
        f"Velocity RMSE:     {metrics.rmse_velocity:>8.2f} m/s\n\n"
        f"Total Distance:    {metrics.total_distance:>8.0f} m\n"
        f"Outage Duration:   {metrics.outage_duration:>8.1f} s\n"
        f"Samples:           {metrics.num_samples:>8d}\n\n"
        f"Target (<10%):     {'PASS ✓' if metrics.drift_percentage < 10 else 'FAIL ✗'}"
    )
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             color=COLOR_TEXT,
             bbox=dict(boxstyle='round', facecolor=COLOR_BG, alpha=0.9,
                       edgecolor=COLOR_OUTAGE, linewidth=2))

    plt.savefig(os.path.join(output_dir, 'plot2_dashboard.png'), dpi=150)
    plt.close()

    print(f"  → plot1_trajectory.png")
    print(f"  → plot2_dashboard.png")


if __name__ == "__main__":
    run_experiment("progressive")
