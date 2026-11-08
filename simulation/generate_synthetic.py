"""Generate 60 seconds of synchronized synthetic wearable sensor data.

Produces a single CSV where GPS, IMU and PPG samples share one 100 Hz time
axis, so the downstream filtering / FFT / metrics code can be exercised
without a Raspberry Pi or a real match.

The session is built around a three-phase activity profile:
    0-20 s   jog / walk      (~1.5 m/s)
    20-35 s  sprint          (~6.5 m/s, above the 5.5 m/s sprint threshold)
    35-60 s  cooldown walk   (~1.0 m/s)
"""

import os

import numpy as np
import pandas as pd

# --- Acquisition constants -------------------------------------------------
FS = 100.0            # master sampling rate (Hz) -- GPS is upsampled to match
DURATION = 60.0       # session length (s)
N_SAMPLES = int(FS * DURATION)   # 6000 samples

HR_HZ = 2.0           # 120 BPM heart rate -> 2.0 Hz fundamental
CADENCE_HZ = 3.0      # 3 steps/sec running cadence -> motion artifact

SPRINT_START = 20.0   # start of the high-intensity window (s)
SPRINT_END = 35.0     # end of the high-intensity window (s)

BASE_LAT = 37.7749    # starting position (San Francisco)
BASE_LON = -122.4194

M_PER_DEG_LAT = 111_320.0   # metres per degree of latitude (near enough)

rng = np.random.default_rng(42)   # fixed seed -> reproducible test fixture


def build_time_axis():
    """Uniform 100 Hz time axis.

    Built with arange/FS rather than linspace(0, 60) so the sample interval is
    exactly 0.01 s. An exact rate matters because the FFT stage resolves the
    2 Hz heart rate and 3 Hz cadence peaks from this spacing.
    """
    return np.arange(N_SAMPLES) / FS


def build_ppg(t):
    """PPG optical signal: heart rate buried under running motion artifact.

    Three additive components, mirroring what the MAX30102 actually sees on a
    moving player:
      * cardiac pulse   -- 2.0 Hz sine, amplitude 0.5
      * motion artifact -- 3.0 Hz sine, amplitude 1.2 (arm swing / foot strike
        dominates the true pulse, which is the whole point of the filter work)
      * sensor noise    -- Gaussian, sigma 0.3
    """
    cardiac = 0.5 * np.sin(2 * np.pi * HR_HZ * t)
    motion = 1.2 * np.sin(2 * np.pi * CADENCE_HZ * t)
    noise = rng.normal(0.0, 0.3, size=t.size)
    return cardiac + motion + noise


def build_accel_z(t):
    """Vertical accelerometer trace in g.

    Baseline is the 1.0 g static gravity offset plus small movement jitter.
    Inside the sprint window a 3.0 Hz impact wave (one peak per foot strike)
    is superimposed with a 1.5 g amplitude boost.
    """
    accel = 1.0 + rng.normal(0.0, 0.05, size=t.size)

    sprint = (t >= SPRINT_START) & (t < SPRINT_END)
    impacts = 1.5 * np.sin(2 * np.pi * CADENCE_HZ * t[sprint])
    accel[sprint] += impacts

    return accel


def build_speed(t):
    """Piecewise ground speed (m/s) with realistic second-to-second variation."""
    speed = np.empty(t.size)

    jog = t < SPRINT_START
    sprint = (t >= SPRINT_START) & (t < SPRINT_END)
    cooldown = t >= SPRINT_END

    speed[jog] = 1.5
    speed[sprint] = 6.5
    speed[cooldown] = 1.0

    # GPS speed is never perfectly steady -- add small Gaussian wobble and
    # clamp at zero so no sample reports negative motion.
    speed += rng.normal(0.0, 0.15, size=t.size)
    return np.clip(speed, 0.0, None)


def build_track(speed):
    """Integrate speed into smoothly updating latitude / longitude.

    Each sample advances the player by speed * dt metres along a slowly
    wandering heading, so the track curves like a real run instead of a
    straight line. Metre offsets are converted back to degrees and
    cumulatively summed onto the base position.
    """
    dt = 1.0 / FS

    # Heading drifts a few tens of degrees over the session (a wide arc).
    heading = np.deg2rad(45.0 + 30.0 * np.sin(2 * np.pi * np.arange(speed.size) / speed.size))

    step_m = speed * dt
    north_m = np.cumsum(step_m * np.cos(heading))
    east_m = np.cumsum(step_m * np.sin(heading))

    m_per_deg_lon = M_PER_DEG_LAT * np.cos(np.deg2rad(BASE_LAT))

    latitude = BASE_LAT + north_m / M_PER_DEG_LAT
    longitude = BASE_LON + east_m / m_per_deg_lon
    return latitude, longitude


def main():
    t = build_time_axis()
    speed = build_speed(t)
    latitude, longitude = build_track(speed)

    df = pd.DataFrame({
        "timestamp": t,
        "ppg_raw": build_ppg(t),
        "accel_z": build_accel_z(t),
        "latitude": latitude,
        "longitude": longitude,
        "speed_ms": speed,
    })

    # Write next to this script so the output lands in simulation/ no matter
    # which directory the script is launched from.
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "simulated_match.csv")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} samples ({DURATION:.1f} s @ {FS:.0f} Hz) to {out_path}")


if __name__ == "__main__":
    main()
