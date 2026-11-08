"""Post-session analytics: distance covered and sprint count.

Reads a recorded (or simulated) match log and reduces the GPS track to the
numbers a coach actually looks at — how far the player ran, how fast they got,
and how many separate sprint efforts they produced.
"""

import os

import numpy as np
import pandas as pd

# --- Constants -------------------------------------------------------------
EARTH_RADIUS_M = 6_371_000.0   # mean Earth radius, the usual Haversine value

SPRINT_THRESHOLD_MS = 5.5      # speed above which an effort counts as a sprint
MIN_SPRINT_DURATION_S = 0.5    # reject sub-half-second threshold flickers


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between two points on the Earth.

    The Haversine formula treats the Earth as a sphere and solves the
    great-circle arc between two lat/lon pairs:

        a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
        c = 2·atan2(√a, √(1-a))
        d = R·c

    `a` is the square of half the straight-line (chord) distance between the
    points; `c` converts that chord back into the angle subtended at the
    Earth's centre; multiplying by the radius turns the angle into metres.
    atan2 is used rather than asin because it stays numerically stable for the
    very small angles produced by consecutive 100 Hz GPS samples.

    Accepts scalars or numpy arrays — passing arrays applies the identical
    formula to every consecutive pair at once.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return EARTH_RADIUS_M * c


def total_distance(latitude, longitude):
    """Sum the point-to-point Haversine distance along the whole track.

    Each sample is paired with the one before it (the arrays are sliced
    [1:] against [:-1]), giving N-1 step distances that add up to the total
    path length. Summing per-step distances rather than measuring start-to-end
    is what captures turns and direction changes.
    """
    steps = haversine(latitude[:-1], longitude[:-1], latitude[1:], longitude[1:])
    return steps.sum(), steps


def detect_sprints(timestamps, speed):
    """Group contiguous above-threshold samples into distinct sprint efforts.

    A boolean mask marks every frame over the threshold. np.diff on that mask
    gives +1 where a sprint starts and -1 where it ends, so each contiguous
    run collapses to exactly one (start, end) pair — a 3-second sprint counts
    once, not 300 times. Padding the mask with False at both ends makes an
    effort that is already underway at the first sample, or still running at
    the last, close properly instead of being dropped.

    Runs shorter than MIN_SPRINT_DURATION_S are discarded: a lone frame over
    the line is GPS noise, not an effort.
    """
    above = speed > SPRINT_THRESHOLD_MS

    padded = np.concatenate(([False], above, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1    # inclusive last index of the run

    sprints = []
    for start, end in zip(starts, ends):
        duration = timestamps[end] - timestamps[start]
        if duration >= MIN_SPRINT_DURATION_S:
            sprints.append({
                "start_s": timestamps[start],
                "end_s": timestamps[end],
                "duration_s": duration,
                "peak_speed_ms": speed[start:end + 1].max(),
            })

    return sprints


def report(df, distance_m, sprints):
    """Print the session summary."""
    timestamps = df["timestamp"].values
    speed = df["speed_ms"].values

    session_time = timestamps[-1] - timestamps[0]
    max_speed = speed.max()

    # Integrating the logged speed is an independent estimate of distance; a
    # close match means the GPS track and the speed channel agree.
    dt = float(np.median(np.diff(timestamps)))
    distance_from_speed = speed.sum() * dt

    print("=" * 52)
    print("  SESSION SUMMARY")
    print("=" * 52)
    print(f"  Total Session Time .......  {session_time:8.2f} s")
    print(f"  Total Distance Covered ...  {distance_m:8.2f} m")
    print(f"  Maximum Speed Reached ....  {max_speed:8.2f} m/s")
    print(f"  Total Sprints Counted ....  {len(sprints):8d}")
    print("-" * 52)
    print(f"  (cross-check: speed integral {distance_from_speed:.2f} m, "
          f"delta {abs(distance_m - distance_from_speed):.2f} m)")

    if sprints:
        print(f"\n  Sprint detail (threshold {SPRINT_THRESHOLD_MS} m/s):")
        for i, s in enumerate(sprints, start=1):
            print(f"    #{i}  {s['start_s']:6.2f}s -> {s['end_s']:6.2f}s  "
                  f"({s['duration_s']:5.2f}s, peak {s['peak_speed_ms']:.2f} m/s)")
    print("=" * 52)


def main():
    # Resolve the log relative to the project root (the parent of analytics/)
    # so the script runs from any working directory.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "simulation", "simulated_match.csv")

    df = pd.read_csv(csv_path, usecols=["timestamp", "latitude", "longitude", "speed_ms"])
    print(f"Loaded {len(df)} samples from {csv_path}\n")

    distance_m, _ = total_distance(df["latitude"].values, df["longitude"].values)
    sprints = detect_sprints(df["timestamp"].values, df["speed_ms"].values)

    report(df, distance_m, sprints)


if __name__ == "__main__":
    main()
