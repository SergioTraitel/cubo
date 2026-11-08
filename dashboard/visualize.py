"""Session dashboard: speed profile, sprint efforts, and raw PPG.

Loads a match log, recomputes the headline metrics, and renders a two-panel
summary to dashboard/session_summary.png.

Distance and sprint detection are imported from analytics.metrics rather than
reimplemented here, so the dashboard and the analytics report can never drift
apart on the Haversine math or the 5.5 m/s threshold.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from analytics.metrics import (  # noqa: E402  (needs the path fix above)
    SPRINT_THRESHOLD_MS,
    detect_sprints,
    total_distance,
)

PPG_PREVIEW_S = 10.0   # seconds of raw PPG shown in the lower panel

# --- Light-surface palette (validated: all-pairs CVD dE 9.2, normal 24.0) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"           # primary ink -- titles, stat values
INK_SOFT = "#52514e"      # secondary ink -- axis labels
INK_MUTED = "#898781"     # tick labels, stat captions
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SPEED_COLOR = "#2a78d6"   # slot 1, blue -- speed trace
SPEED_FILL = "#cde2fb"    # blue step 100 -- area under the speed curve
PPG_COLOR = "#eb6834"     # slot 2, orange -- raw PPG
THRESHOLD_COLOR = "#d03b3b"   # status:critical -- the sprint threshold line


def load_session(csv_path):
    """Load the log and compute the metrics shown on the dashboard."""
    df = pd.read_csv(csv_path)

    distance_m, _ = total_distance(df["latitude"].values, df["longitude"].values)
    sprints = detect_sprints(df["timestamp"].values, df["speed_ms"].values)

    metrics = {
        # ptp = peak-to-peak: last timestamp minus first, i.e. the span the
        # log actually covers rather than an assumed 60 s.
        "session_time_s": float(np.ptp(df["timestamp"].values)),
        "distance_m": distance_m,
        "sprint_count": len(sprints),
        "max_speed_ms": float(np.max(df["speed_ms"].values)),
    }
    return df, metrics, sprints


def style_axis(ax, title, xlabel, ylabel):
    """Recessive chrome: hairline y-grid, no top/right spines, muted ticks."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=12, fontweight="bold",
                 loc="left", pad=10)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=10)

    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(colors=INK_MUTED, labelsize=9, length=3, width=0.8)


def draw_header(fig, metrics):
    """Title plus a KPI row — the three numbers a coach reads first.

    Drawn as figure-level text rather than a boxed annotation inside an axis so
    the values never collide with the data or clip at a panel edge.
    """
    fig.text(0.065, 0.955, "Session Summary", color=INK,
             fontsize=15, fontweight="bold", va="top")
    fig.text(0.065, 0.915,
             f"Simulated match · peak speed {metrics['max_speed_ms']:.2f} m/s",
             color=INK_MUTED, fontsize=9.5, va="top")

    stats = [
        ("TOTAL TIME", f"{metrics['session_time_s']:.1f} s"),
        ("DISTANCE", f"{metrics['distance_m']:.1f} m"),
        ("SPRINTS", f"{metrics['sprint_count']}"),
    ]
    for i, (caption, value) in enumerate(stats):
        x = 0.50 + i * 0.16
        fig.text(x, 0.955, caption, color=INK_MUTED, fontsize=8.5,
                 fontweight="bold", va="top")
        fig.text(x, 0.925, value, color=INK, fontsize=17, va="top")


def plot_speed_profile(ax, df, sprints):
    """Panel 1 — speed over the session, with the sprint threshold and efforts."""
    t = df["timestamp"].values
    speed = df["speed_ms"].values

    # Fill to zero so the area reads as "ground covered" under the trace.
    ax.fill_between(t, speed, 0, color=SPEED_FILL, zorder=1)
    ax.plot(t, speed, color=SPEED_COLOR, linewidth=1.0, zorder=3)

    # Mark each detected effort — this is the "sprints" stat made visible, so
    # the count above is auditable against the chart rather than asserted.
    # Drawn as a slim bar pinned to the top of the panel (ymin/ymax are axes
    # fractions) instead of a full-height wash, which would sit over the data.
    for i, sprint in enumerate(sprints):
        ax.axvspan(sprint["start_s"], sprint["end_s"], ymin=0.93, ymax=1.0,
                   color=SPEED_COLOR, zorder=5,
                   label="Detected sprint" if i == 0 else None)

    ax.axhline(SPRINT_THRESHOLD_MS, color=THRESHOLD_COLOR, linestyle="--",
               linewidth=1.2, zorder=4,
               label=f"Sprint threshold ({SPRINT_THRESHOLD_MS} m/s)")

    style_axis(ax, "Player Speed Profile", "Time (s)", "Speed (m/s)")
    ax.set_xlim(t.min(), t.max())
    # Headroom so the sprint bar and legend clear the trace.
    ax.set_ylim(0, max(speed.max(), SPRINT_THRESHOLD_MS) * 1.30)

    legend = ax.legend(loc="upper right", frameon=False, fontsize=9,
                       ncol=2, handlelength=1.8, borderaxespad=1.2)
    for text in legend.get_texts():
        text.set_color(INK_SOFT)


def plot_raw_ppg(ax, df):
    """Panel 2 — raw optical pulse, first 10 s.

    Shown unfiltered on purpose: the visible waveform is dominated by the
    running cadence, not the pulse (see simulation/test_fft.py).
    """
    window = df["timestamp"] <= PPG_PREVIEW_S
    ax.plot(df.loc[window, "timestamp"], df.loc[window, "ppg_raw"],
            color=PPG_COLOR, linewidth=1.0)

    style_axis(ax,
               f"Heart Rate & Motion — raw PPG (first {PPG_PREVIEW_S:.0f} s)",
               "Time (s)", "Amplitude (a.u.)")
    ax.set_xlim(0, PPG_PREVIEW_S)


def main():
    csv_path = os.path.join(PROJECT_ROOT, "simulation", "simulated_match.csv")
    png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "session_summary.png")

    df, metrics, sprints = load_session(csv_path)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor(SURFACE)

    draw_header(fig, metrics)
    plot_speed_profile(axes[0], df, sprints)
    plot_raw_ppg(axes[1], df)

    # Explicit spacing rather than tight_layout -- the header occupies the top
    # of the figure and tight_layout would lay the panels over it.
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.065, right=0.97, hspace=0.42)

    fig.savefig(png_path, dpi=150, facecolor=SURFACE)

    print(f"Dashboard generated: {png_path}")
    print(f"  {metrics['session_time_s']:.1f} s  ·  "
          f"{metrics['distance_m']:.1f} m  ·  "
          f"{metrics['sprint_count']} sprint(s)  ·  "
          f"peak {metrics['max_speed_ms']:.2f} m/s")

    # Only pop a window when driven interactively (python -i); a plain script
    # run writes the PNG and exits instead of blocking on a GUI event loop.
    if sys.flags.interactive:
        plt.show()


if __name__ == "__main__":
    main()
