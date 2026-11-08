"""FFT analysis of the synthetic PPG signal.

Reads the simulated match log, moves the optical heart-rate channel into the
frequency domain, and shows why raw PPG is unusable on a running player: the
3 Hz arm-swing artifact is a taller spectral peak than the 2 Hz cardiac
fundamental.

Outputs a 3-panel figure to simulation/fft_analysis.png.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# --- Analysis band ---------------------------------------------------------
# 0.5-5.0 Hz covers 30-300 BPM / SPM: every physiologically plausible heart
# rate and running cadence, while excluding DC drift and high-frequency noise.
BAND_LO, BAND_HI = 0.5, 5.0

SPRINT_START, SPRINT_END = 20.0, 35.0   # known high-intensity window (s)

# --- Light-surface palette (validated: all-pairs CVD dE 9.2, normal 24.0) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"          # primary ink -- titles
INK_SOFT = "#52514e"     # secondary ink -- axis labels, annotations
INK_MUTED = "#898781"    # tick labels
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PPG_COLOR = "#2a78d6"    # slot 1, blue   -- the PPG signal
PEAK_COLOR = "#eb6834"   # slot 2, orange -- detected spectral peaks
ACCEL_COLOR = "#1baf7a"  # slot 3, aqua   -- accelerometer


def load_session(csv_path):
    """Load the log and recover the acquisition parameters from the timestamps."""
    df = pd.read_csv(csv_path)

    n = len(df)
    dt = float(np.median(np.diff(df["timestamp"].values)))
    fs = 1.0 / dt
    duration = n * dt

    print(f"Loaded {csv_path}")
    print(f"  samples N = {n}   fs = {fs:.1f} Hz   duration = {duration:.1f} s")
    return df, n, fs, duration


def compute_spectrum(signal, n, fs):
    """Single-sided amplitude spectrum of `signal`.

    numpy's fft returns a two-sided complex spectrum that is symmetric for a
    real input, so we keep the first half (frequencies >= 0) and scale by
    2/N -- the factor of 2 folds the discarded negative half back in, giving
    magnitudes in the same units as the input signal's amplitude.
    """
    fft_vals = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1.0 / fs)

    half = n // 2
    return freqs[:half], (2.0 / n) * np.abs(fft_vals[:half])


def find_band_peaks(freqs, mags):
    """Locate prominent peaks inside the 0.5-5.0 Hz analysis band.

    Prominence is set relative to the band's own maximum so the threshold
    scales with signal strength instead of being a hard-coded magnitude.
    Returns peaks sorted strongest-first.
    """
    band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
    band_freqs, band_mags = freqs[band], mags[band]

    idx, _ = find_peaks(band_mags, prominence=0.05 * band_mags.max())
    idx = idx[np.argsort(band_mags[idx])[::-1]]   # strongest first

    return band_freqs[idx], band_mags[idx]


def cadence_from_accelerometer(df, fs):
    """Independent cadence estimate from the IMU during the sprint window.

    The accelerometer sees foot strikes but no blood flow, so its dominant
    frequency is pure cadence. Using it to tag the matching PPG peak is more
    honest than assuming the 3 Hz peak is always the artifact -- it is the same
    trick the real filter stage uses to know what to subtract.
    """
    window = (df["timestamp"] >= SPRINT_START) & (df["timestamp"] < SPRINT_END)
    segment = df.loc[window, "accel_z"].values
    segment = segment - segment.mean()   # drop the 1 g gravity offset (DC bin)

    freqs, mags = compute_spectrum(segment, len(segment), fs)
    band = (freqs >= BAND_LO) & (freqs <= BAND_HI)

    return freqs[band][np.argmax(mags[band])]


def classify_peaks(peak_freqs, peak_mags, accel_cadence):
    """Label each spectral peak as cadence artifact, heart rate, or harmonic.

    The peak closest to the accelerometer's cadence is the motion artifact;
    the strongest remaining peak is the cardiac fundamental.
    """
    labels = ["Harmonic / residual"] * len(peak_freqs)

    cadence_i = int(np.argmin(np.abs(peak_freqs - accel_cadence)))
    labels[cadence_i] = "Running cadence (motion artifact)"

    remaining = [i for i in range(len(peak_freqs)) if i != cadence_i]
    if remaining:
        hr_i = max(remaining, key=lambda i: peak_mags[i])
        labels[hr_i] = "Heart rate (cardiac fundamental)"

    return labels


def report(peak_freqs, peak_mags, labels, accel_cadence):
    """Print the detected peaks in Hz and the matching BPM / SPM."""
    print(f"\nIMU cadence reference ({SPRINT_START:.0f}-{SPRINT_END:.0f}s sprint): "
          f"{accel_cadence:.2f} Hz -> {accel_cadence * 60:.0f} SPM")
    print(f"\nPPG spectral peaks in {BAND_LO}-{BAND_HI} Hz "
          f"({len(peak_freqs)} found, strongest first):")

    for f, m, label in zip(peak_freqs, peak_mags, labels):
        # 1 Hz = 60 cycles/min, so the same number reads as BPM for a pulse
        # and SPM for a footstep.
        unit = "SPM" if "cadence" in label.lower() else "BPM"
        print(f"  {f:5.2f} Hz  ->  {f * 60:3.0f} {unit}   "
              f"magnitude {m:.3f}   {label}")

    if len(peak_mags) >= 2 and "cadence" in labels[0].lower():
        print(f"\n  Artifact dominates the pulse by "
              f"{peak_mags[0] / peak_mags[1]:.1f}x -- raw PPG would report the "
              f"cadence as the heart rate.")


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


def plot_time_domain(ax, df):
    """Panel 1 -- the first 5 s of raw PPG, where the artifact is visible."""
    window = df["timestamp"] <= 5.0
    ax.plot(df.loc[window, "timestamp"], df.loc[window, "ppg_raw"],
            color=PPG_COLOR, linewidth=1.2)

    style_axis(ax,
               "Raw PPG signal — time domain (first 5 s)",
               "Time (s)", "Amplitude (a.u.)")
    ax.set_xlim(0, 5)


def plot_spectrum(ax, freqs, mags, peak_freqs, peak_mags, labels):
    """Panel 2 -- the spectrum, with each detected peak marked and named."""
    view = freqs <= 10.0
    ax.plot(freqs[view], mags[view], color=PPG_COLOR, linewidth=1.2)

    # Mark only the peaks that carry meaning; a dot on every local maximum
    # would bury the two that matter.
    named = [i for i, label in enumerate(labels) if "Harmonic" not in label]
    ax.scatter(peak_freqs[named], peak_mags[named],
               s=42, color=PEAK_COLOR, zorder=3,
               edgecolors=SURFACE, linewidths=1.5)   # 2px surface ring

    for rank, i in enumerate(named):
        unit = "SPM" if "cadence" in labels[i].lower() else "BPM"
        short = labels[i].split(" (")[0]
        # Fan the labels outward -- tallest peak's label to the right, the
        # rest to the left, so no caption lies across a neighbouring spike.
        side = 1 if rank == 0 else -1
        ax.annotate(f"{short}\n{peak_freqs[i]:.2f} Hz · {peak_freqs[i] * 60:.0f} {unit}",
                    xy=(peak_freqs[i], peak_mags[i]),
                    xytext=(26 * side, 10), textcoords="offset points",
                    color=INK_SOFT, fontsize=9, va="center",
                    ha="left" if side > 0 else "right",
                    arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8))

    style_axis(ax,
               "PPG frequency spectrum — motion artifact outranks the pulse",
               "Frequency (Hz)", "Magnitude (2/N · |FFT|)")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, peak_mags.max() * 1.35 if len(peak_mags) else None)


def plot_accelerometer(ax, df):
    """Panel 3 -- vertical acceleration, with the sprint window called out."""
    ax.plot(df["timestamp"], df["accel_z"], color=ACCEL_COLOR, linewidth=0.7)

    # Shade the sprint as a region annotation, not a series -- neutral, behind
    # the data, so it never competes with the trace.
    ax.axvspan(SPRINT_START, SPRINT_END, color=GRID, zorder=0)
    ax.text((SPRINT_START + SPRINT_END) / 2, ax.get_ylim()[1],
            f"Sprint {SPRINT_START:.0f}–{SPRINT_END:.0f}s",
            ha="center", va="top", color=INK_SOFT, fontsize=9)

    style_axis(ax,
               "Accelerometer Z-axis — foot-strike impacts during the sprint",
               "Time (s)", "Acceleration (g)")
    ax.set_xlim(0, df["timestamp"].max())


def main():
    sim_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(sim_dir, "simulated_match.csv")
    png_path = os.path.join(sim_dir, "fft_analysis.png")

    df, n, fs, _ = load_session(csv_path)

    # --- Frequency-domain analysis of the PPG channel ---------------------
    freqs, mags = compute_spectrum(df["ppg_raw"].values, n, fs)
    peak_freqs, peak_mags = find_band_peaks(freqs, mags)

    accel_cadence = cadence_from_accelerometer(df, fs)
    labels = classify_peaks(peak_freqs, peak_mags, accel_cadence)
    report(peak_freqs, peak_mags, labels, accel_cadence)

    # --- Figure ------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.patch.set_facecolor(SURFACE)

    plot_time_domain(axes[0], df)
    plot_spectrum(axes[1], freqs, mags, peak_freqs, peak_mags, labels)
    plot_accelerometer(axes[2], df)

    fig.tight_layout(pad=2.0)
    fig.savefig(png_path, dpi=150, facecolor=SURFACE)
    print(f"\nSaved plot to {png_path}")

    # Only pop a window when driven interactively (python -i); a plain script
    # run just writes the PNG and exits.
    if sys.flags.interactive:
        plt.show()


if __name__ == "__main__":
    main()
