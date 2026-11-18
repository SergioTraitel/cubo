"""Live acquisition pipeline: MAX30102 heart rate (I2C) + Ultimate GPS (UART).

Records a session on the Raspberry Pi and writes one CSV row per PPG sample
(100 Hz), with the most recent GPS fix carried alongside each row. When the
recording stops (duration reached or Ctrl+C) the log is handed to
analytics.metrics for the distance / sprint summary.

    python run_pipeline.py                 # record until Ctrl+C
    python run_pipeline.py --duration 600  # record 10 minutes

Hardware state:
    * MAX30102 pulse oximeter  -- I2C bus 1, address 0x57          (required)
    * Adafruit Ultimate GPS    -- UART /dev/serial0, 9600 baud     (optional)
    * IMU (I2C 0x6a / 0x6b)    -- OFFLINE. The pipeline never touches these
      addresses, so it cannot hang waiting for them. Without the IMU there is
      no motion-artifact cancellation: while the player is running, the live
      HR estimate can lock onto the running cadence instead of the pulse.

Pi dependencies:  pip install pyserial pynmea2 smbus2 numpy pandas
"""

import argparse
import collections
import csv
import os
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# --- GPS (Adafruit Ultimate GPS, MTK3339) ------------------------------------
GPS_PORT = "/dev/serial0"
GPS_BAUD = 9600                 # module's factory default
KNOTS_TO_MS = 0.514444
GPS_STALE_S = 3.0               # no valid sentence for this long -> treat as no fix

# PMTK314: emit only RMC (position, speed, validity) and GGA (fix quality,
# satellites). Cuts the UART traffic to the two sentences we actually parse.
PMTK_RMC_GGA_ONLY = "PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"

# --- Heart rate (MAX30102) ---------------------------------------------------
I2C_BUS = 1
MAX30102_ADDR = 0x57
MAX30102_PART_ID = 0x15
PPG_FS = 100.0                  # must match SPO2_CONFIG below

HR_WINDOW_S = 8.0               # spectral window for the live BPM estimate
HR_BAND_HZ = (0.7, 3.5)         # 42-210 BPM
NO_FINGER_IR = 50_000           # IR DC level below this = sensor not on skin

LOG_DIR = os.path.join(PROJECT_ROOT, "logger", "sessions")
LOG_COLUMNS = [
    "timestamp", "ppg_raw", "ppg_red",
    "latitude", "longitude", "speed_ms", "gps_fix", "gps_sats",
    "hr_bpm",
]


# =============================================================================
# GPS
# =============================================================================

def nmea_command(body):
    """Wrap a PMTK body in '$...*CS\\r\\n' with the XOR checksum NMEA requires."""
    checksum = 0
    for ch in body:
        checksum ^= ord(ch)
    return f"${body}*{checksum:02X}\r\n".encode("ascii")


@dataclass
class GPSFix:
    valid: bool = False
    latitude: float = None
    longitude: float = None
    speed_ms: float = None
    sats: int = None
    updated_at: float = 0.0     # time.monotonic() of the last valid RMC


class GPSReader(threading.Thread):
    """Background thread: reads NMEA from the UART and keeps the latest fix.

    Runs as a daemon with a 1 s serial timeout, so a silent or unplugged GPS
    never blocks the heart-rate loop. If the port can't be opened, `error` is
    set and the thread exits; the pipeline keeps logging heart rate.
    """

    def __init__(self, port=GPS_PORT, baud=GPS_BAUD):
        super().__init__(name="gps-reader", daemon=True)
        self.port = port
        self.baud = baud
        self.error = None
        self._fix = GPSFix()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    def snapshot(self):
        """Copy of the latest fix; marked invalid if it has gone stale."""
        with self._lock:
            fix = replace(self._fix)
        if fix.valid and time.monotonic() - fix.updated_at > GPS_STALE_S:
            fix.valid = False
        return fix

    def stop(self):
        self._stop_event.set()

    def run(self):
        try:
            import pynmea2
            import serial
        except ImportError as exc:
            self.error = f"{exc} (pip install pyserial pynmea2)"
            return

        try:
            ser = serial.Serial(self.port, self.baud, timeout=1.0)
        except serial.SerialException as exc:
            self.error = str(exc)
            return

        with ser:
            ser.write(nmea_command(PMTK_RMC_GGA_ONLY))
            while not self._stop_event.is_set():
                try:
                    raw = ser.readline()
                except serial.SerialException as exc:
                    self.error = str(exc)
                    return
                line = raw.decode("ascii", errors="replace").strip()
                if not line.startswith("$"):
                    continue
                try:
                    self._handle(pynmea2.parse(line))
                except (pynmea2.ParseError, ValueError, TypeError, AttributeError):
                    continue   # garbled / partial sentence -- skip it

    def _handle(self, msg):
        kind = msg.sentence_type
        if kind == "RMC":
            with self._lock:
                if msg.status != "A":          # 'V' = receiver has no fix
                    self._fix.valid = False
                    return
                knots = msg.spd_over_grnd
                self._fix.valid = True
                self._fix.latitude = msg.latitude
                self._fix.longitude = msg.longitude
                self._fix.speed_ms = float(knots) * KNOTS_TO_MS if knots is not None else 0.0
                self._fix.updated_at = time.monotonic()
        elif kind == "GGA":
            with self._lock:
                self._fix.sats = int(msg.num_sats) if msg.num_sats else None


# =============================================================================
# MAX30102
# =============================================================================

class MAX30102:
    """Minimal MAX30102 driver: SpO2 mode (red + IR), 100 Hz, 18-bit samples."""

    REG_INTR_ENABLE_1 = 0x02
    REG_INTR_ENABLE_2 = 0x03
    REG_FIFO_WR_PTR = 0x04
    REG_OVF_COUNTER = 0x05
    REG_FIFO_RD_PTR = 0x06
    REG_FIFO_DATA = 0x07
    REG_FIFO_CONFIG = 0x08
    REG_MODE_CONFIG = 0x09
    REG_SPO2_CONFIG = 0x0A
    REG_LED1_PA = 0x0C          # red
    REG_LED2_PA = 0x0D          # IR
    REG_PART_ID = 0xFF

    BYTES_PER_SAMPLE = 6        # 3 bytes red + 3 bytes IR
    MAX_SAMPLES_PER_READ = 5    # 30 bytes, inside SMBus's 32-byte block limit

    def __init__(self, bus=I2C_BUS, address=MAX30102_ADDR):
        from smbus2 import SMBus

        self.address = address
        self.bus = SMBus(bus)
        part = self._read(self.REG_PART_ID)
        if part != MAX30102_PART_ID:
            self.bus.close()
            raise RuntimeError(f"MAX30102 not found at 0x{address:02x} "
                               f"(part id 0x{part:02x}, expected 0x{MAX30102_PART_ID:02x})")
        self._configure()

    def _read(self, reg):
        return self.bus.read_byte_data(self.address, reg)

    def _write(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value)

    def _configure(self):
        self._write(self.REG_MODE_CONFIG, 0x40)            # soft reset
        deadline = time.monotonic() + 1.0
        while self._read(self.REG_MODE_CONFIG) & 0x40:
            if time.monotonic() > deadline:
                raise RuntimeError("MAX30102 reset did not complete")
            time.sleep(0.01)

        self._write(self.REG_INTR_ENABLE_1, 0x00)          # polled, no interrupts
        self._write(self.REG_INTR_ENABLE_2, 0x00)
        self._write(self.REG_FIFO_WR_PTR, 0x00)
        self._write(self.REG_OVF_COUNTER, 0x00)
        self._write(self.REG_FIFO_RD_PTR, 0x00)
        self._write(self.REG_FIFO_CONFIG, 0x1F)            # no averaging, rollover on
        # ADC range 4096 nA | 100 samples/s | 411 us pulse (18-bit)
        self._write(self.REG_SPO2_CONFIG, 0b0_01_001_11)
        self._write(self.REG_LED1_PA, 0x24)                # ~7 mA
        self._write(self.REG_LED2_PA, 0x24)
        self._write(self.REG_MODE_CONFIG, 0x03)            # SpO2 mode: red + IR

    def read_fifo(self):
        """Drain the FIFO. Returns ([(red, ir), ...], samples_lost_to_overflow)."""
        wr = self._read(self.REG_FIFO_WR_PTR) & 0x1F
        rd = self._read(self.REG_FIFO_RD_PTR) & 0x1F
        lost = self._read(self.REG_OVF_COUNTER) & 0x1F
        available = (wr - rd) & 0x1F
        if available == 0 and lost:
            available = 32     # pointers equal because the FIFO is full, not empty

        samples = []
        while available > 0:
            k = min(available, self.MAX_SAMPLES_PER_READ)
            data = self.bus.read_i2c_block_data(self.address, self.REG_FIFO_DATA,
                                                k * self.BYTES_PER_SAMPLE)
            for i in range(k):
                b = data[i * 6:(i + 1) * 6]
                red = ((b[0] << 16) | (b[1] << 8) | b[2]) & 0x3FFFF
                ir = ((b[3] << 16) | (b[4] << 8) | b[5]) & 0x3FFFF
                samples.append((red, ir))
            available -= k
        return samples, lost

    def close(self):
        try:
            self._write(self.REG_MODE_CONFIG, 0x80)        # shutdown, LEDs off
        finally:
            self.bus.close()


# =============================================================================
# Live heart-rate estimate (no IMU -> no motion-artifact cancellation)
# =============================================================================

def estimate_hr_bpm(ir_window):
    """Dominant frequency of the IR channel in the cardiac band, in BPM.

    IMU BYPASS: the original design subtracts the accelerometer's cadence peak
    before picking the pulse. With the IMU offline we simply take the
    strongest in-band peak, which is correct at rest but may report the
    running cadence (SPM) as the heart rate during hard running.
    """
    n = len(ir_window)
    if n < HR_WINDOW_S * PPG_FS:
        return None
    x = np.asarray(ir_window, dtype=float)
    if x.mean() < NO_FINGER_IR:
        return None

    t = np.arange(n)
    x = x - np.polyval(np.polyfit(t, x, 1), t)   # remove DC + slow drift
    x *= np.hanning(n)

    nfft = 4096                                   # zero-pad -> ~1.5 BPM bins
    mags = np.abs(np.fft.rfft(x, nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / PPG_FS)
    band = (freqs >= HR_BAND_HZ[0]) & (freqs <= HR_BAND_HZ[1])
    return 60.0 * freqs[band][np.argmax(mags[band])]


# =============================================================================
# Pipeline
# =============================================================================

def _fmt(value, spec):
    return "" if value is None else format(value, spec)


def record(out_path, duration_s=None, gps_port=GPS_PORT, gps_baud=GPS_BAUD):
    """Log PPG + GPS until `duration_s` elapses or Ctrl+C. Returns rows written."""
    sensor = MAX30102()
    gps = GPSReader(gps_port, gps_baud)
    gps.start()

    ir_window = collections.deque(maxlen=int(HR_WINDOW_S * PPG_FS))
    sample_index = 0          # sensor-clock sample count -> uniform timestamps
    rows = 0
    hr_bpm = None
    gps_error_reported = False

    print(f"Recording to {out_path}  (Ctrl+C to stop)")
    t_start = time.monotonic()
    next_status = t_start + 1.0

    try:
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(LOG_COLUMNS)

            try:
                while duration_s is None or time.monotonic() - t_start < duration_s:
                    samples, lost = sensor.read_fifo()
                    if lost:
                        print(f"  ! PPG FIFO overflow: {lost} sample(s) dropped")
                        sample_index += lost       # keep timestamps on the sensor clock

                    fix = gps.snapshot()
                    lat = fix.latitude if fix.valid else None
                    lon = fix.longitude if fix.valid else None
                    spd = fix.speed_ms if fix.valid else None

                    for red, ir in samples:
                        ir_window.append(ir)
                        writer.writerow([
                            f"{sample_index / PPG_FS:.2f}", ir, red,
                            _fmt(lat, ".7f"), _fmt(lon, ".7f"), _fmt(spd, ".3f"),
                            int(fix.valid), _fmt(fix.sats, "d"),
                            _fmt(hr_bpm, ".1f"),
                        ])
                        sample_index += 1
                        rows += 1

                    now = time.monotonic()
                    if now >= next_status:
                        next_status += 1.0
                        hr_bpm = estimate_hr_bpm(ir_window)
                        f.flush()

                        if gps.error and not gps_error_reported:
                            print(f"  ! GPS unavailable ({gps.error}); "
                                  f"logging heart rate only")
                            gps_error_reported = True

                        gps_txt = (f"{fix.latitude:.6f},{fix.longitude:.6f} "
                                   f"{fix.speed_ms:4.1f} m/s sats={fix.sats}"
                                   if fix.valid else "no fix")
                        hr_txt = f"{hr_bpm:5.1f} BPM" if hr_bpm else "  --- BPM"
                        print(f"  t={now - t_start:7.1f}s  {hr_txt}  GPS {gps_txt}")

                    time.sleep(0.02)   # FIFO holds 32 samples = 320 ms of headroom
            except KeyboardInterrupt:
                print("\nStopped by user.")
    finally:
        gps.stop()
        sensor.close()
        gps.join(timeout=2.0)

    print(f"Wrote {rows} samples ({rows / PPG_FS:.1f} s) to {out_path}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--duration", type=float, default=None,
                        help="seconds to record (default: until Ctrl+C)")
    parser.add_argument("--out", default=None,
                        help="CSV path (default: logger/sessions/session_<time>.csv)")
    parser.add_argument("--gps-port", default=GPS_PORT)
    parser.add_argument("--gps-baud", type=int, default=GPS_BAUD)
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(LOG_DIR, f"session_{stamp}.csv")

    rows = record(out_path, args.duration, args.gps_port, args.gps_baud)

    if rows:
        from analytics.metrics import summarize
        print()
        summarize(out_path)


if __name__ == "__main__":
    main()
