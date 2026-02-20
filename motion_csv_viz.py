#!/usr/bin/env python3
"""
Visualize Gyroflow motion CSV exports and inspect vibration frequencies.

Expected columns (when exported from Motion Data -> Export camera data):
  - timestamp_ms
  - org_gyro_x, org_gyro_y, org_gyro_z
  - optional: org_acc_x/y/z, stab_pitch/yaw/roll, etc.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib numpy"
    ) from exc


def load_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        cols: Dict[str, List[float]] = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k in cols:
                v = row.get(k, "")
                if v is None or v == "":
                    cols[k].append(float("nan"))
                else:
                    try:
                        cols[k].append(float(v))
                    except ValueError:
                        cols[k].append(float("nan"))

    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def estimate_sample_rate(timestamp_ms: np.ndarray) -> float:
    ts = timestamp_ms[np.isfinite(timestamp_ms)]
    if ts.size < 2:
        return 0.0
    d = np.diff(ts)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return 0.0
    return 1000.0 / float(np.median(d))


def sanitize_signal(x: np.ndarray) -> np.ndarray:
    y = np.array(x, dtype=float)
    finite = np.isfinite(y)
    if not finite.any():
        return np.zeros_like(y)
    fill = float(np.nanmedian(y[finite]))
    y[~finite] = fill
    y -= np.mean(y)
    return y


def spectrum(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    if fs <= 0 or x.size < 8:
        return np.array([]), np.array([])
    y = sanitize_signal(x)
    win = np.hanning(y.size)
    yw = y * win
    yf = np.fft.rfft(yw)
    f = np.fft.rfftfreq(y.size, d=1.0 / fs)
    a = np.abs(yf)
    return f, a


def top_peaks(freq: np.ndarray, amp: np.ndarray, low_hz: float, high_hz: float, top_n: int) -> List[Tuple[float, float]]:
    if freq.size == 0:
        return []
    mask = (freq >= low_hz) & (freq <= high_hz)
    f = freq[mask]
    a = amp[mask]
    if a.size == 0:
        return []
    idx = np.argpartition(a, -min(top_n, a.size))[-min(top_n, a.size):]
    peaks = sorted(((float(f[i]), float(a[i])) for i in idx), key=lambda t: t[1], reverse=True)
    return peaks


def get_cols(data: Dict[str, np.ndarray], names: List[str]) -> List[np.ndarray]:
    out = []
    for n in names:
        if n not in data:
            raise KeyError(f"Missing column '{n}' in CSV")
        out.append(data[n])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize Gyroflow motion CSV and analyze vibration frequencies")
    p.add_argument("csv_file", type=Path, help="Path to exported gyro CSV")
    p.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <csv_stem>_viz.png)")
    p.add_argument("--show", action="store_true", help="Show interactive window")
    p.add_argument("--peak-low", type=float, default=10.0, help="Low frequency bound for peak search")
    p.add_argument("--peak-high", type=float, default=400.0, help="High frequency bound for peak search")
    p.add_argument("--top-n", type=int, default=5, help="Number of top peaks to print")
    args = p.parse_args()

    data = load_csv(args.csv_file)
    if "timestamp_ms" not in data:
        raise SystemExit("CSV must contain 'timestamp_ms'")

    t_ms = data["timestamp_ms"]
    t_s = (t_ms - np.nanmin(t_ms)) / 1000.0
    fs = estimate_sample_rate(t_ms)
    print(f"Estimated sample rate: {fs:.2f} Hz")

    gyro_names = ["org_gyro_x", "org_gyro_y", "org_gyro_z"]
    has_gyro = all(n in data for n in gyro_names)
    acc_names = ["org_acc_x", "org_acc_y", "org_acc_z"]
    has_acc = all(n in data for n in acc_names)
    stab_euler_names = ["stab_pitch", "stab_yaw", "stab_roll"]
    has_stab = all(n in data for n in stab_euler_names)

    if not has_gyro and not has_stab:
        raise SystemExit(
            "CSV does not contain expected motion columns. Include original gyroscope and/or stabilized euler angles in export."
        )

    rows = 0
    if has_gyro:
        rows += 2
    if has_acc:
        rows += 1
    if has_stab:
        rows += 1

    fig, axes = plt.subplots(rows, 1, figsize=(14, 3.2 * rows), constrained_layout=True)
    if rows == 1:
        axes = [axes]
    ax_i = 0

    if has_gyro:
        gx, gy, gz = get_cols(data, gyro_names)
        gmag = np.sqrt(np.square(gx) + np.square(gy) + np.square(gz))
        axes[ax_i].plot(t_s, gx, lw=0.8, label="gyro x")
        axes[ax_i].plot(t_s, gy, lw=0.8, label="gyro y")
        axes[ax_i].plot(t_s, gz, lw=0.8, label="gyro z")
        axes[ax_i].plot(t_s, gmag, lw=1.0, alpha=0.7, label="gyro |mag|")
        axes[ax_i].set_title("Original Gyro Signal (time domain)")
        axes[ax_i].set_xlabel("Time (s)")
        axes[ax_i].set_ylabel("Angular rate")
        axes[ax_i].legend(loc="upper right", ncol=4, fontsize=8)
        axes[ax_i].grid(alpha=0.25)
        ax_i += 1

        f_x, a_x = spectrum(gx, fs)
        _, a_y = spectrum(gy, fs)
        _, a_z = spectrum(gz, fs)
        _, a_m = spectrum(gmag, fs)
        axes[ax_i].plot(f_x, a_x, lw=1.0, label="x")
        axes[ax_i].plot(f_x, a_y, lw=1.0, label="y")
        axes[ax_i].plot(f_x, a_z, lw=1.0, label="z")
        axes[ax_i].plot(f_x, a_m, lw=1.2, label="|mag|")
        axes[ax_i].set_title("Original Gyro Spectrum (FFT magnitude)")
        axes[ax_i].set_xlabel("Frequency (Hz)")
        axes[ax_i].set_ylabel("Magnitude")
        axes[ax_i].set_xlim(left=0, right=max(10.0, args.peak_high * 1.1))
        axes[ax_i].legend(loc="upper right", ncol=4, fontsize=8)
        axes[ax_i].grid(alpha=0.25)

        peaks = top_peaks(f_x, a_m, args.peak_low, args.peak_high, args.top_n)
        if peaks:
            print(f"Top {len(peaks)} vibration peaks from gyro |mag| in [{args.peak_low:.1f}, {args.peak_high:.1f}] Hz:")
            for hz, amp in peaks:
                print(f"  {hz:8.2f} Hz  amp={amp:.3f}")
        ax_i += 1

    if has_acc:
        ax, ay, az = get_cols(data, acc_names)
        amag = np.sqrt(np.square(ax) + np.square(ay) + np.square(az))
        axes[ax_i].plot(t_s, ax, lw=0.8, label="acc x")
        axes[ax_i].plot(t_s, ay, lw=0.8, label="acc y")
        axes[ax_i].plot(t_s, az, lw=0.8, label="acc z")
        axes[ax_i].plot(t_s, amag, lw=1.0, alpha=0.7, label="acc |mag|")
        axes[ax_i].set_title("Original Accelerometer Signal (time domain)")
        axes[ax_i].set_xlabel("Time (s)")
        axes[ax_i].set_ylabel("Acceleration")
        axes[ax_i].legend(loc="upper right", ncol=4, fontsize=8)
        axes[ax_i].grid(alpha=0.25)
        ax_i += 1

    if has_stab:
        sp, sy, sr = get_cols(data, stab_euler_names)
        axes[ax_i].plot(t_s, sp, lw=0.9, label="stab pitch")
        axes[ax_i].plot(t_s, sy, lw=0.9, label="stab yaw")
        axes[ax_i].plot(t_s, sr, lw=0.9, label="stab roll")
        axes[ax_i].set_title("Stabilized Motion (Euler angles)")
        axes[ax_i].set_xlabel("Time (s)")
        axes[ax_i].set_ylabel("Degrees")
        axes[ax_i].legend(loc="upper right", ncol=3, fontsize=8)
        axes[ax_i].grid(alpha=0.25)
        ax_i += 1

    out = args.out or args.csv_file.with_name(f"{args.csv_file.stem}_viz.png")
    fig.savefig(out, dpi=150)
    print(f"Wrote: {out}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

