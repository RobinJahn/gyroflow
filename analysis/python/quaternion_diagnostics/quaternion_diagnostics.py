#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = SCRIPT_DIR.parent
DATA_DIR = PYTHON_ROOT / "data"


Q_BUTTERWORTH = 1.0 / math.sqrt(2.0)


def parse_float(raw: str) -> float:
    if raw is None:
        return float("nan")
    s = raw.strip()
    if not s:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def is_finite(v: float) -> bool:
    return math.isfinite(v)


def to_iso_time(t_sec: float) -> str:
    base = dt.datetime(1970, 1, 1)
    return (base + dt.timedelta(seconds=t_sec)).isoformat(timespec="milliseconds")


def load_csv(path: Path) -> Dict[str, List[float]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        cols: Dict[str, List[float]] = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k in cols:
                cols[k].append(parse_float(row.get(k, "")))
    return cols


def normalize_timestamps_ms(ts_ms: List[float]) -> List[float]:
    finite = [x for x in ts_ms if is_finite(x)]
    if not finite:
        return [float("nan")] * len(ts_ms)
    t0 = min(finite)
    return [((x - t0) / 1000.0) if is_finite(x) else float("nan") for x in ts_ms]


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    n = len(vals)
    m = n // 2
    if n % 2 == 1:
        return vals[m]
    return 0.5 * (vals[m - 1] + vals[m])


def estimate_sample_rate(t_sec: List[float]) -> float:
    valid = [x for x in t_sec if is_finite(x)]
    if len(valid) < 2:
        return 0.0
    valid.sort()
    diffs = [b - a for a, b in zip(valid, valid[1:]) if b > a]
    if not diffs:
        return 0.0
    md = median(diffs)
    return 1.0 / md if md > 0 else 0.0


def quat_norm(q: Tuple[float, float, float, float]) -> float:
    w, x, y, z = q
    return math.sqrt(w * w + x * x + y * y + z * z)


def quat_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    n = quat_norm(q)
    if n <= 0 or not is_finite(n):
        return (float("nan"), float("nan"), float("nan"), float("nan"))
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def quat_dot(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def quat_conj(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def quat_mul(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_inverse_unit(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return quat_conj(q)


def quat_scaled_axis(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    # Equivalent to nalgebra UnitQuaternion::scaled_axis() for unit q.
    w, x, y, z = quat_normalize(q)
    if not all(is_finite(v) for v in (w, x, y, z)):
        return (float("nan"), float("nan"), float("nan"))

    w = max(-1.0, min(1.0, w))
    v_norm = math.sqrt(x * x + y * y + z * z)
    angle = 2.0 * math.atan2(v_norm, w)
    if v_norm < 1e-12:
        return (0.0, 0.0, 0.0)
    s = angle / v_norm
    return (x * s, y * s, z * s)


def compute_quat_series(data: Dict[str, List[float]]) -> List[Tuple[float, float, float, float]]:
    keys = ("org_quat_w", "org_quat_x", "org_quat_y", "org_quat_z")
    if not all(k in data for k in keys):
        raise ValueError("Input CSV must contain org_quat_w, org_quat_x, org_quat_y, org_quat_z")
    out = []
    for w, x, y, z in zip(data[keys[0]], data[keys[1]], data[keys[2]], data[keys[3]]):
        if all(is_finite(v) for v in (w, x, y, z)):
            out.append(quat_normalize((w, x, y, z)))
        else:
            out.append((float("nan"), float("nan"), float("nan"), float("nan")))
    return out


def compute_dot_and_jump(t: List[float], q: List[Tuple[float, float, float, float]], large_jump_thr: float) -> Tuple[List[float], List[dict]]:
    dots = [float("nan")] * len(q)
    events: List[dict] = []
    for i in range(1, len(q)):
        qa, qb = q[i - 1], q[i]
        if not all(is_finite(v) for v in qa + qb):
            continue
        d = quat_dot(qa, qb)
        dots[i] = d
        if d < 0.0:
            events.append({"event_type": "sign_flip_candidate", "index": i, "timestamp_s": t[i], "value": d, "detail": "dot<0"})
        if d < large_jump_thr:
            events.append({"event_type": "large_jump_candidate", "index": i, "timestamp_s": t[i], "value": d, "detail": f"dot<{large_jump_thr}"})
    return dots, events


def compute_dt_stats(t: List[float], z_thresh: float) -> Tuple[List[float], dict, List[dict]]:
    dt = [float("nan")] * len(t)
    vals: List[float] = []
    for i in range(1, len(t)):
        if is_finite(t[i]) and is_finite(t[i - 1]):
            d = t[i] - t[i - 1]
            dt[i] = d
            if d > 0:
                vals.append(d)

    if vals:
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(var)
        mn = min(vals)
        mx = max(vals)
    else:
        mean = std = mn = mx = float("nan")

    events: List[dict] = []
    hi = (mean + z_thresh * std) if is_finite(mean) and is_finite(std) else float("nan")
    for i in range(1, len(dt)):
        d = dt[i]
        if not is_finite(d):
            continue
        if d <= 0:
            events.append({"event_type": "dt_non_positive", "index": i, "timestamp_s": t[i], "value": d, "detail": "dt<=0"})
        elif is_finite(hi) and d > hi:
            events.append({"event_type": "dt_outlier", "index": i, "timestamp_s": t[i], "value": d, "detail": f"dt>{hi:.9f}"})

    return dt, {"min": mn, "max": mx, "mean": mean, "std": std, "high_threshold": hi}, events


def derive_gyro_from_quats(t: List[float], q: List[Tuple[float, float, float, float]]) -> Tuple[List[Tuple[float, float, float]], List[float]]:
    # Mirrors src/core/gyro_export.rs::derived_gyro_from_quats
    w_xyz: List[Tuple[float, float, float]] = [(float("nan"), float("nan"), float("nan")) for _ in q]
    mag: List[float] = [float("nan")] * len(q)

    for i in range(1, len(q)):
        prev_q = q[i - 1]
        cur_q = q[i]
        if not all(is_finite(v) for v in prev_q + cur_q):
            continue
        dt_s = t[i] - t[i - 1]
        if not is_finite(dt_s) or dt_s <= 0:
            continue

        delta = quat_mul(quat_inverse_unit(prev_q), cur_q)
        rot = quat_scaled_axis(delta)  # radians
        wx = rot[0] / dt_s
        wy = rot[1] / dt_s
        wz = rot[2] / dt_s
        w_xyz[i] = (wx, wy, wz)
        mag[i] = math.sqrt(wx * wx + wy * wy + wz * wz)

    return w_xyz, mag


class BiquadLP:
    def __init__(self, fs: float, fc: float, q: float = Q_BUTTERWORTH):
        omega = 2.0 * math.pi * (fc / fs)
        cosw = math.cos(omega)
        sinw = math.sin(omega)
        alpha = sinw / (2.0 * q)

        b0 = (1.0 - cosw) * 0.5
        b1 = 1.0 - cosw
        b2 = (1.0 - cosw) * 0.5
        a0 = 1.0 + alpha
        a1 = -2.0 * cosw
        a2 = 1.0 - alpha

        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a1 = a1 / a0
        self.a2 = a2 / a0

        self.z1 = 0.0
        self.z2 = 0.0

    def run(self, x: float) -> float:
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y


def finite_fill(x: List[float]) -> List[float]:
    valid = [v for v in x if is_finite(v)]
    fill = median(valid) if valid else 0.0
    return [v if is_finite(v) else fill for v in x]


def lowpass_forward_backward(x: List[float], fs: float, fc: float, strength: float = 1.0) -> List[float]:
    if fs <= 0 or fc <= 0:
        return x[:]
    nyq = 0.5 * fs
    if fc >= nyq * 0.999:
        return x[:]

    passes = max(0, int(round(strength)))
    if passes == 0:
        return x[:]

    y = finite_fill(x)
    for _ in range(passes):
        fwd = BiquadLP(fs, fc)
        y = [fwd.run(v) for v in y]

        bwd = BiquadLP(fs, fc)
        y_rev = [bwd.run(v) for v in reversed(y)]
        y = list(reversed(y_rev))
    return y


def top_peaks(signal_t: List[float], y: List[float], n: int) -> List[Tuple[int, float, float]]:
    cand: List[Tuple[int, float, float]] = []
    for i in range(1, len(y) - 1):
        a, b, c = y[i - 1], y[i], y[i + 1]
        if all(is_finite(v) for v in (a, b, c)) and b >= a and b >= c:
            cand.append((i, signal_t[i], b))
    cand.sort(key=lambda x: x[2], reverse=True)
    return cand[: max(0, n)]


def write_events_csv(path: Path, events: List[dict]) -> None:
    fields = ["event_type", "index", "timestamp_s", "value", "detail"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in events:
            w.writerow({k: e.get(k, "") for k in fields})


def build_plot_html(
    out_path: Path,
    title: str,
    t_iso: List[str],
    dot_series: List[float],
    sign_flip_events: List[dict],
    large_jump_events: List[dict],
    dt_series: List[float],
    dt_events: List[dict],
    w350: List[float],
    w2: List[float],
    residual: List[float],
    residual_peaks: List[Tuple[int, float, float]],
) -> None:
    def pick_xy(events: List[dict]) -> Tuple[List[str], List[float]]:
        xs, ys = [], []
        for e in events:
            i = int(e["index"])
            if 0 <= i < len(t_iso):
                xs.append(t_iso[i])
                ys.append(float(e["value"]))
        return xs, ys

    sf_x, sf_y = pick_xy(sign_flip_events)
    lj_x, lj_y = pick_xy(large_jump_events)
    dt_x, dt_y = pick_xy(dt_events)

    rp_x = [t_iso[i] for i, _, _ in residual_peaks if 0 <= i < len(t_iso)]
    rp_y = [v for _, _, v in residual_peaks]

    payload = {
        "t": t_iso,
        "dot": dot_series,
        "sf_x": sf_x,
        "sf_y": sf_y,
        "lj_x": lj_x,
        "lj_y": lj_y,
        "dt": dt_series,
        "dt_x": dt_x,
        "dt_y": dt_y,
        "w350": w350,
        "w2": w2,
        "residual": residual,
        "rp_x": rp_x,
        "rp_y": rp_y,
    }

    html = f"""<!doctype html>
<html><head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
<title>{title}</title>
<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 14px; background: #f6f8fb; }}
.panel {{ background: white; border: 1px solid #dbe2ea; border-radius: 10px; padding: 10px; margin-bottom: 14px; }}
.plot {{ width: 100%; height: 420px; }}
</style>
</head><body>
<div class=\"panel\"><div id=\"p1\" class=\"plot\"></div></div>
<div class=\"panel\"><div id=\"p2\" class=\"plot\"></div></div>
<div class=\"panel\"><div id=\"p3\" class=\"plot\"></div></div>
<script>
const D = {json.dumps(payload)};
const tl = {{type:'date', tickformat:'%M:%S', hoverformat:'%M:%S.%L'}};

Plotly.newPlot('p1', [
  {{x:D.t, y:D.dot, type:'scatter', mode:'lines', name:'dot(q[i-1], q[i])'}},
  {{x:D.sf_x, y:D.sf_y, type:'scatter', mode:'markers', name:'sign flip candidates', marker:{{color:'#d62728', size:8}}}},
  {{x:D.lj_x, y:D.lj_y, type:'scatter', mode:'markers', name:'large jump candidates', marker:{{color:'#ff7f0e', size:7}}}}
], {{title:'Quaternion Continuity', template:'plotly_white', hovermode:'x unified', xaxis:tl, yaxis:{{title:'dot'}}}}, {{responsive:true}});

Plotly.newPlot('p2', [
  {{x:D.t, y:D.dt, type:'scatter', mode:'lines', name:'dt (s)'}},
  {{x:D.dt_x, y:D.dt_y, type:'scatter', mode:'markers', name:'dt outliers', marker:{{color:'#d62728', size:8}}}}
], {{title:'Timestamp Delta Sanity', template:'plotly_white', hovermode:'x unified', xaxis:tl, yaxis:{{title:'dt (s)'}}}}, {{responsive:true}});

Plotly.newPlot('p3', [
  {{x:D.t, y:D.w350, type:'scatter', mode:'lines', name:'|ω|_350Hz', line:{{color:'#1f77b4'}}}},
  {{x:D.t, y:D.w2, type:'scatter', mode:'lines', name:'|ω|_2Hz', line:{{color:'#2ca02c'}}}},
  {{x:D.t, y:D.residual, type:'scatter', mode:'lines', name:'residual |ω|_350-|ω|_2', line:{{color:'#e4572e'}}}},
  {{x:D.rp_x, y:D.rp_y, type:'scatter', mode:'markers', name:'top residual peaks', marker:{{color:'#d62728', size:8}}}}
], {{title:'High-frequency Residual / Divergence Indicator', template:'plotly_white', hovermode:'x unified', xaxis:tl, yaxis:{{title:'rad/s'}}}}, {{responsive:true}});
</script>
</body></html>
"""
    out_path.write_text(html, encoding="utf-8")


def filter_window(t: List[float], start: Optional[float], end: Optional[float]) -> List[int]:
    idx = []
    for i, ts in enumerate(t):
        if not is_finite(ts):
            continue
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        idx.append(i)
    return idx


def slice_by_idx(arr: List, idx: List[int]) -> List:
    return [arr[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser(description="Quaternion diagnostics for Gyroflow motion export")
    ap.add_argument("input", nargs="?", type=Path, default=DATA_DIR / "export.csv", help="CSV input (must include timestamp_ms and org_quat_w/x/y/z)")
    ap.add_argument("--start", type=float, default=None, help="Optional start time in seconds")
    ap.add_argument("--end", type=float, default=None, help="Optional end time in seconds")
    ap.add_argument("--outdir", type=Path, default=SCRIPT_DIR / "output", help="Output directory")
    ap.add_argument("--large-jump-dot", type=float, default=0.5, help="Threshold for large jump candidates")
    ap.add_argument("--dt-z", type=float, default=5.0, help="Outlier threshold in std-dev units")
    ap.add_argument("--lp2-hz", type=float, default=2.0, help="Low-pass cutoff for low-frequency curve")
    ap.add_argument("--lp350-hz", type=float, default=350.0, help="High-cutoff curve cutoff (use high value for near-unfiltered)")
    ap.add_argument("--filter-strength", type=float, default=1.0, help="LPF strength as pass count (rounded)")
    ap.add_argument("--top-n", type=int, default=20, help="Top residual peaks to report")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    data = load_csv(args.input)
    if "timestamp_ms" not in data:
        raise SystemExit("Input CSV must contain timestamp_ms")

    t_all = normalize_timestamps_ms(data["timestamp_ms"])
    q_all = compute_quat_series(data)

    keep_idx = filter_window(t_all, args.start, args.end)
    if len(keep_idx) < 3:
        raise SystemExit("Not enough samples in selected time window")

    t = slice_by_idx(t_all, keep_idx)
    q = slice_by_idx(q_all, keep_idx)
    t0 = t[0]
    t = [x - t0 for x in t]
    t_iso = [to_iso_time(x) for x in t]

    dots, dot_events = compute_dot_and_jump(t, q, args.large_jump_dot)
    sign_flip_events = [e for e in dot_events if e["event_type"] == "sign_flip_candidate"]
    large_jump_events = [e for e in dot_events if e["event_type"] == "large_jump_candidate"]

    dt_series, dt_stats, dt_events = compute_dt_stats(t, args.dt_z)

    _w_xyz, wmag = derive_gyro_from_quats(t, q)
    fs = estimate_sample_rate(t)
    if fs <= 0:
        raise SystemExit("Could not estimate sample rate from timestamps")

    w2 = lowpass_forward_backward(wmag, fs, args.lp2_hz, strength=args.filter_strength)
    w350 = lowpass_forward_backward(wmag, fs, args.lp350_hz, strength=args.filter_strength)

    residual = []
    for a, b in zip(w350, w2):
        if is_finite(a) and is_finite(b):
            residual.append(abs(a - b))
        else:
            residual.append(float("nan"))

    peaks = top_peaks(t, residual, args.top_n)
    peak_events = [
        {"event_type": "residual_peak", "index": i, "timestamp_s": ts, "value": v, "detail": f"top_{k+1}"}
        for k, (i, ts, v) in enumerate(peaks)
    ]

    all_events = sign_flip_events + large_jump_events + dt_events + peak_events
    all_events.sort(key=lambda e: (e.get("timestamp_s", float("inf")), e.get("event_type", "")))

    events_csv = args.outdir / "quaternion_diagnostics_events.csv"
    write_events_csv(events_csv, all_events)

    html_plot = args.outdir / "quaternion_diagnostics_plots.html"
    build_plot_html(
        out_path=html_plot,
        title=f"Quaternion Diagnostics: {args.input.name}",
        t_iso=t_iso,
        dot_series=dots,
        sign_flip_events=sign_flip_events,
        large_jump_events=large_jump_events,
        dt_series=dt_series,
        dt_events=dt_events,
        w350=w350,
        w2=w2,
        residual=residual,
        residual_peaks=peaks,
    )

    summary_txt = args.outdir / "quaternion_diagnostics_summary.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        f.write(f"Input: {args.input}\n")
        f.write(f"Samples: {len(t)}\n")
        f.write(f"Window: start={args.start} end={args.end}\n")
        f.write(f"Estimated sample rate: {fs:.6f} Hz\n")
        f.write(f"LP settings: lp2={args.lp2_hz}Hz lp350={args.lp350_hz}Hz strength={args.filter_strength}\n")
        f.write("\nQuaternion continuity:\n")
        f.write(f"  sign flip candidates (dot<0): {len(sign_flip_events)}\n")
        f.write(f"  large jump candidates (dot<{args.large_jump_dot}): {len(large_jump_events)}\n")
        f.write("\nTimestamp dt stats (seconds):\n")
        f.write(f"  min={dt_stats['min']:.9f} max={dt_stats['max']:.9f} mean={dt_stats['mean']:.9f} std={dt_stats['std']:.9f}\n")
        f.write(f"  high threshold (mean+{args.dt_z}*std)={dt_stats['high_threshold']:.9f}\n")
        f.write(f"  dt outliers/non-positive: {len(dt_events)}\n")
        f.write("\nTop residual peaks (|ω|_350 - |ω|_2, abs):\n")
        for i, ts, v in peaks:
            f.write(f"  idx={i:8d} t={ts:10.6f}s residual={v:.9f}\n")

    print(f"Input: {args.input}")
    print(f"Samples: {len(t)} | Estimated sample rate: {fs:.3f} Hz")
    print(f"Sign flip candidates (dot<0): {len(sign_flip_events)}")
    print(f"Large jump candidates (dot<{args.large_jump_dot}): {len(large_jump_events)}")
    print("Timestamp dt stats (s): "
          f"min={dt_stats['min']:.9f}, max={dt_stats['max']:.9f}, mean={dt_stats['mean']:.9f}, std={dt_stats['std']:.9f}")
    print(f"dt outliers/non-positive: {len(dt_events)}")
    print(f"Top residual peaks reported: {len(peaks)}")
    print(f"Wrote: {events_csv}")
    print(f"Wrote: {html_plot}")
    print(f"Wrote: {summary_txt}")


if __name__ == "__main__":
    main()
