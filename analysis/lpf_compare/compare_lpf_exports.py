#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Tuple


def parse_mmss(v: str) -> float:
    parts = v.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Expected MM:SS, got '{v}'")
    return int(parts[0]) * 60.0 + float(parts[1])


def format_mmss(seconds: float) -> str:
    s = max(0.0, seconds)
    m = int(s // 60)
    rem = s - m * 60
    return f"{m:02d}:{rem:04.1f}"


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


def sorted_unique_pairs(t: List[float], y: List[float]) -> Tuple[List[float], List[float]]:
    pairs = [(tt, yy) for tt, yy in zip(t, y) if is_finite(tt) and is_finite(yy)]
    if len(pairs) < 2:
        return [], []

    pairs.sort(key=lambda p: p[0])
    out_t: List[float] = []
    out_y: List[float] = []
    last_t = None
    for tt, yy in pairs:
        if last_t is not None and abs(tt - last_t) < 1e-9:
            out_y[-1] = yy
        else:
            out_t.append(tt)
            out_y.append(yy)
            last_t = tt
    return out_t, out_y


def interp_to_time(src_t: List[float], src_y: List[float], dst_t: List[float]) -> List[float]:
    t, y = sorted_unique_pairs(src_t, src_y)
    if len(t) < 2:
        return [float("nan")] * len(dst_t)

    out: List[float] = []
    for x in dst_t:
        if not is_finite(x) or x < t[0] or x > t[-1]:
            out.append(float("nan"))
            continue

        i = bisect_right(t, x)
        if i == 0:
            out.append(y[0])
            continue
        if i >= len(t):
            out.append(y[-1])
            continue

        t0, t1 = t[i - 1], t[i]
        y0, y1 = y[i - 1], y[i]
        if abs(t1 - t0) < 1e-12:
            out.append(y0)
            continue
        w = (x - t0) / (t1 - t0)
        out.append(y0 + (y1 - y0) * w)
    return out


def magnitude(x: List[float], y: List[float], z: List[float]) -> List[float]:
    out: List[float] = []
    for a, b, c in zip(x, y, z):
        if is_finite(a) and is_finite(b) and is_finite(c):
            out.append(math.sqrt(a * a + b * b + c * c))
        else:
            out.append(float("nan"))
    return out


def min_finite(values: List[float]) -> float:
    m = None
    for v in values:
        if is_finite(v):
            if m is None or v < m:
                m = v
    if m is None:
        raise ValueError("No finite timestamps found")
    return m


def max_finite(values: List[float]) -> float:
    m = None
    for v in values:
        if is_finite(v):
            if m is None or v > m:
                m = v
    if m is None:
        raise ValueError("No finite timestamps found")
    return m


def to_iso_time(t_sec: float) -> str:
    base = dt.datetime(1970, 1, 1)
    return (base + dt.timedelta(seconds=t_sec)).isoformat(timespec="milliseconds")


def normalize_time(ts_ms: List[float], t0_ms: float) -> List[float]:
    out: List[float] = []
    for v in ts_ms:
        if is_finite(v):
            out.append((v - t0_ms) / 1000.0)
        else:
            out.append(float("nan"))
    return out


def clean_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def ensure_gyro_mag(data: Dict[str, List[float]]) -> None:
    keys = ("org_gyro_x", "org_gyro_y", "org_gyro_z")
    if all(k in data for k in keys):
        data["org_gyro_mag"] = magnitude(data["org_gyro_x"], data["org_gyro_y"], data["org_gyro_z"])


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def estimate_sample_rate(t_sec: List[float]) -> float:
    valid = [v for v in t_sec if is_finite(v)]
    if len(valid) < 2:
        return 0.0
    valid.sort()
    diffs: List[float] = []
    prev = valid[0]
    for cur in valid[1:]:
        d = cur - prev
        if d > 0:
            diffs.append(d)
        prev = cur
    if not diffs:
        return 0.0
    md = median(diffs)
    return 1.0 / md if md > 0 else 0.0


def sanitize_signal(x: List[float]) -> List[float]:
    valid = [v for v in x if is_finite(v)]
    fill = median(valid) if valid else 0.0
    y = [v if is_finite(v) else fill for v in x]
    mean = sum(y) / len(y) if y else 0.0
    return [v - mean for v in y]


def resample_uniform(t_sec: List[float], y: List[float], fs_target: float, max_points: int = 250000) -> Tuple[List[float], List[float], float]:
    ts, ys = sorted_unique_pairs(t_sec, y)
    if len(ts) < 2 or fs_target <= 0:
        return [], [], 0.0

    start = ts[0]
    end = ts[-1]
    dur = end - start
    if dur <= 0:
        return [], [], 0.0

    n = int(dur * fs_target) + 1
    if max_points > 0 and n > max_points:
        n = max_points
        fs_target = (n - 1) / dur if dur > 0 else fs_target

    if n < 8:
        return [], [], 0.0

    dt_s = 1.0 / fs_target
    t_uniform = [start + i * dt_s for i in range(n)]
    y_uniform = interp_to_time(ts, ys, t_uniform)
    y_uniform = sanitize_signal(y_uniform)
    return t_uniform, y_uniform, fs_target


def build_uniform_grid(start: float, end: float, fs: float, max_points: int = 250000) -> Tuple[List[float], float]:
    dur = end - start
    if fs <= 0 or dur <= 0:
        return [], 0.0

    n = int(dur * fs) + 1
    if max_points > 0 and n > max_points:
        n = max_points
        fs = (n - 1) / dur if dur > 0 else fs
    if n < 8:
        return [], 0.0

    dt_s = 1.0 / fs
    return [start + i * dt_s for i in range(n)], fs


def next_pow2(v: int) -> int:
    n = 1
    while n < v:
        n <<= 1
    return n


def fft_inplace(a: List[complex]) -> None:
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]

    length = 2
    while length <= n:
        ang = -2.0 * math.pi / length
        wlen = complex(math.cos(ang), math.sin(ang))
        half = length >> 1
        for i in range(0, n, length):
            w = 1 + 0j
            for j in range(i, i + half):
                u = a[j]
                v = a[j + half] * w
                a[j] = u + v
                a[j + half] = u - v
                w *= wlen
        length <<= 1


def rfft_magnitude_db(segment: List[float], n_fft: int) -> List[float]:
    if len(segment) < n_fft:
        seg = segment + [0.0] * (n_fft - len(segment))
    else:
        seg = segment[:n_fft]

    windowed: List[complex] = []
    for i, v in enumerate(seg):
        w = 0.5 - 0.5 * math.cos((2.0 * math.pi * i) / max(1, n_fft - 1))
        windowed.append(complex(v * w, 0.0))

    fft_inplace(windowed)

    out: List[float] = []
    norm = max(1.0, n_fft / 2.0)
    for c in windowed[: (n_fft // 2) + 1]:
        mag = abs(c) / norm
        out.append(20.0 * math.log10(max(mag, 1e-12)))
    return out


def welch_db(y: List[float], fs: float, n_fft: int) -> Tuple[List[float], List[float]]:
    if len(y) < n_fft or fs <= 0:
        return [], []

    hop = max(1, n_fft // 2)
    acc: List[float] = [0.0] * ((n_fft // 2) + 1)
    count = 0

    for start in range(0, len(y) - n_fft + 1, hop):
        db = rfft_magnitude_db(y[start : start + n_fft], n_fft)
        lin = [10.0 ** (v / 20.0) for v in db]
        for i, v in enumerate(lin):
            acc[i] += v
        count += 1

    if count == 0:
        return [], []

    avg_db = [20.0 * math.log10(max(v / count, 1e-12)) for v in acc]
    freqs = [(i * fs) / n_fft for i in range(len(avg_db))]
    return freqs, avg_db


def short_time_spectra_db(y: List[float], fs: float, n_fft: int, window_sec: float, step_sec: float) -> Tuple[List[float], List[List[float]]]:
    if fs <= 0 or len(y) < n_fft:
        return [], []

    win_len = max(n_fft, int(window_sec * fs))
    win_len = next_pow2(win_len)
    win_len = min(win_len, len(y))
    if win_len < 16:
        return [], []

    step_n = max(1, int(step_sec * fs))
    half = win_len // 2

    centers_sec: List[float] = []
    spectra: List[List[float]] = []

    start_center = half
    end_center = len(y) - half

    for c in range(start_center, end_center, step_n):
        seg = y[c - half : c + half]
        if len(seg) < n_fft:
            seg = seg + [0.0] * (n_fft - len(seg))
        db = rfft_magnitude_db(seg, n_fft)
        centers_sec.append(c / fs)
        spectra.append(db)

    return centers_sec, spectra


def clamp_freq(freq: List[float], vals: List[float], max_hz: float) -> Tuple[List[float], List[float]]:
    out_f: List[float] = []
    out_v: List[float] = []
    for f, v in zip(freq, vals):
        if f <= max_hz:
            out_f.append(f)
            out_v.append(v)
    return out_f, out_v


def clamp_freq_frames(freq: List[float], frames: List[List[float]], max_hz: float) -> Tuple[List[float], List[List[float]]]:
    idx = [i for i, f in enumerate(freq) if f <= max_hz]
    out_f = [freq[i] for i in idx]
    out_frames = [[frame[i] for i in idx] for frame in frames]
    return out_f, out_frames


def build_html(
    out_path: Path,
    title: str,
    source_names: List[str],
    x_full_iso: List[str],
    x_zoom_iso: List[str],
    payload: Dict[str, Dict[str, List[float] | str]],
    default_signal: str,
    focus_label: str,
    vib_freq: List[float],
    vib_overall: Dict[str, List[float]],
    vib_local_frames: Dict[str, List[List[float]]],
    vib_frame_times: List[float],
) -> None:
    data_json = json.dumps(payload)
    x_full_json = json.dumps(x_full_iso)
    x_zoom_json = json.dumps(x_zoom_iso)

    vib_freq_json = json.dumps(vib_freq)
    vib_overall_json = json.dumps(vib_overall)
    vib_local_json = json.dumps(vib_local_frames)
    vib_frame_times_json = json.dumps(vib_frame_times)

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>{title}</title>
  <script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; background: #f6f8fb; color: #1f2937; }}
    .panel {{ background: #fff; border: 1px solid #dbe2ea; border-radius: 10px; padding: 12px; margin-bottom: 14px; }}
    #plotFull, #plotZoom, #plotVibOverall, #plotVibLocal {{ width: 100%; }}
    #plotFull {{ height: 460px; }}
    #plotZoom {{ height: 360px; }}
    #plotVibOverall, #plotVibLocal {{ height: 420px; }}
    .row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    label {{ font-weight: 600; }}
    select {{ padding: 5px 8px; border-radius: 6px; border: 1px solid #c7d2df; }}
    .meta {{ font-size: 12px; color: #4b5563; }}
    input[type=range] {{ min-width: 320px; }}
  </style>
</head>
<body>
  <div class=\"panel\">
    <div class=\"row\">
      <label for=\"signalSelect\">Time-domain signal:</label>
      <select id=\"signalSelect\"></select>
      <div class=\"meta\">{source_names[0]} / {source_names[1]} / {source_names[2]}</div>
      <div class=\"meta\">Focus: {focus_label}</div>
    </div>
  </div>
  <div class=\"panel\"><div id=\"plotFull\"></div></div>
  <div class=\"panel\"><div id=\"plotZoom\"></div></div>

  <div class=\"panel\">
    <div class=\"row\">
      <strong>Vibration Spectrum (whole range)</strong>
      <div class=\"meta\">Unit: dB</div>
    </div>
    <div id=\"plotVibOverall\"></div>
  </div>

  <div class=\"panel\">
    <div class=\"row\">
      <strong>Vibration Spectrum (local time window)</strong>
      <label for=\"timeSlider\">Time:</label>
      <input id=\"timeSlider\" type=\"range\" min=\"0\" max=\"0\" step=\"1\" value=\"0\" />
      <span id=\"timeLabel\" class=\"meta\">00:00.0</span>
      <div class=\"meta\">Unit: dB</div>
    </div>
    <div id=\"plotVibLocal\"></div>
  </div>

<script>
const SIGNALS = {data_json};
const X_FULL = {x_full_json};
const X_ZOOM = {x_zoom_json};
const SOURCES = {json.dumps(source_names)};

const VIB_FREQ = {vib_freq_json};
const VIB_OVERALL = {vib_overall_json};
const VIB_LOCAL = {vib_local_json};
const VIB_TIMES = {vib_frame_times_json};

const sourceColors = ['#1f77b4', '#2ca02c', '#e4572e'];

const select = document.getElementById('signalSelect');
const keys = Object.keys(SIGNALS);
for (const k of keys) {{
  const opt = document.createElement('option');
  opt.value = k;
  opt.textContent = SIGNALS[k].label;
  if (k === {json.dumps(default_signal)}) opt.selected = true;
  select.appendChild(opt);
}}

function mmss(s) {{
  const sec = Math.max(0, +s || 0);
  const m = Math.floor(sec / 60);
  const r = sec - m * 60;
  return String(m).padStart(2, '0') + ':' + r.toFixed(1).padStart(4, '0');
}}

function makeTraces(key, zoom) {{
  const d = SIGNALS[key];
  const x = zoom ? X_ZOOM : X_FULL;
  const traces = [];
  for (let i = 0; i < SOURCES.length; i++) {{
    const y = zoom ? d.sources[i].zoom : d.sources[i].full;
    traces.push({{ x, y, type: 'scatter', mode: 'lines', name: SOURCES[i], line: {{ width: 1.4, color: sourceColors[i % sourceColors.length] }} }});
  }}
  return traces;
}}

function baseLayout(title, showRangeSlider) {{
  return {{
    title,
    template: 'plotly_white',
    hovermode: 'x unified',
    legend: {{ orientation: 'h', x: 0, y: 1.15 }},
    margin: {{ t: 56, r: 20, b: 50, l: 60 }},
    xaxis: {{
      type: 'date',
      tickformat: '%M:%S',
      hoverformat: '%M:%S.%L',
      rangeslider: {{ visible: showRangeSlider }}
    }},
    yaxis: {{ title: 'Value' }}
  }};
}}

function renderTime() {{
  const key = select.value;
  const label = SIGNALS[key].label;
  Plotly.react('plotFull', makeTraces(key, false), baseLayout(label + ' (full timeline)', true), {{ responsive: true }});
  Plotly.react('plotZoom', makeTraces(key, true), baseLayout(label + ' (zoom around {focus_label})', false), {{ responsive: true }});
}}

function renderVibrationOverall() {{
  const traces = SOURCES.map((name, i) => {{
    return {{
      x: VIB_FREQ,
      y: VIB_OVERALL[name],
      type: 'scatter',
      mode: 'lines',
      name,
      line: {{ width: 1.6, color: sourceColors[i % sourceColors.length] }}
    }};
  }});
  Plotly.react('plotVibOverall', traces, {{
    title: 'Vibration Spectrum (overall)',
    template: 'plotly_white',
    hovermode: 'x unified',
    legend: {{ orientation: 'h', x: 0, y: 1.15 }},
    margin: {{ t: 56, r: 20, b: 50, l: 60 }},
    xaxis: {{ title: 'Frequency (Hz)' }},
    yaxis: {{ title: 'Level (dB)' }}
  }}, {{ responsive: true }});
}}

const slider = document.getElementById('timeSlider');
const timeLabel = document.getElementById('timeLabel');

function renderVibrationLocal() {{
  const idx = Math.max(0, Math.min(VIB_TIMES.length - 1, +slider.value || 0));
  const traces = SOURCES.map((name, i) => {{
    return {{
      x: VIB_FREQ,
      y: (VIB_LOCAL[name] && VIB_LOCAL[name][idx]) ? VIB_LOCAL[name][idx] : [],
      type: 'scatter',
      mode: 'lines',
      name,
      line: {{ width: 1.6, color: sourceColors[i % sourceColors.length] }}
    }};
  }});

  timeLabel.textContent = mmss(VIB_TIMES[idx] || 0);
  Plotly.react('plotVibLocal', traces, {{
    title: 'Vibration Spectrum (local around selected time)',
    template: 'plotly_white',
    hovermode: 'x unified',
    legend: {{ orientation: 'h', x: 0, y: 1.15 }},
    margin: {{ t: 56, r: 20, b: 50, l: 60 }},
    xaxis: {{ title: 'Frequency (Hz)' }},
    yaxis: {{ title: 'Level (dB)' }}
  }}, {{ responsive: true }});
}}

select.addEventListener('change', renderTime);
slider.addEventListener('input', renderVibrationLocal);

if (VIB_TIMES.length > 0) {{
  slider.min = '0';
  slider.max = String(VIB_TIMES.length - 1);
  slider.value = String(Math.floor((VIB_TIMES.length - 1) / 2));
}}

renderTime();
renderVibrationOverall();
renderVibrationLocal();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare Original/LP350/LP2 exports and generate interactive HTML plots")
    p.add_argument("original_csv", type=Path, help="Original CSV")
    p.add_argument("lp350_csv", type=Path, help="LP350 CSV")
    p.add_argument("lp2_csv", type=Path, help="LP2 CSV")
    p.add_argument("--outdir", type=Path, default=Path("analysis/lpf_compare/output"), help="Dedicated output folder")
    p.add_argument("--focus", default="03:16", help="Interesting timestamp in MM:SS")
    p.add_argument("--focus-window", type=float, default=24.0, help="Zoom window width in seconds")
    p.add_argument("--max-full-points", type=int, default=120000, help="Maximum number of points in full timeline")
    p.add_argument("--vib-max-hz", type=float, default=0.0, help="Maximum frequency shown in vibration plots; <=0 means auto (near Nyquist)")
    p.add_argument("--vib-nfft", type=int, default=2048, help="FFT size (power of 2 recommended)")
    p.add_argument("--vib-window-sec", type=float, default=4.0, help="Local spectrum window size in seconds")
    p.add_argument("--vib-step-sec", type=float, default=1.0, help="Step between local spectrum frames in seconds")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    datasets: List[Tuple[str, Path, Dict[str, List[float]]]] = [
        ("Original", args.original_csv, load_csv(args.original_csv)),
        ("LP350", args.lp350_csv, load_csv(args.lp350_csv)),
        ("LP2", args.lp2_csv, load_csv(args.lp2_csv)),
    ]

    for _, _, data in datasets:
        if "timestamp_ms" not in data:
            raise SystemExit("All CSV files must include timestamp_ms")
        ensure_gyro_mag(data)

    t0_ms = min(min_finite(data["timestamp_ms"]) for _, _, data in datasets)
    t_norm_by_name: Dict[str, List[float]] = {}
    for name, _, data in datasets:
        t_norm_by_name[name] = normalize_time(data["timestamp_ms"], t0_ms)

    common_t = max(t_norm_by_name.values(), key=len)

    focus_center = parse_mmss(args.focus)
    half = max(args.focus_window * 0.5, 0.5)
    focus_start = max(0.0, focus_center - half)
    focus_end = focus_center + half

    focus_idx = [i for i, t in enumerate(common_t) if is_finite(t) and focus_start <= t <= focus_end]
    if not focus_idx:
        raise SystemExit("Focus window has no samples. Try a different --focus or --focus-window")

    if args.max_full_points > 0 and len(common_t) > args.max_full_points:
        step = max(1, math.ceil(len(common_t) / args.max_full_points))
        full_idx = list(range(0, len(common_t), step))
    else:
        full_idx = list(range(len(common_t)))

    x_full_iso = [to_iso_time(common_t[i] if is_finite(common_t[i]) else 0.0) for i in full_idx]
    x_zoom_iso = [to_iso_time(common_t[i]) for i in focus_idx]

    candidates = [
        ("org_gyro_x", "Gyro X"),
        ("org_gyro_y", "Gyro Y"),
        ("org_gyro_z", "Gyro Z"),
        ("org_gyro_mag", "Gyro Magnitude"),
        ("stab_pitch", "Stabilized Pitch"),
        ("stab_yaw", "Stabilized Yaw"),
        ("stab_roll", "Stabilized Roll"),
    ]

    payload: Dict[str, Dict[str, List[float] | str]] = {}
    for key, label in candidates:
        if not all(key in data for _, _, data in datasets):
            continue

        src_payload: List[Dict[str, List[float]]] = []
        for name, _, data in datasets:
            interp = interp_to_time(t_norm_by_name[name], data[key], common_t)
            src_payload.append({
                "full": [interp[i] for i in full_idx],
                "zoom": [interp[i] for i in focus_idx],
            })

        payload[key] = {
            "label": label,
            "sources": src_payload,
        }

    if not payload:
        raise SystemExit("No comparable signals found in all three files")

    default_signal = "org_gyro_mag" if "org_gyro_mag" in payload else next(iter(payload.keys()))

    # Vibration analysis for all three datasets (gyro magnitude)
    if not all("org_gyro_mag" in data for _, _, data in datasets):
        raise SystemExit("Vibration analysis requires org_gyro_x/y/z columns in all three files")

    vib_overall: Dict[str, List[float]] = {}
    vib_local: Dict[str, List[List[float]]] = {}
    vib_freq_ref: List[float] = []
    vib_times_ref: List[float] = []

    fs_candidates = [estimate_sample_rate(t_norm_by_name[name]) for name, _, _ in datasets]
    if any(fs <= 0 for fs in fs_candidates):
        raise SystemExit("Could not estimate sample rate for all datasets")
    fs_common = min(fs_candidates)

    t_start = max(min_finite(t_norm_by_name[name]) for name, _, _ in datasets)
    t_end = min(max_finite(t_norm_by_name[name]) for name, _, _ in datasets)
    t_uniform, fs_common = build_uniform_grid(t_start, t_end, fs_common, max_points=0)
    if not t_uniform:
        raise SystemExit("Could not build common uniform timeline for vibration analysis")

    y_uniform_by_name: Dict[str, List[float]] = {}
    for name, _, data in datasets:
        y_interp = interp_to_time(t_norm_by_name[name], data["org_gyro_mag"], t_uniform)
        y_uniform_by_name[name] = sanitize_signal(y_interp)

    n_fft = next_pow2(max(256, args.vib_nfft))
    while n_fft > len(t_uniform) and n_fft > 256:
        n_fft >>= 1
    n_fft = max(256, n_fft)

    nyquist = fs_common * 0.5
    vib_max_hz = (nyquist * 0.98) if args.vib_max_hz <= 0 else min(args.vib_max_hz, nyquist * 0.98)

    for idx_ds, (name, _, _data) in enumerate(datasets):
        yu = y_uniform_by_name[name]

        f_all, db_all = welch_db(yu, fs_common, n_fft)
        if not f_all:
            raise SystemExit(f"Could not compute overall spectrum for {name}")
        f_all, db_all = clamp_freq(f_all, db_all, vib_max_hz)

        frame_times, frame_db = short_time_spectra_db(yu, fs_common, n_fft, args.vib_window_sec, args.vib_step_sec)
        if not frame_times or not frame_db:
            raise SystemExit(f"Could not compute local spectra for {name}")
        freq_bins = [(i * fs_common) / n_fft for i in range((n_fft // 2) + 1)]
        freq_bins, frame_db = clamp_freq_frames(freq_bins, frame_db, vib_max_hz)

        if idx_ds == 0:
            vib_freq_ref = f_all
            vib_times_ref = frame_times
            vib_overall[name] = db_all
            vib_local[name] = frame_db
        else:
            n_time = min(len(vib_times_ref), len(frame_times))
            n_freq = min(len(vib_freq_ref), len(f_all), len(freq_bins))
            vib_times_ref = vib_times_ref[:n_time]
            vib_freq_ref = vib_freq_ref[:n_freq]
            vib_overall = {k: v[:n_freq] for k, v in vib_overall.items()}
            vib_local = {k: [row[:n_freq] for row in vals[:n_time]] for k, vals in vib_local.items()}
            vib_overall[name] = db_all[:n_freq]
            vib_local[name] = [row[:n_freq] for row in frame_db[:n_time]]

    out_html = args.outdir / "lpf_compare_interactive.html"

    build_html(
        out_path=out_html,
        title=f"LPF Comparison: {clean_name(args.original_csv)} vs {clean_name(args.lp350_csv)} vs {clean_name(args.lp2_csv)}",
        source_names=[name for name, _, _ in datasets],
        x_full_iso=x_full_iso,
        x_zoom_iso=x_zoom_iso,
        payload=payload,
        default_signal=default_signal,
        focus_label=f"{args.focus} (+/- {half:.1f}s)",
        vib_freq=vib_freq_ref,
        vib_overall=vib_overall,
        vib_local_frames=vib_local,
        vib_frame_times=vib_times_ref,
    )

    print(f"Wrote: {out_html}")
    print(f"Vibration frequency range: 0..{vib_max_hz:.1f} Hz (Nyquist: {nyquist:.1f} Hz)")
    print(f"Local vibration frame count: {len(vib_times_ref)}")


if __name__ == "__main__":
    main()
