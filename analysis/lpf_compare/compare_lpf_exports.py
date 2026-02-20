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


def diff(a: List[float], b: List[float]) -> List[float]:
    out: List[float] = []
    for x, y in zip(a, b):
        if is_finite(x) and is_finite(y):
            out.append(x - y)
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


def build_html(
    out_path: Path,
    title: str,
    source_a: str,
    source_b: str,
    x_full_iso: List[str],
    x_zoom_iso: List[str],
    payload: Dict[str, Dict[str, List[float] | str]],
    default_signal: str,
    focus_label: str,
) -> None:
    data_json = json.dumps(payload)
    x_full_json = json.dumps(x_full_iso)
    x_zoom_json = json.dumps(x_zoom_iso)

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
    #plotFull, #plotZoom {{ width: 100%; height: 500px; }}
    #plotZoom {{ height: 420px; }}
    .row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    label {{ font-weight: 600; }}
    select {{ padding: 5px 8px; border-radius: 6px; border: 1px solid #c7d2df; }}
    .meta {{ font-size: 12px; color: #4b5563; }}
  </style>
</head>
<body>
  <div class=\"panel\">
    <div class=\"row\">
      <label for=\"signalSelect\">Signal:</label>
      <select id=\"signalSelect\"></select>
      <div class=\"meta\">{source_a} vs {source_b}</div>
      <div class=\"meta\">Focus: {focus_label}</div>
    </div>
  </div>
  <div class=\"panel\"><div id=\"plotFull\"></div></div>
  <div class=\"panel\"><div id=\"plotZoom\"></div></div>

<script>
const SIGNALS = {data_json};
const X_FULL = {x_full_json};
const X_ZOOM = {x_zoom_json};
const SOURCE_A = {json.dumps(source_a)};
const SOURCE_B = {json.dumps(source_b)};

const select = document.getElementById('signalSelect');
const keys = Object.keys(SIGNALS);
for (const k of keys) {{
  const opt = document.createElement('option');
  opt.value = k;
  opt.textContent = SIGNALS[k].label;
  if (k === {json.dumps(default_signal)}) opt.selected = true;
  select.appendChild(opt);
}}

function makeTraces(key, zoom) {{
  const d = SIGNALS[key];
  const x = zoom ? X_ZOOM : X_FULL;
  const a = zoom ? d.a_zoom : d.a_full;
  const b = zoom ? d.b_zoom : d.b_full;
  const c = zoom ? d.diff_zoom : d.diff_full;
  return [
    {{ x, y: a, type: 'scatter', mode: 'lines', name: SOURCE_A, line: {{ width: 1.6, color: '#1f77b4' }} }},
    {{ x, y: b, type: 'scatter', mode: 'lines', name: SOURCE_B, line: {{ width: 1.6, color: '#2ca02c' }} }},
    {{ x, y: c, type: 'scatter', mode: 'lines', name: 'Difference (A-B)', line: {{ width: 1.4, color: '#e4572e' }} }}
  ];
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

function render() {{
  const key = select.value;
  const label = SIGNALS[key].label;

  Plotly.react('plotFull', makeTraces(key, false), baseLayout(label + ' (full timeline)', true), {{ responsive: true }});
  Plotly.react('plotZoom', makeTraces(key, true), baseLayout(label + ' (zoom around {focus_label})', false), {{ responsive: true }});
}}

select.addEventListener('change', render);
render();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare LPF exports and generate interactive HTML plots")
    p.add_argument("lp2_csv", type=Path, help="First CSV (e.g., LPF 2Hz double pass)")
    p.add_argument("lp350_csv", type=Path, help="Second CSV (e.g., LPF 350Hz double pass)")
    p.add_argument("--outdir", type=Path, default=Path("analysis/lpf_compare/output"), help="Dedicated output folder")
    p.add_argument("--focus", default="03:16", help="Interesting timestamp in MM:SS")
    p.add_argument("--focus-window", type=float, default=24.0, help="Zoom window width in seconds")
    p.add_argument("--max-full-points", type=int, default=120000, help="Maximum number of points in full-timeline plot")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    a = load_csv(args.lp2_csv)
    b = load_csv(args.lp350_csv)

    if "timestamp_ms" not in a or "timestamp_ms" not in b:
        raise SystemExit("Both CSV files must include timestamp_ms")

    ensure_gyro_mag(a)
    ensure_gyro_mag(b)

    t0_ms = min(min_finite(a["timestamp_ms"]), min_finite(b["timestamp_ms"]))
    ta = normalize_time(a["timestamp_ms"], t0_ms)
    tb = normalize_time(b["timestamp_ms"], t0_ms)

    common_t = ta if len(ta) >= len(tb) else tb

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
        if key not in a or key not in b:
            continue
        a_full = interp_to_time(ta, a[key], common_t)
        b_full = interp_to_time(tb, b[key], common_t)
        d_full = diff(a_full, b_full)

        payload[key] = {
            "label": label,
            "a_full": [a_full[i] for i in full_idx],
            "b_full": [b_full[i] for i in full_idx],
            "diff_full": [d_full[i] for i in full_idx],
            "a_zoom": [a_full[i] for i in focus_idx],
            "b_zoom": [b_full[i] for i in focus_idx],
            "diff_zoom": [d_full[i] for i in focus_idx],
        }

    if not payload:
        raise SystemExit("No comparable signals found in both files")

    default_signal = "org_gyro_mag" if "org_gyro_mag" in payload else next(iter(payload.keys()))

    source_a = clean_name(args.lp2_csv)
    source_b = clean_name(args.lp350_csv)
    out_html = args.outdir / "lpf_compare_interactive.html"

    build_html(
        out_path=out_html,
        title=f"LPF Comparison: {source_a} vs {source_b}",
        source_a=source_a,
        source_b=source_b,
        x_full_iso=x_full_iso,
        x_zoom_iso=x_zoom_iso,
        payload=payload,
        default_signal=default_signal,
        focus_label=f"{args.focus} (+/- {half:.1f}s)",
    )

    print(f"Wrote: {out_html}")


if __name__ == "__main__":
    main()
