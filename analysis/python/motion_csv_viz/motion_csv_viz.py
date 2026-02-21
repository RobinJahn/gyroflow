#!/usr/bin/env python3
"""
Interactive visualization for a single Gyroflow motion CSV export.

Outputs a self-contained HTML report (Plotly loaded from CDN).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = SCRIPT_DIR.parent
DATA_DIR = PYTHON_ROOT / "data"


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


def min_finite(values: List[float]) -> float:
    m = None
    for v in values:
        if is_finite(v):
            if m is None or v < m:
                m = v
    if m is None:
        raise ValueError("No finite values found")
    return m


def normalize_time(ts_ms: List[float], t0_ms: float) -> List[float]:
    out: List[float] = []
    for v in ts_ms:
        if is_finite(v):
            out.append((v - t0_ms) / 1000.0)
        else:
            out.append(float("nan"))
    return out


def magnitude(x: List[float], y: List[float], z: List[float]) -> List[float]:
    out: List[float] = []
    for a, b, c in zip(x, y, z):
        if is_finite(a) and is_finite(b) and is_finite(c):
            out.append(math.sqrt(a * a + b * b + c * c))
        else:
            out.append(float("nan"))
    return out


def to_iso_time(t_sec: float) -> str:
    base = dt.datetime(1970, 1, 1)
    return (base + dt.timedelta(seconds=t_sec)).isoformat(timespec="milliseconds")


def finite_mean(values: List[float]) -> float:
    s = 0.0
    n = 0
    for v in values:
        if is_finite(v):
            s += v
            n += 1
    return s / n if n else float("nan")


def estimate_sample_rate(t_ms: List[float]) -> float:
    valid = [v for v in t_ms if is_finite(v)]
    if len(valid) < 2:
        return 0.0
    valid.sort()
    diffs = []
    prev = valid[0]
    for cur in valid[1:]:
        d = cur - prev
        if d > 0:
            diffs.append(d)
        prev = cur
    if not diffs:
        return 0.0
    diffs.sort()
    med = diffs[len(diffs) // 2]
    return 1000.0 / med if med > 0 else 0.0


def downsample_indices(n: int, max_points: int) -> List[int]:
    if max_points <= 0 or n <= max_points:
        return list(range(n))
    step = max(1, math.ceil(n / max_points))
    return list(range(0, n, step))


def build_html(
    out_path: Path,
    title: str,
    source_name: str,
    x_iso: List[str],
    payload: Dict[str, Dict[str, List[float] | str]],
    summary: Dict[str, str],
    default_signal: str,
) -> None:
    data_json = json.dumps(payload)
    x_json = json.dumps(x_iso)
    summary_json = json.dumps(summary)

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
    #plot {{ width: 100%; height: 620px; }}
    .row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
    .meta {{ font-size: 12px; color: #4b5563; }}
    select {{ padding: 5px 8px; border-radius: 6px; border: 1px solid #c7d2df; }}
  </style>
</head>
<body>
  <div class=\"panel\">
    <div class=\"row\">
      <strong>{source_name}</strong>
      <label for=\"signalSelect\">Signal group:</label>
      <select id=\"signalSelect\"></select>
      <span class=\"meta\" id=\"summary\"></span>
    </div>
  </div>
  <div class=\"panel\"><div id=\"plot\"></div></div>

<script>
const GROUPS = {data_json};
const X = {x_json};
const SUMMARY = {summary_json};
const select = document.getElementById('signalSelect');
const summary = document.getElementById('summary');

for (const k of Object.keys(GROUPS)) {{
  const opt = document.createElement('option');
  opt.value = k;
  opt.textContent = GROUPS[k].label;
  if (k === {json.dumps(default_signal)}) opt.selected = true;
  select.appendChild(opt);
}}

function render() {{
  const key = select.value;
  const g = GROUPS[key];
  const traces = [];
  for (const s of g.series) {{
    traces.push({{
      x: X,
      y: s.values,
      type: 'scatter',
      mode: 'lines',
      name: s.name,
      line: {{ width: 1.4 }}
    }});
  }}

  const layout = {{
    title: g.label,
    template: 'plotly_white',
    hovermode: 'x unified',
    legend: {{ orientation: 'h', x: 0, y: 1.15 }},
    margin: {{ t: 58, r: 20, b: 50, l: 60 }},
    xaxis: {{
      type: 'date',
      tickformat: '%M:%S',
      hoverformat: '%M:%S.%L',
      rangeslider: {{ visible: true }}
    }},
    yaxis: {{ title: g.y_label }}
  }};

  summary.textContent = SUMMARY[key] || '';
  Plotly.react('plot', traces, layout, {{ responsive: true }});
}}

select.addEventListener('change', render);
render();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Interactive visualization for a single Gyroflow motion CSV export")
    p.add_argument("csv_file", nargs="?", type=Path, default=DATA_DIR / "export.csv", help="Path to exported gyro CSV")
    p.add_argument("--out", type=Path, default=None, help="Output HTML path (default: analysis/python/motion_csv_viz/output/<csv_stem>_viz_interactive.html)")
    p.add_argument("--max-points", type=int, default=120000, help="Max points per plotted series")
    args = p.parse_args()

    data = load_csv(args.csv_file)
    if "timestamp_ms" not in data:
        raise SystemExit("CSV must contain timestamp_ms")

    t0_ms = min_finite(data["timestamp_ms"])
    t_s = normalize_time(data["timestamp_ms"], t0_ms)

    idx = downsample_indices(len(t_s), args.max_points)
    x_iso = [to_iso_time(t_s[i] if is_finite(t_s[i]) else 0.0) for i in idx]

    payload: Dict[str, Dict[str, List[float] | str]] = {}
    summary: Dict[str, str] = {}

    fs = estimate_sample_rate(data["timestamp_ms"])

    if all(k in data for k in ("org_gyro_x", "org_gyro_y", "org_gyro_z")):
        gx = data["org_gyro_x"]
        gy = data["org_gyro_y"]
        gz = data["org_gyro_z"]
        gm = magnitude(gx, gy, gz)
        payload["gyro"] = {
            "label": "Original Gyroscope",
            "y_label": "Angular rate",
            "series": [
                {"name": "gyro x", "values": [gx[i] for i in idx]},
                {"name": "gyro y", "values": [gy[i] for i in idx]},
                {"name": "gyro z", "values": [gz[i] for i in idx]},
                {"name": "gyro |mag|", "values": [gm[i] for i in idx]},
            ],
        }
        summary["gyro"] = f"Estimated sample rate: {fs:.2f} Hz | Mean |gyro|: {finite_mean(gm):.6f}"

    if all(k in data for k in ("org_acc_x", "org_acc_y", "org_acc_z")):
        ax = data["org_acc_x"]
        ay = data["org_acc_y"]
        az = data["org_acc_z"]
        am = magnitude(ax, ay, az)
        payload["acc"] = {
            "label": "Original Accelerometer",
            "y_label": "Acceleration",
            "series": [
                {"name": "acc x", "values": [ax[i] for i in idx]},
                {"name": "acc y", "values": [ay[i] for i in idx]},
                {"name": "acc z", "values": [az[i] for i in idx]},
                {"name": "acc |mag|", "values": [am[i] for i in idx]},
            ],
        }
        summary["acc"] = f"Estimated sample rate: {fs:.2f} Hz | Mean |acc|: {finite_mean(am):.6f}"

    if all(k in data for k in ("stab_pitch", "stab_yaw", "stab_roll")):
        sp = data["stab_pitch"]
        sy = data["stab_yaw"]
        sr = data["stab_roll"]
        payload["stab"] = {
            "label": "Stabilized Motion (Euler)",
            "y_label": "Degrees",
            "series": [
                {"name": "stab pitch", "values": [sp[i] for i in idx]},
                {"name": "stab yaw", "values": [sy[i] for i in idx]},
                {"name": "stab roll", "values": [sr[i] for i in idx]},
            ],
        }
        summary["stab"] = f"Estimated sample rate: {fs:.2f} Hz"

    if not payload:
        raise SystemExit("CSV does not contain expected columns (gyro/acc/stab)")

    default_signal = "gyro" if "gyro" in payload else next(iter(payload.keys()))

    if args.out is None:
        out = SCRIPT_DIR / "output" / f"{args.csv_file.stem}_viz_interactive.html"
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    build_html(
        out_path=out,
        title=f"Motion CSV Interactive View: {args.csv_file.stem}",
        source_name=args.csv_file.name,
        x_iso=x_iso,
        payload=payload,
        summary=summary,
        default_signal=default_signal,
    )

    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
