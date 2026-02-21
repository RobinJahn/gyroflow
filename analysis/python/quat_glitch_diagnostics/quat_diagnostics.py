#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


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


def to_iso_time(t_sec: np.ndarray) -> List[str]:
    base = dt.datetime(1970, 1, 1)
    out: List[str] = []
    for x in t_sec:
        if np.isfinite(x):
            out.append((base + dt.timedelta(seconds=float(x))).isoformat(timespec="milliseconds"))
        else:
            out.append(base.isoformat(timespec="milliseconds"))
    return out


def load_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")

        cols: Dict[str, List[float]] = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k in cols:
                cols[k].append(parse_float(row.get(k, "")))

    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def detect_timestamp_col(cols: List[str]) -> Optional[str]:
    lower = [c.lower() for c in cols]
    for preferred in ["timestamp_ms", "timestamp", "time_ms", "time"]:
        if preferred in lower:
            return cols[lower.index(preferred)]
    for i, c in enumerate(lower):
        if "timestamp" in c or c.endswith("_ms") or c == "t":
            return cols[i]
    return None


def detect_quat_cols(cols: List[str]) -> Optional[Tuple[str, str, str, str]]:
    pattern = re.compile(r"^(.*?)(?:[._-]?)([wxyz])$", re.IGNORECASE)

    groups: Dict[str, Dict[str, str]] = {}
    for col in cols:
        lc = col.lower()
        if "quat" not in lc:
            continue
        m = pattern.match(lc)
        if not m:
            continue
        stem, comp = m.group(1), m.group(2).lower()
        groups.setdefault(stem, {})[comp] = col

    candidates = []
    for stem, g in groups.items():
        if all(c in g for c in "wxyz"):
            score = 0
            if stem.startswith("org_quat"):
                score += 10
            if stem.endswith("org_quat"):
                score += 5
            if stem == "org_quat":
                score += 20
            candidates.append((score, stem, g))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        g = candidates[0][2]
        return (g["w"], g["x"], g["y"], g["z"])

    # final fallback exact conventional names
    lower = {c.lower(): c for c in cols}
    keys = ["org_quat_w", "org_quat_x", "org_quat_y", "org_quat_z"]
    if all(k in lower for k in keys):
        return (lower[keys[0]], lower[keys[1]], lower[keys[2]], lower[keys[3]])
    return None


def normalize_time(t_raw: np.ndarray, col_name: str) -> np.ndarray:
    t = np.asarray(t_raw, dtype=float).copy()
    finite = np.isfinite(t)
    if not finite.any():
        return np.full_like(t, np.nan)

    t0 = np.nanmin(t)
    t = t - t0

    lc = col_name.lower()
    if "ms" in lc:
        t = t / 1000.0
    else:
        # heuristic: if median positive delta > 1, likely ms
        d = np.diff(t)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size > 0 and float(np.median(d)) > 1.0:
            t = t / 1000.0

    return t


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=1)
    out = q.copy()
    valid = np.isfinite(n) & (n > 0)
    out[valid] = out[valid] / n[valid, None]
    out[~valid] = np.nan
    return out


def quat_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[:, 1:] *= -1.0
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.column_stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_scaled_axis(q: np.ndarray) -> np.ndarray:
    qn = quat_normalize(q)
    w = np.clip(qn[:, 0], -1.0, 1.0)
    v = qn[:, 1:]
    vnorm = np.linalg.norm(v, axis=1)
    angle = 2.0 * np.arctan2(vnorm, w)

    out = np.zeros_like(v)
    valid = np.isfinite(vnorm) & (vnorm > 1e-12) & np.isfinite(angle)
    out[:] = np.nan
    out[valid] = v[valid] * (angle[valid] / vnorm[valid])[:, None]
    zeroish = np.isfinite(vnorm) & (vnorm <= 1e-12)
    out[zeroish] = 0.0
    return out


def derive_omega_mag(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    n = len(q)
    out = np.full(n, np.nan, dtype=float)
    if n < 2:
        return out

    prev = q[:-1]
    cur = q[1:]
    valid_q = np.all(np.isfinite(prev), axis=1) & np.all(np.isfinite(cur), axis=1)

    dt = t[1:] - t[:-1]
    valid_dt = np.isfinite(dt) & (dt > 0)
    valid = valid_q & valid_dt

    if not np.any(valid):
        return out

    delta = quat_mul(quat_conj(prev[valid]), cur[valid])
    rot = quat_scaled_axis(delta)
    omega = rot / dt[valid, None]
    mag = np.linalg.norm(omega, axis=1)

    out_idx = np.where(valid)[0] + 1
    out[out_idx] = mag
    return out


def enforce_hemisphere_continuity(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for i in range(1, len(out)):
        if not (np.all(np.isfinite(out[i - 1])) and np.all(np.isfinite(out[i]))):
            continue
        if float(np.dot(out[i - 1], out[i])) < 0.0:
            out[i] = -out[i]
    return out


def find_dt_outliers(dt: np.ndarray, factor: float) -> Tuple[np.ndarray, float]:
    pos = dt[np.isfinite(dt) & (dt > 0)]
    if pos.size == 0:
        return np.zeros_like(dt, dtype=bool), float("nan")
    med = float(np.median(pos))
    mask = (np.isfinite(dt) & (dt <= 0)) | (np.isfinite(dt) & (dt > factor * med))
    return mask, med


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def max_abs_dev(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) == 0:
        return float("nan")
    return float(np.max(np.abs(a[m] - b[m])))


def top_k(values: np.ndarray, t: np.ndarray, k: int) -> List[Tuple[int, float, float]]:
    idx = np.where(np.isfinite(values))[0]
    if idx.size == 0:
        return []
    order = idx[np.argsort(values[idx])[::-1]]
    out = []
    for i in order[: max(0, k)]:
        out.append((int(i), float(t[i]), float(values[i])))
    return out


def build_interactive_html(
    out_html: Path,
    title: str,
    t_iso_full: List[str],
    t_iso_zoom: List[str],
    naive_full: np.ndarray,
    cont_full: np.ndarray,
    exp_full: Optional[np.ndarray],
    naive_zoom: np.ndarray,
    cont_zoom: np.ndarray,
    exp_zoom: Optional[np.ndarray],
) -> None:
    payload = {
        "t_full": t_iso_full,
        "t_zoom": t_iso_zoom,
        "naive_full": np.where(np.isfinite(naive_full), naive_full, np.nan).tolist(),
        "cont_full": np.where(np.isfinite(cont_full), cont_full, np.nan).tolist(),
        "naive_zoom": np.where(np.isfinite(naive_zoom), naive_zoom, np.nan).tolist(),
        "cont_zoom": np.where(np.isfinite(cont_zoom), cont_zoom, np.nan).tolist(),
        "exp_full": None if exp_full is None else np.where(np.isfinite(exp_full), exp_full, np.nan).tolist(),
        "exp_zoom": None if exp_zoom is None else np.where(np.isfinite(exp_zoom), exp_zoom, np.nan).tolist(),
    }

    html = f"""<!doctype html>
<html lang=\"en\"><head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>{title}</title>
<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<style>
body {{ font-family: Arial, sans-serif; margin: 16px; background: #f6f8fb; color: #1f2937; }}
.panel {{ background: #fff; border: 1px solid #dbe2ea; border-radius: 10px; padding: 10px; margin-bottom: 14px; }}
.plot {{ width: 100%; height: 440px; }}
</style>
</head><body>
<div class=\"panel\"><div id=\"full\" class=\"plot\"></div></div>
<div class=\"panel\"><div id=\"zoom\" class=\"plot\"></div></div>
<script>
const D = {json.dumps(payload)};

function traces(t, naive, cont, exp) {{
  const tr = [
    {{ x: t, y: naive, type: 'scatter', mode: 'lines', name: '|omega| naive', line: {{color:'#1f77b4'}} }},
    {{ x: t, y: cont,  type: 'scatter', mode: 'lines', name: '|omega| continuous', line: {{color:'#2ca02c'}} }}
  ];
  if (exp) tr.push({{ x: t, y: exp, type: 'scatter', mode: 'lines', name: 'org_gyro_mag (export)', line: {{color:'#e4572e'}} }});
  return tr;
}}

const axisTime = {{ type:'date', tickformat:'%M:%S', hoverformat:'%M:%S.%L' }};

Plotly.newPlot('full', traces(D.t_full, D.naive_full, D.cont_full, D.exp_full), {{
  title: 'Full Timeline: |omega| from quaternions',
  template:'plotly_white', hovermode:'x unified',
  xaxis: Object.assign({{}}, axisTime, {{ rangeslider: {{visible:true}} }}),
  yaxis: {{ title:'Angular speed' }}
}}, {{responsive:true}});

Plotly.newPlot('zoom', traces(D.t_zoom, D.naive_zoom, D.cont_zoom, D.exp_zoom), {{
  title: 'Zoom Window',
  template:'plotly_white', hovermode:'x unified',
  xaxis: axisTime,
  yaxis: {{ title:'Angular speed' }}
}}, {{responsive:true}});
</script>
</body></html>
"""
    out_html.write_text(html, encoding="utf-8")


def maybe_export_mag(data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if "org_gyro_mag" in data:
        return data["org_gyro_mag"].astype(float)
    keys = ("org_gyro_x", "org_gyro_y", "org_gyro_z")
    if all(k in data for k in keys):
        gx, gy, gz = data[keys[0]], data[keys[1]], data[keys[2]]
        return np.sqrt(gx * gx + gy * gy + gz * gz)
    return None


def analyze_one(
    path: Path,
    center_s: float,
    half_window_s: float,
    outdir: Path,
    dt_factor: float,
    top_k_spikes: int,
    max_full_points: int,
) -> None:
    data = load_csv(path)
    cols = list(data.keys())

    ts_col = detect_timestamp_col(cols)
    quat_cols = detect_quat_cols(cols)

    print(f"\n=== {path} ===")
    print(f"Detected timestamp column: {ts_col}")
    print(f"Detected quaternion columns: {quat_cols}")

    if ts_col is None or quat_cols is None:
        raise SystemExit(f"Could not auto-detect timestamp/quaternion columns in {path}")

    t = normalize_time(data[ts_col], ts_col)
    q = np.column_stack([data[quat_cols[0]], data[quat_cols[1]], data[quat_cols[2]], data[quat_cols[3]]])
    q = quat_normalize(q)

    # full timeline diagnostics
    dt = np.full_like(t, np.nan)
    dt[1:] = t[1:] - t[:-1]
    dt_mask, dt_median = find_dt_outliers(dt, dt_factor)
    dt_pos = dt[np.isfinite(dt) & (dt > 0)]

    dots = np.full_like(t, np.nan)
    dots[1:] = quat_dot(q[1:], q[:-1])

    sign_flip_idx = np.where(np.isfinite(dots) & (dots < 0))[0]
    worst_dot_idx = np.where(np.isfinite(dots))[0]
    worst_dot_idx = worst_dot_idx[np.argsort(dots[worst_dot_idx])] if worst_dot_idx.size else np.array([], dtype=int)

    q_cont = enforce_hemisphere_continuity(q)
    omega_naive = derive_omega_mag(t, q)
    omega_cont = derive_omega_mag(t, q_cont)

    exp_mag = maybe_export_mag(data)

    # focus window
    w0 = max(0.0, center_s - half_window_s)
    w1 = center_s + half_window_s
    zoom_mask = np.isfinite(t) & (t >= w0) & (t <= w1)

    corr_full_naive = finite_corr(omega_naive, exp_mag) if exp_mag is not None else float("nan")
    corr_full_cont = finite_corr(omega_cont, exp_mag) if exp_mag is not None else float("nan")
    corr_zoom_naive = finite_corr(omega_naive[zoom_mask], exp_mag[zoom_mask]) if exp_mag is not None else float("nan")
    corr_zoom_cont = finite_corr(omega_cont[zoom_mask], exp_mag[zoom_mask]) if exp_mag is not None else float("nan")

    dev_zoom_naive = max_abs_dev(omega_naive[zoom_mask], exp_mag[zoom_mask]) if exp_mag is not None else float("nan")
    dev_zoom_cont = max_abs_dev(omega_cont[zoom_mask], exp_mag[zoom_mask]) if exp_mag is not None else float("nan")

    peak_naive = top_k(omega_naive, t, top_k_spikes)
    peak_cont = top_k(omega_cont, t, top_k_spikes)

    print("-- dt diagnostics (full) --")
    if dt_pos.size:
        print(f"dt min/median/max: {np.min(dt_pos):.9f} / {np.median(dt_pos):.9f} / {np.max(dt_pos):.9f} s")
    else:
        print("No positive dt samples")
    print(f"dt outliers count (dt > {dt_factor}x median or dt<=0): {int(np.sum(dt_mask))}")
    outlier_idx = np.where(dt_mask)[0][:20]
    for i in outlier_idx:
        print(f"  dt outlier idx={i} t={t[i]:.6f}s dt={dt[i]:.9f}")

    print("-- quaternion dot diagnostics (full) --")
    print(f"sign-flip candidates count (dot<0): {sign_flip_idx.size}")
    for i in sign_flip_idx[:20]:
        print(f"  sign-flip idx={i} t={t[i]:.6f}s dot={dots[i]:.9f}")
    print("worst dots:")
    for i in worst_dot_idx[:10]:
        print(f"  idx={i} t={t[i]:.6f}s dot={dots[i]:.9f}")

    print("-- omega spikes (full) --")
    print("Top naive |omega| spikes:")
    for i, ts, v in peak_naive:
        print(f"  idx={i} t={ts:.6f}s |omega|={v:.9f}")
    print("Top continuous |omega| spikes:")
    for i, ts, v in peak_cont:
        print(f"  idx={i} t={ts:.6f}s |omega|={v:.9f}")

    if exp_mag is not None:
        print("-- comparison to exported org_gyro_mag --")
        print(f"Full corr naive/cont: {corr_full_naive:.6f} / {corr_full_cont:.6f}")
        print(f"Zoom corr naive/cont: {corr_zoom_naive:.6f} / {corr_zoom_cont:.6f}")
        print(f"Zoom max |dev| naive/cont: {dev_zoom_naive:.9f} / {dev_zoom_cont:.9f}")
    else:
        print("org_gyro_mag not present; skipping correlation/deviation comparison")

    # events CSV
    events_csv = outdir / f"{path.stem}_events.csv"
    with events_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_type", "index", "timestamp_s", "value", "detail"])
        for i in np.where(dt_mask)[0]:
            w.writerow(["dt_outlier", int(i), float(t[i]), float(dt[i]), f"factor={dt_factor}x_median={dt_median:.9f}"])
        for i in sign_flip_idx:
            w.writerow(["sign_flip_candidate", int(i), float(t[i]), float(dots[i]), "dot<0"])
        for rank, (i, ts, v) in enumerate(peak_naive, 1):
            w.writerow(["omega_peak_naive", int(i), float(ts), float(v), f"rank={rank}"])
        for rank, (i, ts, v) in enumerate(peak_cont, 1):
            w.writerow(["omega_peak_continuous", int(i), float(ts), float(v), f"rank={rank}"])

    # prepare plotting arrays
    keep_full = np.where(np.isfinite(t))[0]
    if max_full_points > 0 and keep_full.size > max_full_points:
        step = int(math.ceil(keep_full.size / max_full_points))
        keep_full = keep_full[::step]

    keep_zoom = np.where(zoom_mask)[0]

    t_full = t[keep_full]
    t_zoom = t[keep_zoom]

    out_html = outdir / f"{path.stem}_quat_diagnostics.html"
    build_interactive_html(
        out_html=out_html,
        title=f"Quaternion diagnostics: {path.name}",
        t_iso_full=to_iso_time(t_full),
        t_iso_zoom=to_iso_time(t_zoom),
        naive_full=omega_naive[keep_full],
        cont_full=omega_cont[keep_full],
        exp_full=None if exp_mag is None else exp_mag[keep_full],
        naive_zoom=omega_naive[keep_zoom],
        cont_zoom=omega_cont[keep_zoom],
        exp_zoom=None if exp_mag is None else exp_mag[keep_zoom],
    )

    print(f"Wrote events CSV: {events_csv}")
    print(f"Wrote interactive plot: {out_html}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    python_root = script_dir.parent
    default_data = python_root / "data"

    p = argparse.ArgumentParser(
        description="Diagnose quaternion stream glitches (dt, sign flips, omega spikes) from exported motion CSV"
    )
    p.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        default=[default_data / "export.csv"],
        help="Input exported CSV(s). Defaults to analysis/python/data/export.csv",
    )
    p.add_argument("--center", type=float, default=196.0, help="Zoom center in seconds (default: 196 ~= 3:16)")
    p.add_argument("--half-window", type=float, default=20.0, help="Half-width of zoom window in seconds")
    p.add_argument("--outdir", type=Path, default=script_dir / "output", help="Output directory")
    p.add_argument("--dt-factor", type=float, default=5.0, help="dt outlier factor vs median")
    p.add_argument("--top-k", type=int, default=10, help="Top omega spikes to report")
    p.add_argument("--max-full-points", type=int, default=120000, help="Downsample cap for full-timeline plot")
    args = p.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("Example usage:")
    print("  python3 analysis/python/quat_glitch_diagnostics/quat_diagnostics.py analysis/python/data/export.csv --center 196 --half-window 20")

    for inp in args.inputs:
        analyze_one(
            path=inp,
            center_s=args.center,
            half_window_s=args.half_window,
            outdir=args.outdir,
            dt_factor=args.dt_factor,
            top_k_spikes=args.top_k,
            max_full_points=args.max_full_points,
        )


if __name__ == "__main__":
    main()
