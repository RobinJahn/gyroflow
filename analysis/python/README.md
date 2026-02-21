# Python Analysis Workspace

All analysis scripts and motion exports are organized under `analysis/python/`.

## Structure

- `analysis/python/data/`
  - `export.csv`
  - `export_LP350.csv`
  - `export_LP2.csv`

- `analysis/python/compare_lpf_exports/`
  - `compare_lpf_exports.py`
  - `output/`

- `analysis/python/motion_csv_viz/`
  - `motion_csv_viz.py`
  - `output/`

- `analysis/python/quaternion_diagnostics/`
  - `quaternion_diagnostics.py`
  - `output/`

- `analysis/python/quat_glitch_diagnostics/`
  - `quat_diagnostics.py`
  - `output/`

## Usage

Run from repo root.

### 1) Compare Original / LP350 / LP2

```bash
python3 analysis/python/compare_lpf_exports/compare_lpf_exports.py
```

### 2) Single-file interactive motion plot

```bash
python3 analysis/python/motion_csv_viz/motion_csv_viz.py
```

### 3) Quaternion diagnostics

```bash
python3 analysis/python/quaternion_diagnostics/quaternion_diagnostics.py
```

### 4) Quaternion glitch diagnostics (raw-stream checks)

```bash
python3 analysis/python/quat_glitch_diagnostics/quat_diagnostics.py analysis/python/data/export.csv --center 196 --half-window 20
```

Each script defaults to data files in `analysis/python/data/` and writes outputs to its own `output/` subfolder.
