# LPF Export Comparison

Scripts:
- `analysis/lpf_compare/compare_lpf_exports.py`
- `analysis/lpf_compare/motion_csv_viz.py`

Generates an interactive HTML report comparing three Gyroflow motion CSV exports:
- Original
- LP350
- LP2

## Run

```bash
python3 analysis/lpf_compare/compare_lpf_exports.py \
  export.csv export_LP350.csv export_LP2.csv \
  --outdir analysis/lpf_compare/output \
  --focus 03:16 \
  --focus-window 24
```

## Output

- `analysis/lpf_compare/output/lpf_compare_interactive.html`
- `analysis/lpf_compare/output/<csv_stem>_viz_interactive.html`

Notes:
- Time axis uses `MM:SS` labels.
- Full timeline is downsampled for responsiveness.
- Zoom panel keeps full data around the focus timestamp.
- Includes vibration spectrum plots (`dB` vs `Hz`):
  - Whole data range.
  - Local time-window spectrum driven by an interactive slider.
  - Frequency range defaults to near Nyquist (auto), so it can show beyond 300 Hz.

## Single-file interactive view

```bash
python3 analysis/lpf_compare/motion_csv_viz.py \
  export_LP2.csv \
  --out analysis/lpf_compare/output/export_LP2_viz_interactive.html
```
