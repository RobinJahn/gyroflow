# LPF Export Comparison

Scripts:
- `analysis/lpf_compare/compare_lpf_exports.py`
- `analysis/lpf_compare/motion_csv_viz.py`

Generates an interactive HTML report comparing two Gyroflow motion CSV exports.

## Run

```bash
python3 analysis/lpf_compare/compare_lpf_exports.py \
  export_LP2.csv export_LP350.csv \
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

## Single-file interactive view

```bash
python3 analysis/lpf_compare/motion_csv_viz.py \
  export_LP2.csv \
  --out analysis/lpf_compare/output/export_LP2_viz_interactive.html
```
