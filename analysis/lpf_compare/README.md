# LPF Export Comparison

Script: `analysis/lpf_compare/compare_lpf_exports.py`

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

Notes:
- Time axis uses `MM:SS` labels.
- Full timeline is downsampled for responsiveness.
- Zoom panel keeps full data around the focus timestamp.
