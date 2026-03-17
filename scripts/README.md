# Scripts
Run from project root (e.g. `python scripts/train.py`).

| Script | Purpose |
|--------|--------|
| `run_data_demo.py` | Verify data pipeline: SST-2, Wikitext-103, corrupted SST-2. |
| `train.py` | Fine-tune BERT on SST-2; saves best checkpoint by val accuracy. |
| `eval_baseline.py` | Standard BERT inference (no MC Dropout); saves ID predictions for baseline calibration. |
| `eval_mc_dropout.py` | Run MC Dropout inference on ID/OOD/corrupted; saves prob + logit variance. Use `--mc_dropout_prob 0.2` for stronger variance signal. |
| `eval_calibration.py` | Compute reliability bins and ECE from MC (and optionally `--baseline` npz) predictions. |
