# Scripts
Run from project root (e.g. `python scripts/train.py`).

| Script | Purpose |
|--------|--------|
| `run_data_demo.py` | Verify data pipeline: SST-2, Wikitext-103, corrupted SST-2. |
| `train.py` | Fine-tune BERT on SST-2; saves best checkpoint by val accuracy. |
| `eval_mc_dropout.py` | Run MC Dropout inference on ID/OOD/corrupted; saves predictions to npz. |
| `eval_calibration.py` | Compute reliability bins and ECE from `mc_predictions.npz`. |
