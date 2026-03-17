#!/usr/bin/env python3
"""
Run standard BERT inference (no MC Dropout): single forward pass with dropout disabled.
Saves ID predictions and confidences for baseline calibration comparison.
Run from project root: python scripts/eval_baseline.py --checkpoint checkpoints/bert_sst2 --out results/baseline_predictions.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.data import get_tokenizer, get_id_loaders
from src.models import BERTSentimentClassifier


@torch.no_grad()
def run_baseline_inference(model, loader, device, desc="Eval"):
    """Single pass, model.eval() so dropout is off. Return mean_positive, preds, labels."""
    model.eval()
    mean_pos_list, preds_list, labels_list = [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        probs = F.softmax(logits, dim=-1)
        # Binary: P(positive) = probs[:, 1]
        mean_pos_list.append(probs[:, 1].cpu().numpy())
        preds_list.append(logits.argmax(dim=-1).cpu().numpy())
        if "labels" in batch:
            labels_list.append(batch["labels"].numpy())
    return (
        np.concatenate(mean_pos_list),
        np.concatenate(preds_list),
        np.concatenate(labels_list) if labels_list else None,
    )


def main():
    parser = argparse.ArgumentParser(description="Standard BERT inference (baseline, no MC Dropout)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved BERT checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=config.DEFAULT_MAX_LENGTH)
    parser.add_argument("--out", type=str, default="results/baseline_predictions.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    model = BERTSentimentClassifier(
        model_name=config.BERT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Use validation split for ID eval: GLUE SST-2 test set has hidden labels (-1), so use val for accuracy/calibration.
    _, val_loader, _ = get_id_loaders(
        tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_length
    )

    print("Running baseline (standard BERT) on ID (SST-2 validation)...")
    id_mean, id_pred, id_labels = run_baseline_inference(model, val_loader, device, desc="ID")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        id_mean_positive=id_mean,
        id_pred=id_pred,
        id_labels=id_labels,
    )
    acc = (id_pred == id_labels).mean()
    print(f"Saved to {out_path}")
    print(f"Baseline ID accuracy = {acc:.4f}  (n={len(id_labels)})")


if __name__ == "__main__":
    main()
