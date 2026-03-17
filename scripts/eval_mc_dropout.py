#!/usr/bin/env python3
"""
Run MC Dropout inference on ID (SST-2 test), OOD (Wikitext-103), and corrupted SST-2.
Saves mean_probs, var_positive, predictions, and labels (where available) for calibration and OOD analysis.
Run from project root: python scripts/eval_mc_dropout.py --checkpoint checkpoints/bert_sst2 --T 30 --out results/mc_predictions.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.data import get_tokenizer, get_id_loaders, get_ood_wikitext_loader, get_corrupted_sst2_loader
from src.models import BERTSentimentClassifier, MCDropoutWrapper


def run_mc_inference(model, loader, device, desc="Eval"):
    mean_pos_list, var_pos_list, preds_list, labels_list = [], [], [], []
    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        mean_pos_list.append(out.mean_positive.cpu().numpy())
        var_pos_list.append(out.var_positive.cpu().numpy())
        preds_list.append(out.mean_probs.argmax(dim=-1).cpu().numpy())
        if "labels" in batch:
            labels_list.append(batch["labels"].numpy())
    return (
        np.concatenate(mean_pos_list),
        np.concatenate(var_pos_list),
        np.concatenate(preds_list),
        np.concatenate(labels_list) if labels_list else None,
    )


def main():
    parser = argparse.ArgumentParser(description="MC Dropout inference on ID and OOD")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved BERT checkpoint")
    parser.add_argument("--T", type=int, default=30, help="Number of MC samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=config.DEFAULT_MAX_LENGTH)
    parser.add_argument("--max_ood_samples", type=int, default=config.WIKITEXT_MAX_SAMPLES)
    parser.add_argument("--out", type=str, default="results/mc_predictions.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer()

    # Load model and wrap with MC Dropout
    base = BERTSentimentClassifier(
        model_name=config.BERT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(ckpt["model_state_dict"])
    model = MCDropoutWrapper(base, num_samples=args.T).to(device)
    model.eval()

    # Use validation split for ID/corrupted: GLUE SST-2 test set has hidden labels (-1).
    _, val_loader, _ = get_id_loaders(
        tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_length
    )
    wikitext_loader = get_ood_wikitext_loader(
        tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_length, max_samples=args.max_ood_samples
    )
    corrupted_loader = get_corrupted_sst2_loader(
        tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_length, split=config.SST2_SPLIT_VAL
    )

    print("Running MC Dropout on ID (SST-2 validation)...")
    id_mean, id_var, id_pred, id_labels = run_mc_inference(model, val_loader, device, desc="ID")
    print("Running MC Dropout on OOD (Wikitext-103)...")
    ood_mean, ood_var, ood_pred, _ = run_mc_inference(model, wikitext_loader, device, desc="OOD Wikitext")
    print("Running MC Dropout on corrupted SST-2...")
    corr_mean, corr_var, corr_pred, corr_labels = run_mc_inference(model, corrupted_loader, device, desc="Corrupted")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        id_mean_positive=id_mean,
        id_var_positive=id_var,
        id_pred=id_pred,
        id_labels=id_labels,
        ood_wikitext_mean=ood_mean,
        ood_wikitext_var=ood_var,
        ood_wikitext_pred=ood_pred,
        corr_mean_positive=corr_mean,
        corr_var_positive=corr_var,
        corr_pred=corr_pred,
        corr_labels=corr_labels,
    )
    print(f"Saved to {out_path}")
    print(f"ID    mean(var) = {id_var.mean():.4f}  (n={len(id_var)})")
    print(f"OOD   mean(var) = {ood_var.mean():.4f}  (n={len(ood_var)})")
    print(f"Corr  mean(var) = {corr_var.mean():.4f}  (n={len(corr_var)})")
    if id_var.mean() > 0:
        print(f"OOD/ID variance ratio = {ood_var.mean() / id_var.mean():.2f}x")
        print(f"Corr/ID variance ratio = {corr_var.mean() / id_var.mean():.2f}x")


if __name__ == "__main__":
    main()
