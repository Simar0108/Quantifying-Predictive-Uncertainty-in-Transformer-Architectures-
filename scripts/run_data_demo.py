#!/usr/bin/env python3
"""
Verify the data pipeline: load SST-2, Wikitext-103, and corrupted SST-2;
print shapes and a few samples. Run from project root: python scripts/run_data_demo.py
"""
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import (
    get_sst2_dataset,
    get_wikitext103_dataset,
    get_corrupted_sst2_dataset,
    get_tokenizer,
    get_id_loaders,
    get_ood_wikitext_loader,
    get_corrupted_sst2_loader,
    corrupt_text,
)
from src import config


def main():
    print("=== 1. SST-2 (ID) ===")
    train = get_sst2_dataset(config.SST2_SPLIT_TRAIN)
    test = get_sst2_dataset(config.SST2_SPLIT_TEST)
    print(f"  Train: {len(train)}, Test: {len(test)}")
    print(f"  Example: {train[0]}")

    print("\n=== 2. Wikitext-103 (OOD) ===")
    ood = get_wikitext103_dataset(max_samples=100)
    print(f"  Samples: {len(ood)}")
    print(f"  Example sentence (first 80 chars): {ood[0]['sentence'][:80]}...")
    print(f"  Label (dummy): {ood[0]['label']}")

    print("\n=== 3. Corrupted SST-2 ===")
    clean = "A visually stunning masterpiece."
    corrupted = corrupt_text(clean, prob=0.2, seed=config.SEED)
    print(f"  Clean:    {clean}")
    print(f"  Corrupted: {corrupted}")
    corrupt_ds = get_corrupted_sst2_dataset(split=config.SST2_SPLIT_TEST, prob=0.15)
    print(f"  Test corrupted size: {len(corrupt_ds)}")
    print(f"  Example: {corrupt_ds[0]}")

    print("\n=== 4. DataLoaders (one batch) ===")
    tokenizer = get_tokenizer()
    train_loader, val_loader, test_loader = get_id_loaders(tokenizer=tokenizer, batch_size=4, max_length=64)
    batch = next(iter(train_loader))
    print(f"  ID batch keys: {list(batch.keys())}")
    print(f"  input_ids shape: {batch['input_ids'].shape}, labels shape: {batch['labels'].shape}")

    wikitext_loader = get_ood_wikitext_loader(tokenizer=tokenizer, batch_size=4, max_samples=20)
    ood_batch = next(iter(wikitext_loader))
    print(f"  OOD Wikitext batch input_ids shape: {ood_batch['input_ids'].shape}")

    corrupt_loader = get_corrupted_sst2_loader(tokenizer=tokenizer, batch_size=4, split=config.SST2_SPLIT_TEST)
    corr_batch = next(iter(corrupt_loader))
    print(f"  Corrupted SST-2 batch input_ids shape: {corr_batch['input_ids'].shape}, labels: {corr_batch['labels'].tolist()}")

    print("\n=== Data pipeline OK ===")


if __name__ == "__main__":
    main()
