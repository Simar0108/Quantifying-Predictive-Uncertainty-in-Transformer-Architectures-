"""
Data loading: SST-2 (ID), Wikitext-103 (OOD), and corrupted SST-2 (adversarial-style OOD).
Provides a simple API: get_*_loader / get_*_dataset for use in scripts and notebooks.
"""
import random
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from . import config

# -----------------------------------------------------------------------------
# Character-level corruption (typos, swaps, insertions)
# -----------------------------------------------------------------------------

# Simple keyboard neighborhood for typo simulation (US QWERTY)
_KEY_NEIGHBORS = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wrsdf",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huiknm",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}


def _do_typo(c: str) -> str:
    c_lower = c.lower()
    if c_lower in _KEY_NEIGHBORS:
        return random.choice(_KEY_NEIGHBORS[c_lower])
    return c


def _do_swap(text: str, i: int) -> str:
    if i + 1 >= len(text):
        return text
    chars = list(text)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _do_insert(text: str, i: int) -> str:
    chars = list(text)
    r = random.choice("abcdefghijklmnopqrstuvwxyz ")
    chars.insert(i + 1, r)
    return "".join(chars)


def corrupt_text(
    text: str,
    prob: float = 0.15,
    seed: Optional[int] = None,
    corruption_types: tuple = ("typo", "swap", "insert"),
) -> str:
    """
    Apply character-level noise: typo (replace with nearby key), swap (adjacent chars),
    or insert (random char). Each character is considered with probability `prob`.
    """
    if seed is not None:
        random.seed(seed)
    if not text or prob <= 0:
        return text
    out = text
    # Work on indices; we may change length so track position
    i = 0
    while i < len(out):
        if random.random() > prob:
            i += 1
            continue
        choice = random.choice(corruption_types)
        if choice == "typo" and out[i].isalpha():
            out = out[:i] + _do_typo(out[i]) + out[i + 1 :]
        elif choice == "swap":
            out = _do_swap(out, i)
            i += 1  # skip next so we don't double-perturb
        elif choice == "insert":
            out = _do_insert(out, i)
            i += 1
        i += 1
    return out


# -----------------------------------------------------------------------------
# HuggingFace dataset loading
# -----------------------------------------------------------------------------


def get_sst2_dataset(split: str):
    """Load SST-2 (Stanford Sentiment Treebank) for in-distribution. Splits: train, validation, test."""
    return load_dataset("glue", "sst2", split=split, trust_remote_code=True)


def get_wikitext103_dataset(
    max_samples: Optional[int] = None,
    split: str = "test",
):
    """
    Load Wikitext-103 as OOD (non-sentiment text). Returns dataset with 'text' and dummy 'label' -1.
    """
    ds = load_dataset("wikitext", config.WIKITEXT103_SUBSET, split=split, trust_remote_code=True)
    # Wikitext has 'text'; filter empty lines and take contiguous text as "samples"
    text_key = "text" if "text" in ds.column_names else "paragraph"
    lines = [row[text_key] for row in ds if row[text_key].strip()]
    if max_samples is not None:
        lines = lines[:max_samples]
    # Build a simple Dataset-like structure: list of dicts
    from datasets import Dataset as HFDataset
    return HFDataset.from_dict({"sentence": lines, "label": [-1] * len(lines)})


def get_corrupted_sst2_dataset(split: str = "test", prob: float = 0.15, seed: Optional[int] = None):
    """Load SST-2 and apply character-level corruption. Labels unchanged (same as clean)."""
    ds = get_sst2_dataset(split)
    seed = seed if seed is not None else config.SEED

    def map_fn(ex):
        idx = ex.get("idx", 0)
        if isinstance(idx, (list, np.ndarray)):
            idx = int(idx[0]) if len(idx) else 0
        return {"sentence": corrupt_text(ex["sentence"], prob=prob, seed=seed + idx)}

    return ds.map(map_fn, desc="corrupt")


# -----------------------------------------------------------------------------
# Tokenization and DataLoaders
# -----------------------------------------------------------------------------


def _collate_sentiment_batch(
    batch: list,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    label_key: str = "label",
) -> dict:
    """Turn list of {sentence, label} into tokenized batch."""
    sentences = [b["sentence"] for b in batch]
    labels = [b[label_key] for b in batch]
    enc = tokenizer(
        sentences,
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels_t = torch.tensor(labels, dtype=torch.long)
    return {**enc, "labels": labels_t}


def get_tokenizer():
    """Load tokenizer for the configured BERT model."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.BERT_MODEL_NAME)


def get_id_loaders(
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    batch_size: int = 32,
    max_length: int = 128,
):
    """
    Return (train_loader, val_loader, test_loader) for SST-2.
    Use for training and in-distribution evaluation.
    """
    tokenizer = tokenizer or get_tokenizer()
    train_ds = get_sst2_dataset(config.SST2_SPLIT_TRAIN)
    val_ds = get_sst2_dataset(config.SST2_SPLIT_VAL)
    test_ds = get_sst2_dataset(config.SST2_SPLIT_TEST)

    def collate(batch):
        return _collate_sentiment_batch(batch, tokenizer, max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    return train_loader, val_loader, test_loader


def get_ood_wikitext_loader(
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    batch_size: int = 32,
    max_length: int = 128,
    max_samples: Optional[int] = None,
):
    """DataLoader for Wikitext-103 OOD. Labels are -1 (dummy)."""
    tokenizer = tokenizer or get_tokenizer()
    ds = get_wikitext103_dataset(max_samples=max_samples or config.WIKITEXT_MAX_SAMPLES)
    def collate(batch):
        return _collate_sentiment_batch(batch, tokenizer, max_length, label_key="label")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)


def get_corrupted_sst2_loader(
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    split: str = "test",
    batch_size: int = 32,
    max_length: int = 128,
    prob: float = 0.15,
    seed: Optional[int] = None,
):
    """DataLoader for character-corrupted SST-2 (OOD / adversarial-style). Labels are real SST-2 labels."""
    tokenizer = tokenizer or get_tokenizer()
    ds = get_corrupted_sst2_dataset(split=split, prob=prob, seed=seed)
    def collate(batch):
        return _collate_sentiment_batch(batch, tokenizer, max_length)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
</think>
Fixing the corrupted-SST-2 map to preserve labels and other columns.
<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>
StrReplace