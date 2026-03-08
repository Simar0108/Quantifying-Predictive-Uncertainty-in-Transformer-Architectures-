#!/usr/bin/env python3
"""
Fine-tune BERT on SST-2 for sentiment classification. Saves best checkpoint by validation accuracy.
Run from project root: python scripts/train.py [--epochs 3] [--batch_size 32] [--out checkpoints/bert_sst2]
"""
import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from src.data import get_tokenizer, get_id_loaders
from src.models import BERTSentimentClassifier


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on SST-2")
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=config.DEFAULT_MAX_LENGTH)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--out", type=str, default=str(config.CHECKPOINT_DIR / config.DEFAULT_CHECKPOINT_NAME))
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = get_tokenizer()
    train_loader, val_loader, _ = get_id_loaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    model = BERTSentimentClassifier(
        model_name=config.BERT_MODEL_NAME,
        num_labels=config.NUM_LABELS,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        val_acc = eval_accuracy(model, val_loader, device)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "config": model.config, "epoch": epoch, "val_acc": val_acc},
                out_path,
            )
            print(f"  -> saved best to {out_path}")

    print(f"Done. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
