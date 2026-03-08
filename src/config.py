"""
Central config: paths, model name, seeds, and data/eval constants.
"""
from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data and cache
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model
BERT_MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2  # SST-2 binary sentiment

# Reproducibility
SEED = 42

# Data defaults
SST2_SPLIT_TRAIN = "train"
SST2_SPLIT_VAL = "validation"
SST2_SPLIT_TEST = "test"
WIKITEXT103_SUBSET = "wikitext-103-raw-v1"  # HuggingFace dataset id
WIKITEXT_MAX_SAMPLES = 2000  # cap OOD eval size
CORRUPTION_PROB = 0.15  # probability per character to apply a corruption
CORRUPTION_TYPES = ("typo", "swap", "insert")  # character-level noise

# Training (for later use)
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 128
