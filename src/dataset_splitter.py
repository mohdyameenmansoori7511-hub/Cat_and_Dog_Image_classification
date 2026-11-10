"""Train, Validation and Test Data Splitting Module """
from pathlib import Path
import random
from typing import Dict, List


def partition_file_list(
    files: List[Path], train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42
) -> Dict[str, List[Path]]:
    """Partitions verified unique files into disjoint train, validation, and test lists."""
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }
