"""
Vill -- Data Pipeline
-----------------------
Streaming data loading from HuggingFace datasets.
Handles tokenization, sequence packing, and data mixing.
"""

from __future__ import annotations

import logging
from typing import Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader

logger = logging.getLogger(__name__)


class PretrainingDataset(IterableDataset):
    """
    Streaming dataset for pre-training.

    Loads data from HuggingFace datasets in streaming mode (no full
    download required). Tokenizes on-the-fly and packs sequences to
    the target length for maximum GPU utilization.

    Args:
        dataset_name: HuggingFace dataset identifier.
        tokenizer: A VillTokenizer instance.
        seq_length: Target sequence length for packed examples.
        split: Dataset split to use.
        text_field: Name of the text column in the dataset.
    """

    def __init__(
        self,
        dataset_name: str,
        tokenizer,
        seq_length: int = 2048,
        split: str = "train",
        text_field: str = "text",
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.split = split
        self.text_field = text_field

    def __iter__(self) -> Iterator[dict]:
        from datasets import load_dataset

        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            streaming=True,
            trust_remote_code=True,
        )

        buffer = []

        for example in dataset:
            text = example.get(self.text_field, "")
            if not text or len(text) < 50:
                continue

            tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_length + 1:
                chunk = buffer[: self.seq_length + 1]
                buffer = buffer[self.seq_length:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}


class TextFileDataset(IterableDataset):
    """
    Simple streaming dataset from local text files.
    Useful for initial testing on small local corpora.
    """

    def __init__(self, file_paths: List[str], tokenizer, seq_length: int = 2048):
        super().__init__()
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __iter__(self) -> Iterator[dict]:
        buffer = []

        for path in self.file_paths:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    tokens = self.tokenizer.encode(line, add_bos=False, add_eos=False)
                    buffer.extend(tokens)

                    while len(buffer) >= self.seq_length + 1:
                        chunk = buffer[: self.seq_length + 1]
                        buffer = buffer[self.seq_length:]
                        yield {
                            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                            "labels": torch.tensor(chunk[1:], dtype=torch.long),
                        }


def create_dataloader(
    dataset: IterableDataset,
    batch_size: int = 4,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader from a streaming dataset."""
    use_pin = torch.cuda.is_available()  # Only pin memory on CUDA
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin,
    )
