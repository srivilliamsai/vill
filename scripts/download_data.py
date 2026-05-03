#!/usr/bin/env python3
"""
Download a sample corpus from FineWeb-Edu for tokenizer training.
Streams data from HuggingFace and writes plain text files.
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def download_sample(output_dir: str = "data/raw", num_samples: int = 50000):
    """Download text samples for tokenizer training."""
    from datasets import load_dataset

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("Streaming FineWeb-Edu sample from HuggingFace...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu-score-2",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    output_file = out / "corpus.txt"
    count = 0
    total_chars = 0

    with open(output_file, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example.get("text", "")
            if not text or len(text) < 100:
                continue

            f.write(text.strip() + "\n\n")
            count += 1
            total_chars += len(text)

            if count % 5000 == 0:
                logger.info("  Downloaded %d samples (%.1f MB)", count, total_chars / 1e6)

            if count >= num_samples:
                break

    logger.info("Done: %d samples, %.1f MB written to %s", count, total_chars / 1e6, output_file)
    return str(output_file)


if __name__ == "__main__":
    download_sample()
