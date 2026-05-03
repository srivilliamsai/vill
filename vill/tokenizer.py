"""
Vill -- BPE Tokenizer
-----------------------
Trains and provides a Byte Pair Encoding tokenizer using the
HuggingFace tokenizers library. This is the same algorithm used
by GPT, Llama, and Mistral.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

logger = logging.getLogger(__name__)

# Special tokens used by the Vill chat format
SPECIAL_TOKENS = [
    "<|bos|>",       # Beginning of sequence
    "<|eos|>",       # End of sequence
    "<|pad|>",       # Padding
    "<|user|>",      # Start of user turn
    "<|assistant|>",  # Start of assistant turn
    "<|system|>",    # System prompt marker
]


class VillTokenizer:
    """
    Wrapper around a trained BPE tokenizer.
    Handles encoding, decoding, and special token management.
    """

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self.bos_id = tokenizer.token_to_id("<|bos|>")
        self.eos_id = tokenizer.token_to_id("<|eos|>")
        self.pad_id = tokenizer.token_to_id("<|pad|>")

    @classmethod
    def train(
        cls,
        files: List[str],
        vocab_size: int = 32000,
        output_dir: str = "tokenizer_model",
        min_frequency: int = 2,
    ) -> "VillTokenizer":
        """
        Train a new BPE tokenizer from text files.

        Args:
            files: List of paths to training text files.
            vocab_size: Target vocabulary size.
            output_dir: Directory to save the trained tokenizer.
            min_frequency: Minimum frequency for a merge to occur.

        Returns:
            A trained VillTokenizer instance.
        """
        tokenizer = Tokenizer(models.BPE(unk_token="<|eos|>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )

        logger.info("Training BPE tokenizer on %d files with vocab_size=%d", len(files), vocab_size)
        tokenizer.train(files, trainer)

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(out / "tokenizer.json"))

        meta = {"vocab_size": tokenizer.get_vocab_size(), "special_tokens": SPECIAL_TOKENS}
        (out / "tokenizer_config.json").write_text(json.dumps(meta, indent=2))
        logger.info("Tokenizer saved to %s (vocab_size=%d)", output_dir, meta["vocab_size"])

        return cls(tokenizer)

    @classmethod
    def from_pretrained(cls, path: str) -> "VillTokenizer":
        """Load a previously trained tokenizer."""
        tokenizer = Tokenizer.from_file(str(Path(path) / "tokenizer.json"))
        return cls(tokenizer)

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        ids = self._tokenizer.encode(text).ids
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        if skip_special:
            special_ids = {self.bos_id, self.eos_id, self.pad_id}
            ids = [i for i in ids if i not in special_ids]
        return self._tokenizer.decode(ids)

    def encode_chat(self, messages: List[dict]) -> List[int]:
        """
        Encode a chat conversation in the Vill chat format.

        Args:
            messages: List of {"role": "user"|"assistant"|"system", "content": "..."}

        Returns:
            Token IDs for the full conversation.
        """
        role_tokens = {
            "system": self._tokenizer.token_to_id("<|system|>"),
            "user": self._tokenizer.token_to_id("<|user|>"),
            "assistant": self._tokenizer.token_to_id("<|assistant|>"),
        }

        ids = [self.bos_id]
        for msg in messages:
            role_id = role_tokens.get(msg["role"])
            if role_id is not None:
                ids.append(role_id)
            ids.extend(self._tokenizer.encode(msg["content"]).ids)
        ids.append(self.eos_id)
        return ids

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()
