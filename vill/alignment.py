"""
Vill -- Alignment (SFT + DPO)
-------------------------------
Post-training alignment pipeline:
1. Supervised Fine-Tuning (SFT) on instruction-response pairs.
2. Direct Preference Optimization (DPO) on preference data.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from vill.model.transformer import VillForCausalLM

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SFT Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.

    Expects data in the format:
        [{"instruction": "...", "input": "...", "output": "..."}, ...]
    or:
        [{"messages": [{"role": "user", "content": "..."}, ...]}, ...]
    """

    def __init__(self, examples: List[dict], tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for ex in examples:
            if "messages" in ex:
                ids = tokenizer.encode_chat(ex["messages"])
            else:
                text = self._format_instruction(ex)
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)

            if len(ids) > max_length:
                ids = ids[:max_length]

            self.examples.append(ids)

    def _format_instruction(self, ex: dict) -> str:
        instruction = ex.get("instruction", "")
        inp = ex.get("input", "")
        output = ex.get("output", "")
        if inp:
            return f"{instruction}\n{inp}\n{output}"
        return f"{instruction}\n{output}"

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ids = self.examples[idx]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# DPO Loss
# ---------------------------------------------------------------------------

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Direct Preference Optimization loss (Rafailov et al., 2023).

    Eliminates the need for a separate reward model by directly
    optimizing the policy to prefer chosen responses over rejected ones,
    using a reference model as the anchor.

    Args:
        policy_chosen_logps: Log probs of chosen sequences under the policy.
        policy_rejected_logps: Log probs of rejected sequences under the policy.
        reference_chosen_logps: Log probs of chosen sequences under the reference.
        reference_rejected_logps: Log probs of rejected sequences under the reference.
        beta: Temperature parameter controlling deviation from the reference.

    Returns:
        Scalar DPO loss.
    """
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss


def compute_sequence_logps(
    model: VillForCausalLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the total log probability of a sequence under the model."""
    outputs = model(input_ids=input_ids, labels=labels)
    logits = outputs["logits"][:, :-1, :]
    target = labels[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    per_token = torch.gather(log_probs, 2, target.unsqueeze(-1)).squeeze(-1)
    mask = (target != -100).float()
    return (per_token * mask).sum(dim=-1)


@dataclass
class AlignmentConfig:
    """Configuration for alignment training."""
    learning_rate: float = 5e-6
    num_epochs: int = 3
    batch_size: int = 2
    max_length: int = 2048
    beta: float = 0.1      # DPO temperature
    output_dir: str = "checkpoints"


def run_sft(
    model: VillForCausalLM,
    train_data: List[dict],
    tokenizer,
    config: AlignmentConfig,
    device: torch.device = None,
) -> None:
    """
    Run Supervised Fine-Tuning.

    Args:
        model: The pre-trained Vill model.
        train_data: List of instruction-response examples.
        tokenizer: VillTokenizer instance.
        config: Alignment configuration.
        device: Target device.
    """
    if device is None:
        device = next(model.parameters()).device

    dataset = SFTDataset(train_data, tokenizer, config.max_length)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    model.train()

    for epoch in range(config.num_epochs):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)
        logger.info("SFT Epoch %d/%d  loss=%.4f", epoch + 1, config.num_epochs, avg_loss)

    logger.info("SFT complete.")
