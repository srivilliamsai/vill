#!/usr/bin/env python3
"""
Vill Model Diagnostic & Noise Analysis Tool
============================================
Runs a complete health check on a Vill checkpoint including:
  - Architecture & parameter count
  - Weight statistics (mean, std, NaN/Inf detection)
  - Noise/dead neuron analysis
  - Perplexity estimation
  - Generation quality test
  - Improvement recommendations
"""

import sys
import os
import math
import torch
import json
from pathlib import Path
from collections import defaultdict

# ── colour helpers ─────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✓{RESET} {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET} {msg}")
def err(msg):   print(f"  {RED}✗{RESET} {msg}")
def info(msg):  print(f"  {CYAN}→{RESET} {msg}")
def section(title):
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


# ── 1. Load checkpoint ─────────────────────────────────────────────────────────
CHECKPOINT = "checkpoints/vill_step_50000.pt"

section("1 · LOADING CHECKPOINT")
if not Path(CHECKPOINT).exists():
    err(f"Checkpoint not found: {CHECKPOINT}")
    sys.exit(1)

info(f"Loading {CHECKPOINT} ...")
ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
info(f"Checkpoint keys: {list(ckpt.keys())}")

# Extract config and state_dict
config_dict   = ckpt.get("config", {})
state_dict    = ckpt.get("model_state_dict", ckpt)
training_step = ckpt.get("step", 50000)
train_loss    = ckpt.get("loss", None)

ok(f"Checkpoint loaded  (step={training_step:,})")
if train_loss:
    ok(f"Last recorded loss: {train_loss:.4f}")


# ── 2. Architecture info ───────────────────────────────────────────────────────
section("2 · ARCHITECTURE & CAPACITY")

sys.path.insert(0, str(Path(__file__).parent.parent))
from vill.model.config import get_config, VillConfig
from vill.model.transformer import VillForCausalLM

if config_dict:
    config = VillConfig(**config_dict)
else:
    config = get_config("vill-nano")

model = VillForCausalLM(config)
model.load_state_dict(state_dict)
model.eval()

total_params  = model.count_parameters()
trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
size_fp32_gb  = total_params * 4 / 1e9
size_bf16_gb  = total_params * 2 / 1e9

info(f"Config:              {getattr(config, 'name', 'vill-nano')}")
info(f"Total parameters:    {total_params:,}")
info(f"Trainable params:    {trainable:,}")
info(f"Size (float32):      {size_fp32_gb:.3f} GB")
info(f"Size (bfloat16):     {size_bf16_gb:.3f} GB")
info(f"Layers:              {config.num_hidden_layers}")
info(f"Attention heads:     {config.num_attention_heads}  (KV heads: {config.num_key_value_heads})")
info(f"Hidden size:         {config.hidden_size}")
info(f"Context length:      {config.max_position_embeddings:,} tokens")


# ── 3. Weight noise / health analysis ─────────────────────────────────────────
section("3 · WEIGHT NOISE & HEALTH ANALYSIS")

has_nan = False
has_inf = False
dead_layers = []
noisy_layers = []
layer_stats = {}

for name, param in model.named_parameters():
    data = param.data.float()
    n_nan = torch.isnan(data).sum().item()
    n_inf = torch.isinf(data).sum().item()
    mean  = data.mean().item()
    std   = data.std().item()
    abs_max = data.abs().max().item()

    layer_stats[name] = {
        "shape": list(param.shape),
        "numel": param.numel(),
        "mean":  round(mean, 6),
        "std":   round(std, 6),
        "abs_max": round(abs_max, 6),
        "nan": n_nan,
        "inf": n_inf,
    }

    if n_nan > 0:
        has_nan = True
        err(f"NaN detected in '{name}'  ({n_nan} values)")
    if n_inf > 0:
        has_inf = True
        err(f"Inf detected in '{name}'  ({n_inf} values)")

    # Dead neurons: std near zero (weights not learning)
    if std < 1e-6 and "norm" not in name and "embed" not in name:
        dead_layers.append(name)

    # Exploding weights
    if abs_max > 100:
        noisy_layers.append((name, abs_max))

# Summary
total_layers = len(layer_stats)
if not has_nan:
    ok("No NaN values found in any layer ✓")
if not has_inf:
    ok("No Inf values found in any layer ✓")

if dead_layers:
    warn(f"Potentially dead/frozen layers ({len(dead_layers)}):")
    for l in dead_layers[:5]:
        warn(f"  • {l}")
else:
    ok("No dead layers detected")

if noisy_layers:
    warn(f"Large weight magnitudes ({len(noisy_layers)} layers):")
    for l, v in noisy_layers[:5]:
        warn(f"  • {l}  max={v:.2f}")
else:
    ok("Weight magnitudes are healthy (all < 100)")

# Overall weight stats
all_weights = torch.cat([p.data.float().flatten() for p in model.parameters()])
global_mean = all_weights.mean().item()
global_std  = all_weights.std().item()
info(f"Global weight mean:  {global_mean:.6f}  (ideal ≈ 0.0)")
info(f"Global weight std:   {global_std:.6f}  (ideal ≈ 0.02 for trained model)")

if abs(global_mean) > 0.1:
    warn("Mean is far from 0 — possible bias shift in training")
else:
    ok("Weight mean is close to 0 — healthy distribution")

if global_std < 0.005:
    warn("Std very low — possible underfitting or dead weights")
elif global_std > 1.0:
    warn("Std very high — possible exploding gradients during training")
else:
    ok(f"Weight std is in healthy range ({global_std:.4f})")


# ── 4. Loss / Perplexity estimation ───────────────────────────────────────────
section("4 · LOSS & PERPLEXITY ESTIMATE")

# Use a small synthetic batch for a quick forward pass
vocab_size = config.vocab_size
seq_len    = min(128, config.max_position_embeddings)
batch_size = 2

torch.manual_seed(42)
dummy_ids = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
input_ids = dummy_ids[:, :-1]
labels    = dummy_ids[:, 1:]

with torch.no_grad():
    outputs = model(input_ids=input_ids, labels=labels)
    loss    = outputs.loss.item() if hasattr(outputs, "loss") else float("nan")

perplexity = math.exp(loss) if loss < 100 else float("inf")

info(f"Random-baseline loss:        {math.log(vocab_size):.3f}  (untrained model)")
info(f"Model loss (synthetic data): {loss:.4f}")
info(f"Model perplexity:            {perplexity:,.1f}")

if perplexity < 50:
    ok(f"Perplexity {perplexity:.1f} — model has learned language structure!")
elif perplexity < 200:
    warn(f"Perplexity {perplexity:.1f} — model is partially trained, needs more steps")
else:
    err(f"Perplexity {perplexity:.1f} — model has not converged (random guessing is {vocab_size:,})")

# Grade
if train_loss:
    ppl_train = math.exp(train_loss) if train_loss < 100 else float("inf")
    info(f"Perplexity from saved loss:  {ppl_train:,.1f}")

    if ppl_train < 30:
        grade = "A — Excellent"
    elif ppl_train < 60:
        grade = "B — Good, ready for SFT alignment"
    elif ppl_train < 150:
        grade = "C — Fair, more pre-training recommended"
    else:
        grade = "D — Needs significant improvement"
    info(f"Training grade:              {grade}")


# ── 5. Layer-wise noise report ─────────────────────────────────────────────────
section("5 · LAYER-BY-LAYER NOISE REPORT")

print(f"\n  {'Layer':<45} {'Std':>8} {'Mean':>8} {'MaxAbs':>8} {'Status'}")
print(f"  {'─'*45} {'─'*8} {'─'*8} {'─'*8} {'─'*10}")

for name, s in list(layer_stats.items())[:20]:
    short = name[-44:] if len(name) > 44 else name
    status = "OK"
    if s["nan"] > 0: status = "NaN!"
    elif s["inf"] > 0: status = "Inf!"
    elif s["std"] < 1e-6: status = "Dead?"
    elif s["abs_max"] > 100: status = "Noisy!"
    color = GREEN if status == "OK" else (RED if "!" in status else YELLOW)
    print(f"  {short:<45} {s['std']:>8.5f} {s['mean']:>8.5f} {s['abs_max']:>8.4f} {color}{status}{RESET}")

if len(layer_stats) > 20:
    info(f"... and {len(layer_stats) - 20} more layers (all checked)")


# ── 6. Generation test ────────────────────────────────────────────────────────
section("6 · GENERATION QUALITY TEST")

tokenizer_path = "tokenizer_model"
if Path(tokenizer_path).exists():
    try:
        from vill.tokenizer import VillTokenizer
        tokenizer = VillTokenizer.from_pretrained(tokenizer_path)

        prompts = [
            "The meaning of artificial intelligence is",
            "Once upon a time in a kingdom far away",
            "The most important thing in machine learning is",
        ]

        for prompt in prompts:
            print(f"\n  Prompt: \"{prompt}\"")
            ids = torch.tensor([tokenizer.encode(prompt)])
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=60, temperature=0.8, top_p=0.95)
            text = tokenizer.decode(out[0].tolist())
            print(f"  Output: {text[:200]}")
            print()
    except Exception as e:
        warn(f"Generation test skipped: {e}")
else:
    warn("Tokenizer not found at 'tokenizer_model/' — skipping generation test")
    info("Train tokenizer with: python3 main.py train-tokenizer --files data/corpus.txt")


# ── 7. Improvement recommendations ────────────────────────────────────────────
section("7 · IMPROVEMENT RECOMMENDATIONS")

print()
print(f"  {'Issue':<35} {'Recommendation'}")
print(f"  {'─'*35} {'─'*35}")

recommendations = []

if has_nan or has_inf:
    recommendations.append(("NaN/Inf in weights", "Retrain with lower LR (1e-4) + gradient clipping 0.5"))

if dead_layers:
    recommendations.append(("Dead layers detected", f"Use higher LR warmup, check {dead_layers[0]}"))

if noisy_layers:
    recommendations.append(("Noisy/large weights", "Add weight decay 0.1 to AdamW optimizer"))

if train_loss and train_loss > 3.0:
    recommendations.append(("High training loss", "Train for more steps (100K+) with fresh data"))

if train_loss and train_loss < 1.5:
    recommendations.append(("Low loss (possible overfit)", "Add SFT alignment on diverse instruction data"))

# Always add general recs
recommendations += [
    ("Scale model", "Move to vill-small (1.5B) for better language understanding"),
    ("More data", "Add SlimPajama + StarCoder datasets for richer knowledge"),
    ("Alignment", "Run SFT on UltraChat-200K for instruction following"),
    ("RAG/Knowledge", "Add ChromaDB vector store for factual Q&A improvement"),
    ("Quantization", "Export to GGUF 4-bit for fast local inference via Ollama"),
]

for issue, rec in recommendations:
    print(f"  {YELLOW}{issue:<35}{RESET} {rec}")


# ── 8. Save report ────────────────────────────────────────────────────────────
section("8 · SAVING DIAGNOSTIC REPORT")

report = {
    "checkpoint": CHECKPOINT,
    "step": training_step,
    "parameters": total_params,
    "size_bf16_gb": size_bf16_gb,
    "has_nan": has_nan,
    "has_inf": has_inf,
    "dead_layers": dead_layers,
    "noisy_layers": [l for l, _ in noisy_layers],
    "global_weight_mean": global_mean,
    "global_weight_std": global_std,
    "synthetic_loss": loss,
    "synthetic_perplexity": perplexity,
    "train_loss": train_loss,
    "recommendations": [r[1] for r in recommendations],
}

report_path = "scripts/vill_diagnostic_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

ok(f"Full report saved to: {report_path}")

print(f"\n{BOLD}{'═'*60}{RESET}")
print(f"{BOLD}  DIAGNOSIS COMPLETE{RESET}")
print(f"{BOLD}{'═'*60}{RESET}\n")
