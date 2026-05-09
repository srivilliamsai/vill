"""
╔══════════════════════════════════════════════════════════════╗
║         VILL LLM TRAINING — GOOGLE COLAB v2.0               ║
║  Fixes: vill-micro config, Drive auto-backup, streaming      ║
║         HuggingFace data (no disk needed), perplexity log    ║
╚══════════════════════════════════════════════════════════════╝

PASTE EACH SECTION INTO A SEPARATE COLAB CELL.
"""

# ════════════════════════════════════════════════════════
# CELL 1 — Mount Drive & Clone Repo
# ════════════════════════════════════════════════════════
"""
from google.colab import drive
drive.mount('/content/drive')

import os, subprocess, sys

DRIVE_DIR = '/content/drive/MyDrive/Vill_AI'
os.makedirs(DRIVE_DIR, exist_ok=True)

REPO_DIR = f'{DRIVE_DIR}/vill'
if not os.path.exists(REPO_DIR):
    subprocess.run(['git', 'clone', 'https://github.com/srivilliamsai/vill.git', REPO_DIR])

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# Install dependencies
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'tokenizers', 'datasets', 'safetensors', 'tqdm', 'pyyaml'])

import torch
print(f"✓ Drive mounted")
print(f"✓ Working dir: {os.getcwd()}")
print(f"✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (no GPU!)'}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")
"""


# ════════════════════════════════════════════════════════
# CELL 2 — Download Data & Train Tokenizer
# (Skip this cell if tokenizer_model/ already exists)
# ════════════════════════════════════════════════════════
"""
import os
from pathlib import Path

TOKENIZER_DIR = "tokenizer_model"

if Path(TOKENIZER_DIR).exists():
    print("✓ Tokenizer already exists — skipping download")
else:
    print("Downloading data from FineWeb-Edu (streaming, no full download)...")
    from scripts.download_data import download_sample
    download_sample(output_dir="data/raw", num_samples=50000)

    print("Training tokenizer...")
    from vill.tokenizer import VillTokenizer
    VillTokenizer.train(
        files=["data/raw/corpus.txt"],
        vocab_size=32000,
        output_dir=TOKENIZER_DIR,
    )
    print(f"✓ Tokenizer saved to {TOKENIZER_DIR}/")
"""


# ════════════════════════════════════════════════════════
# CELL 3 — Configure & Start Training
# ════════════════════════════════════════════════════════
"""
import torch, gc, os, math, time
from pathlib import Path

gc.collect()
torch.cuda.empty_cache()

# ── Imports ───────────────────────────────────────────
from vill.model.config import get_config
from vill.model.transformer import VillForCausalLM
from vill.training import Trainer, TrainingConfig
from vill.data import PretrainingDataset, TextFileDataset, create_dataloader
from vill.tokenizer import VillTokenizer

# ── Settings — change these as needed ────────────────
CONFIG_NAME    = "vill-nano"    # vill-micro=56M  vill-nano=124M  vill-small=1.5B
CHECKPOINT_DIR = "checkpoints"  # local Colab storage (fast write)
DRIVE_BACKUP   = "/content/drive/MyDrive/Vill_AI/vill/checkpoints"  # ← AUTO SAVE TO DRIVE
MAX_STEPS      = 100000         # extend beyond 50K for better quality
BATCH_SIZE     = 4
GRAD_ACCUM     = 8              # effective batch = 4*8 = 32 sequences
LEARNING_RATE  = 3e-4
SAVE_EVERY     = 2500           # save checkpoint every N steps
USE_STREAMING  = True           # True = HuggingFace streaming (no disk needed)

# ── Auto-detect best checkpoint to resume ─────────────
def find_latest_checkpoint(ckpt_dir):
    ckpts = sorted(Path(ckpt_dir).glob("vill_step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[-1]))
    return str(ckpts[-1]) if ckpts else None

resume_path = find_latest_checkpoint(CHECKPOINT_DIR)
if resume_path:
    step_num = int(Path(resume_path).stem.split("_")[-1])
    print(f"▶ Resuming from step {step_num:,}  ({resume_path})")
else:
    print("▶ Starting fresh training (no checkpoint found)")

# ── Load tokenizer ─────────────────────────────────────
tokenizer = VillTokenizer.from_pretrained("tokenizer_model")
print(f"✓ Tokenizer loaded (vocab={tokenizer.vocab_size:,})")

# ── Build model ────────────────────────────────────────
config = get_config(CONFIG_NAME)
model  = VillForCausalLM(config)
params = model.count_parameters()
vram_needed = params * 2 / 1e9  # bfloat16
print(f"✓ Model: {CONFIG_NAME}  |  {params:,} params  |  ~{vram_needed:.2f} GB VRAM (bfloat16)")

# ── Build dataset ──────────────────────────────────────
if USE_STREAMING:
    # Streams directly from HuggingFace — NO disk usage!
    dataset = PretrainingDataset(
        dataset_name="HuggingFaceFW/fineweb-edu-sample",
        tokenizer=tokenizer,
        seq_length=config.max_position_embeddings,
    )
    print("✓ Streaming dataset: HuggingFaceFW/fineweb-edu-sample")
else:
    # Use local corpus if it exists
    dataset = TextFileDataset(
        file_paths=["data/raw/corpus.txt"],
        tokenizer=tokenizer,
        seq_length=config.max_position_embeddings,
    )
    print("✓ Local dataset: data/raw/corpus.txt")

loader = create_dataloader(dataset, batch_size=BATCH_SIZE)

# ── Training config ────────────────────────────────────
train_cfg = TrainingConfig(
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    warmup_steps=500,
    log_interval=50,
    save_interval=SAVE_EVERY,
    output_dir=CHECKPOINT_DIR,
    backup_dir=DRIVE_BACKUP,        # ← AUTO-SAVES TO GOOGLE DRIVE!
    resume_from=resume_path,
    use_amp=True,
    dtype="bfloat16",
)

print(f"\n{'='*55}")
print(f"  Config:           {CONFIG_NAME}")
print(f"  Parameters:       {params:,}")
print(f"  Max Steps:        {MAX_STEPS:,}")
print(f"  Resume From:      {resume_path or 'scratch'}")
print(f"  Effective Batch:  {BATCH_SIZE * GRAD_ACCUM} sequences")
print(f"  Checkpoint every: {SAVE_EVERY:,} steps")
print(f"{'='*55}\n")

# ── Start training ─────────────────────────────────────
trainer = Trainer(model, loader, train_cfg)
trainer.train()
print("✓ Training complete!")
"""


# ════════════════════════════════════════════════════════
# CELL 4 — Backup Checkpoints to Google Drive
# Run this cell anytime to sync your checkpoints safely
# ════════════════════════════════════════════════════════
"""
import shutil, os
from pathlib import Path

SRC  = Path("checkpoints")
DEST = Path("/content/drive/MyDrive/Vill_AI/vill/checkpoints")
DEST.mkdir(parents=True, exist_ok=True)

copied = 0
for f in sorted(SRC.glob("vill_step_*.pt")):
    dest_file = DEST / f.name
    if not dest_file.exists():
        shutil.copy2(f, dest_file)
        print(f"  ✓ Backed up: {f.name}")
        copied += 1
    else:
        print(f"  – Already in Drive: {f.name}")

print(f"\n✓ Backup done. {copied} new file(s) copied to Google Drive.")
"""


# ════════════════════════════════════════════════════════
# CELL 5 — Check Model Quality (Perplexity + Generation)
# Run after training to measure how good your model is
# ════════════════════════════════════════════════════════
"""
import torch, math
from pathlib import Path
from vill.model.config import VillConfig
from vill.model.transformer import VillForCausalLM
from vill.tokenizer import VillTokenizer

# Find best checkpoint
ckpts = sorted(Path("checkpoints").glob("vill_step_*.pt"),
               key=lambda p: int(p.stem.split("_")[-1]))
BEST_CKPT = str(ckpts[-1])
print(f"Loading: {BEST_CKPT}")

ckpt = torch.load(BEST_CKPT, map_location="cpu")
config = VillConfig(**ckpt["config"])
model  = VillForCausalLM(config)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer = VillTokenizer.from_pretrained("tokenizer_model")
step = ckpt.get("global_step", "?")
print(f"✓ Model loaded  (step={step:,}  |  {model.count_parameters():,} params)\n")

# ── Perplexity on a fixed validation sentence ──────────
val_text = "The history of artificial intelligence began in the 1950s when scientists first began to explore the possibility of creating machines that could think and reason like humans."
ids = torch.tensor([tokenizer.encode(val_text)], device=device)

with torch.no_grad():
    out = model(input_ids=ids[:, :-1], labels=ids[:, 1:])
    loss = out["loss"].item()

ppl = math.exp(loss)
grade = (
    "A+ — Excellent (GPT-2 level)" if ppl < 30 else
    "A  — Very Good, ready for SFT" if ppl < 60 else
    "B  — Learning, needs more steps" if ppl < 150 else
    "C  — Early stage, keep training"
)
print(f"  Loss:        {loss:.4f}")
print(f"  Perplexity:  {ppl:.1f}")
print(f"  Grade:       {grade}\n")

# ── Generation test ────────────────────────────────────
prompts = [
    "Artificial intelligence is",
    "Once upon a time",
    "The best way to learn programming is",
]

print("─" * 55)
print("GENERATION SAMPLES")
print("─" * 55)
for prompt in prompts:
    ids_in = torch.tensor([tokenizer.encode(prompt)], device=device)
    with torch.no_grad():
        out_ids = model.generate(ids_in, max_new_tokens=80, temperature=0.8, top_p=0.95)
    text = tokenizer.decode(out_ids[0].tolist())
    print(f"Prompt : {prompt}")
    print(f"Output : {text[:300]}")
    print()
"""


# ════════════════════════════════════════════════════════
# CELL 6 — Export to GGUF (Run after training is done)
# Lets you run the model locally with Ollama / llama.cpp
# ════════════════════════════════════════════════════════
"""
from pathlib import Path

BEST_CKPT = str(sorted(Path("checkpoints").glob("vill_step_*.pt"),
                        key=lambda p: int(p.stem.split("_")[-1]))[-1])

print(f"Exporting {BEST_CKPT} to GGUF...")

import subprocess
subprocess.run(["pip", "install", "-q", "gguf"])

from vill.export import export_gguf
export_gguf(
    checkpoint_path=BEST_CKPT,
    tokenizer_dir="tokenizer_model",
    output_path="vill_model.gguf",
    quantization="q4_k_m",  # 4-bit quantization — 75% smaller
)
print("✓ Exported: vill_model.gguf")
print("Copy to local machine and run: ollama create vill -f Modelfile")
"""
