"""
╔══════════════════════════════════════════════════════════╗
║         VILL LLM — COMPLETE COLAB NOTEBOOK v3.0         ║
║  All fixes included. Copy each cell into Colab.         ║
╚══════════════════════════════════════════════════════════╝
"""

# ════════════════════════════════════════════════════════
# CELL 1 — HF Token + Drive Mount + Repo Setup
# Run this FIRST every session
# ════════════════════════════════════════════════════════
CELL_1 = """
# ── Load HuggingFace Token from Colab Secrets ──────────
from google.colab import userdata, drive
import os, subprocess, sys

try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("✓ HF Token loaded")
except:
    print("⚠ No HF_TOKEN — continuing without it (still works)")

# ── Mount Google Drive ──────────────────────────────────
drive.mount('/content/drive')

# ── Clone or update repo ────────────────────────────────
REPO = '/content/drive/MyDrive/Vill_AI/vill'

if os.path.exists(f'{REPO}/.git'):
    result = subprocess.run(['git', '-C', REPO, 'pull'], capture_output=True, text=True)
    print("✓ Code updated:", result.stdout.strip() or "already up to date")
else:
    os.makedirs('/content/drive/MyDrive/Vill_AI', exist_ok=True)
    subprocess.run(['git', 'clone', 'https://github.com/srivilliamsai/vill.git', REPO])
    print("✓ Repo cloned")

os.chdir(REPO)
sys.path.insert(0, REPO)

# ── Install dependencies ────────────────────────────────
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'tokenizers', 'datasets', 'safetensors', 'tqdm', 'pyyaml'])

import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "❌ NO GPU"
vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

print(f"✓ GPU:  {gpu}")
print(f"✓ VRAM: {vram:.1f} GB")
print(f"✓ Dir:  {os.getcwd()}")
"""

# ════════════════════════════════════════════════════════
# CELL 2 — Train Tokenizer
# Run ONLY if tokenizer_model/ does not exist
# ════════════════════════════════════════════════════════
CELL_2 = """
from pathlib import Path

if Path("tokenizer_model").exists():
    print("✓ Tokenizer already exists — skip this cell")
else:
    print("Downloading data for tokenizer training...")
    from scripts.download_data import download_sample
    download_sample(output_dir="data/raw", num_samples=50000)

    print("Training BPE tokenizer (vocab=32000)...")
    from vill.tokenizer import VillTokenizer
    VillTokenizer.train(
        files=["data/raw/corpus.txt"],
        vocab_size=32000,
        output_dir="tokenizer_model",
    )
    print("✓ Tokenizer saved to tokenizer_model/")
"""

# ════════════════════════════════════════════════════════
# CELL 3 — START TRAINING  ← Main cell
# Resumes automatically from latest checkpoint
# Saves every 2500 steps directly to Google Drive
# ════════════════════════════════════════════════════════
CELL_3 = """
import torch, gc
from pathlib import Path
gc.collect()
torch.cuda.empty_cache()

from vill.model.config import get_config
from vill.model.transformer import VillForCausalLM
from vill.training import Trainer, TrainingConfig
from vill.data import PretrainingDataset, create_dataloader
from vill.tokenizer import VillTokenizer

# ── Auto-find latest checkpoint ────────────────────────
CKPT_DIR = "checkpoints"
ckpts = sorted(
    Path(CKPT_DIR).glob("vill_step_*.pt"),
    key=lambda p: int(p.stem.split("_")[-1])
)
resume_path = str(ckpts[-1]) if ckpts else None
resume_step = int(ckpts[-1].stem.split("_")[-1]) if ckpts else 0
print(f"▶ Resume from: {resume_path or 'fresh start'} (step {resume_step:,})")

# ── Load tokenizer ─────────────────────────────────────
tokenizer = VillTokenizer.from_pretrained("tokenizer_model")
print(f"✓ Tokenizer: vocab={tokenizer.vocab_size:,}")

# ── Build model (MUST match your checkpoint architecture)
config = get_config("vill-micro")       # ← vill-micro=56M matches your checkpoints
model  = VillForCausalLM(config)
params = model.count_parameters()
print(f"✓ Model: vill-micro | {params:,} params | ~{params*2/1e9:.2f} GB (bfloat16)")

# ── Dataset (confirmed working) ─────────────────────────
dataset = PretrainingDataset(
    dataset_name="HuggingFaceFW/fineweb-edu",   # ✅ confirmed working
    tokenizer=tokenizer,
    seq_length=config.max_position_embeddings,   # 512 for vill-micro
)
loader = create_dataloader(dataset, batch_size=4)
print(f"✓ Dataset: HuggingFaceFW/fineweb-edu (streaming)")

# ── Training config ─────────────────────────────────────
train_cfg = TrainingConfig(
    learning_rate=3e-4,
    min_learning_rate=3e-5,
    batch_size=4,
    gradient_accumulation_steps=8,      # effective batch = 32
    max_steps=100000,                    # train to 100K total steps
    warmup_steps=500,
    log_interval=50,                     # print loss every 50 steps
    save_interval=2500,                  # save checkpoint every 2500 steps
    output_dir=CKPT_DIR,
    backup_dir="/content/drive/MyDrive/Vill_AI/vill/checkpoints",  # auto Drive save
    resume_from=resume_path,
    use_amp=True,
    dtype="bfloat16",
)

remaining = train_cfg.max_steps - resume_step
print(f"\\n{'='*50}")
print(f"  Start step:    {resume_step:,}")
print(f"  Target steps:  {train_cfg.max_steps:,}")
print(f"  Remaining:     {remaining:,} steps")
print(f"  Save every:    {train_cfg.save_interval:,} steps → Drive")
print(f"  Eff. batch:    {train_cfg.batch_size * train_cfg.gradient_accumulation_steps}")
print(f"{'='*50}\\n")

Trainer(model, loader, train_cfg).train()
print("\\n✓ Training complete!")
"""

# ════════════════════════════════════════════════════════
# CELL 4 — Check Quality (Perplexity + Generation)
# Run anytime to measure how good the model is
# ════════════════════════════════════════════════════════
CELL_4 = """
import torch, math
from pathlib import Path
from vill.model.config import VillConfig
from vill.model.transformer import VillForCausalLM
from vill.tokenizer import VillTokenizer

# Load latest checkpoint
ckpts = sorted(Path("checkpoints").glob("vill_step_*.pt"),
               key=lambda p: int(p.stem.split("_")[-1]))
if not ckpts:
    print("❌ No checkpoints found!")
else:
    ckpt_path = str(ckpts[-1])
    print(f"Loading: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    config = VillConfig(**ckpt["config"])
    model  = VillForCausalLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenizer = VillTokenizer.from_pretrained("tokenizer_model")
    step = ckpt.get("global_step", "?")
    toks = ckpt.get("tokens_processed", 0)
    print(f"✓ Step: {step:,}  |  Tokens seen: {toks:,}  |  Params: {model.count_parameters():,}\\n")

    # ── Perplexity ─────────────────────────────────────
    val = "Artificial intelligence is the simulation of human intelligence in machines that are programmed to think and learn."
    ids = torch.tensor([tokenizer.encode(val)], device=device)
    with torch.no_grad():
        loss = model(input_ids=ids[:, :-1], labels=ids[:, 1:])["loss"].item()
    ppl = math.exp(loss)

    grade = (
        "A+ — GPT-2 level!"        if ppl < 30  else
        "A  — Ready for SFT"       if ppl < 60  else
        "B  — Keep training"       if ppl < 150 else
        "C  — Needs more steps"
    )
    print(f"  Loss:        {loss:.4f}")
    print(f"  Perplexity:  {ppl:.1f}")
    print(f"  Grade:       {grade}\\n")

    # ── Generation test ────────────────────────────────
    print("─" * 50)
    print("GENERATION SAMPLES")
    print("─" * 50)
    prompts = [
        "Artificial intelligence is",
        "Once upon a time",
        "The best way to learn is",
    ]
    for p in prompts:
        ids_in = torch.tensor([tokenizer.encode(p)], device=device)
        with torch.no_grad():
            out = model.generate(ids_in, max_new_tokens=80, temperature=0.8, top_p=0.95)
        text = tokenizer.decode(out[0].tolist())
        print(f"Prompt: {p}")
        print(f"Output: {text[:250]}")
        print()
"""

# ════════════════════════════════════════════════════════
# CELL 5 — Export to GGUF (Run after training is done)
# For running Vill locally with Ollama
# ════════════════════════════════════════════════════════
CELL_5 = """
import subprocess
subprocess.run(["pip", "install", "-q", "gguf"])

from pathlib import Path
ckpts = sorted(Path("checkpoints").glob("vill_step_*.pt"),
               key=lambda p: int(p.stem.split("_")[-1]))
best = str(ckpts[-1])
print(f"Exporting: {best}")

from vill.export import export_gguf
export_gguf(
    checkpoint_path=best,
    tokenizer_dir="tokenizer_model",
    output_path="vill_model.gguf",
    quantization="q4_k_m",
)

# Copy to Drive for safekeeping
import shutil
dest = "/content/drive/MyDrive/Vill_AI/vill_model.gguf"
shutil.copy2("vill_model.gguf", dest)
print(f"✓ GGUF exported and saved to Drive: {dest}")
print("\\nOn your Mac, run:")
print("  ollama create vill -f Modelfile")
print("  ollama run vill")
"""

print("Cells defined. See colab_final.py for full code.")
print("Copy CELL_1 through CELL_5 into separate Colab cells.")
