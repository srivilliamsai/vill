"""
╔══════════════════════════════════════════════════════════╗
║      VILL LLM — STEP 2: SFT ALIGNMENT (Colab)          ║
║  Teaches vill-micro to follow instructions & chat       ║
╚══════════════════════════════════════════════════════════╝

Copy each CELL into a separate Colab cell.
Run them in order: Cell 1 → Cell 2 → Cell 3 → Cell 4
"""

# ════════════════════════════════════════════════════════
# CELL 1 — Setup + Mount Drive + Install
# ════════════════════════════════════════════════════════
CELL_1 = """
from google.colab import userdata, drive
import os, subprocess, sys

# HF Token
try:
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("✓ HF Token loaded")
except:
    print("⚠ No HF_TOKEN — continuing without it")

# Mount Drive
drive.mount('/content/drive')

# Repo setup
REPO = '/content/drive/MyDrive/Vill_AI/vill'
if os.path.exists(f'{REPO}/.git'):
    subprocess.run(['git', '-C', REPO, 'pull'], capture_output=True, text=True)
    print("✓ Code updated")
else:
    os.makedirs('/content/drive/MyDrive/Vill_AI', exist_ok=True)
    subprocess.run(['git', 'clone', 'https://github.com/srivilliamsai/vill.git', REPO])
    print("✓ Repo cloned")

os.chdir(REPO)
sys.path.insert(0, REPO)

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
    'torch', 'tokenizers', 'datasets', 'safetensors', 'tqdm', 'pyyaml', 'transformers'])

import torch
print(f"✓ GPU:  {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"✓ Dir:  {os.getcwd()}")
print()
print("═" * 50)
print("  STEP 2: Supervised Fine-Tuning (SFT)")
print("  Teaching Vill to follow instructions")
print("═" * 50)
"""

# ════════════════════════════════════════════════════════
# CELL 2 — Load Pre-trained Model + Download SFT Data
# ════════════════════════════════════════════════════════
CELL_2 = """
import torch, gc, json
from pathlib import Path
gc.collect(); torch.cuda.empty_cache()

from vill.model.config import VillConfig
from vill.model.transformer import VillForCausalLM
from vill.tokenizer import VillTokenizer

# ── Load pre-trained checkpoint ────────────────────────
CKPT = "checkpoints/vill_step_100000.pt"
print(f"Loading pre-trained model: {CKPT}")
ckpt_data = torch.load(CKPT, map_location="cpu", weights_only=False)

config = VillConfig(**ckpt_data["config"])
model  = VillForCausalLM(config)
model.load_state_dict(ckpt_data["model_state_dict"])
print(f"✓ Model loaded: {model.count_parameters():,} params")
print(f"  Step: {ckpt_data.get('global_step', '?'):,}")
print(f"  Loss: {ckpt_data.get('best_loss', '?')}")

del ckpt_data
gc.collect()

tokenizer = VillTokenizer.from_pretrained("tokenizer_model")
print(f"✓ Tokenizer loaded: vocab={tokenizer.vocab_size:,}")

# ── Download SFT instruction dataset ──────────────────
from datasets import load_dataset

print("\\nDownloading instruction dataset...")
# Using Alpaca-style instruction data — high quality, small enough for micro model
ds = load_dataset("tatsu-lab/alpaca", split="train")

# Convert to our format
sft_data = []
for row in ds:
    sft_data.append({
        "instruction": row["instruction"],
        "input": row.get("input", ""),
        "output": row["output"],
    })

print(f"✓ SFT dataset: {len(sft_data):,} instruction-response pairs")
print(f"  Example: {sft_data[0]['instruction'][:80]}...")
print(f"  Answer:  {sft_data[0]['output'][:80]}...")

# Also try chat-format data for variety
try:
    ds_chat = load_dataset("HuggingFaceH4/no_robots", split="train")
    chat_data = []
    for row in ds_chat:
        msgs = row.get("messages", [])
        if len(msgs) >= 2:
            chat_data.append({"messages": msgs})
    print(f"✓ Chat dataset: {len(chat_data):,} conversations")
    sft_data_chat = chat_data
except Exception as e:
    print(f"⚠ Chat dataset not available: {e}")
    sft_data_chat = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\\n✓ Ready for SFT on {device}")
"""

# ════════════════════════════════════════════════════════
# CELL 3 — Run SFT Training
# ════════════════════════════════════════════════════════
CELL_3 = """
import torch, math, time, shutil, logging
from pathlib import Path
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')

from vill.alignment import SFTDataset, AlignmentConfig

# ── Config ─────────────────────────────────────────────
sft_config = AlignmentConfig(
    learning_rate=2e-5,     # Much lower LR than pre-training (fine-tuning needs gentleness)
    num_epochs=3,           # 3 passes over the instruction data
    batch_size=4,           # Fits comfortably in T4's 15GB
    max_length=512,         # vill-micro's context window
    output_dir="checkpoints/sft",
)

# ── Build dataset ──────────────────────────────────────
print("Tokenizing instruction data...")
dataset = SFTDataset(sft_data, tokenizer, max_length=sft_config.max_length)
print(f"✓ {len(dataset):,} examples tokenized")

# Collate function for variable-length sequences
def collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    pad_id = tokenizer.pad_id if tokenizer.pad_id is not None else 0
    
    input_ids_padded = []
    labels_padded = []
    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids_padded.append(
            torch.cat([b["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        labels_padded.append(
            torch.cat([b["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )
    return {
        "input_ids": torch.stack(input_ids_padded),
        "labels":    torch.stack(labels_padded),
    }

loader = DataLoader(
    dataset,
    batch_size=sft_config.batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
)
print(f"✓ DataLoader: {len(loader):,} batches per epoch")

# ── Optimizer + Scheduler ──────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=sft_config.learning_rate,
    weight_decay=0.01,
    betas=(0.9, 0.999),
)

total_steps = len(loader) * sft_config.num_epochs
warmup_steps = min(100, total_steps // 10)

def get_lr(step):
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

# ── Training loop ──────────────────────────────────────
print(f"\\n{'='*50}")
print(f"  SFT TRAINING")
print(f"  Epochs:      {sft_config.num_epochs}")
print(f"  Batch size:  {sft_config.batch_size}")
print(f"  Total steps: {total_steps:,}")
print(f"  LR:          {sft_config.learning_rate}")
print(f"{'='*50}\\n")

model.train()
global_step = 0
best_loss = float("inf")

SFT_DIR = Path(sft_config.output_dir)
SFT_DIR.mkdir(parents=True, exist_ok=True)

# Drive backup dir
DRIVE_SFT = Path("/content/drive/MyDrive/Vill_AI/vill/checkpoints/sft")
DRIVE_SFT.mkdir(parents=True, exist_ok=True)

scaler = torch.amp.GradScaler("cuda")

for epoch in range(sft_config.num_epochs):
    epoch_loss = 0.0
    epoch_steps = 0
    t0 = time.time()
    
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        
        epoch_loss += loss.item()
        epoch_steps += 1
        global_step += 1
        
        if global_step % 50 == 0:
            avg = epoch_loss / epoch_steps
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  step={global_step:>5}  loss={avg:.4f}  lr={lr_now:.2e}  "
                  f"time={elapsed:.0f}s  [{epoch_steps}/{len(loader)}]")
    
    # End of epoch
    avg_loss = epoch_loss / max(epoch_steps, 1)
    elapsed = time.time() - t0
    print(f"\\n  ✓ Epoch {epoch+1}/{sft_config.num_epochs}  avg_loss={avg_loss:.4f}  time={elapsed:.0f}s")
    
    # Save checkpoint
    ckpt_path = SFT_DIR / f"vill_sft_epoch{epoch+1}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "sft_epoch": epoch + 1,
        "sft_loss": avg_loss,
        "pretrain_steps": 100000,
    }, ckpt_path)
    print(f"  ✓ Saved: {ckpt_path}")
    
    # Backup to Drive
    drive_path = DRIVE_SFT / ckpt_path.name
    if ckpt_path.resolve() != drive_path.resolve():
        shutil.copy2(ckpt_path, drive_path)
        print(f"  ✓ Backed up to Drive: {drive_path}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_path = SFT_DIR / "vill_sft_best.pt"
        shutil.copy2(ckpt_path, best_path)
        drive_best = DRIVE_SFT / "vill_sft_best.pt"
        if best_path.resolve() != drive_best.resolve():
            shutil.copy2(ckpt_path, drive_best)
        print(f"  ★ New best model! loss={best_loss:.4f}")
    
    print()

print("═" * 50)
print(f"  SFT COMPLETE!")
print(f"  Best loss: {best_loss:.4f}")
print(f"  Best model: checkpoints/sft/vill_sft_best.pt")
print("═" * 50)
"""

# ════════════════════════════════════════════════════════
# CELL 4 — Test the SFT Model (Chat with Vill!)
# ════════════════════════════════════════════════════════
CELL_4 = """
import torch
from pathlib import Path
from vill.model.config import VillConfig
from vill.model.transformer import VillForCausalLM
from vill.tokenizer import VillTokenizer

# Load best SFT model
ckpt = torch.load("checkpoints/sft/vill_sft_best.pt", map_location="cpu", weights_only=False)
config = VillConfig(**ckpt["config"])
model  = VillForCausalLM(config)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

tokenizer = VillTokenizer.from_pretrained("tokenizer_model")

print(f"✓ SFT Model loaded (Epoch {ckpt['sft_epoch']}, Loss {ckpt['sft_loss']:.4f})")
print(f"  Pre-training: {ckpt.get('pretrain_steps', '?'):,} steps")
print()

def chat(prompt, max_tokens=150):
    \"\"\"Chat with the SFT model.\"\"\"
    # Format as instruction
    text = f"{prompt}\\n"
    ids  = torch.tensor([tokenizer.encode(text, add_bos=True)], device=device)
    
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_tokens, temperature=0.7, top_p=0.9)
    
    response = tokenizer.decode(out[0].tolist())
    return response

# ── Test prompts ──────────────────────────────────────
print("=" * 60)
print("  CHATTING WITH VILL (SFT Model)")
print("=" * 60)

test_prompts = [
    "What is artificial intelligence?",
    "Explain photosynthesis in simple terms.",
    "Write a short poem about the moon.",
    "What are the benefits of exercise?",
    "How does a computer work?",
]

for i, prompt in enumerate(test_prompts, 1):
    response = chat(prompt)
    print(f"\\n{'─'*50}")
    print(f"  Q{i}: {prompt}")
    print(f"  A:  {response[:300]}")

print(f"\\n{'═'*60}")
print("  SFT testing complete!")
print("  Compare these answers to the pre-trained model's word salad.")
print(f"{'═'*60}")

# ── Interactive chat (optional) ────────────────────────
print("\\n💬 Interactive mode — type your questions (type 'quit' to exit):\\n")
while True:
    try:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit", "q"):
            break
        response = chat(user_input)
        print(f"Vill: {response[:400]}\\n")
    except (KeyboardInterrupt, EOFError):
        break

print("\\n✓ Chat session ended.")
"""

print("SFT Colab script ready!")
print("Copy CELL_1 through CELL_4 into separate Colab cells.")
print()
print("Timeline:")
print("  Cell 1: Setup (~1 min)")
print("  Cell 2: Load model + download data (~3 min)")
print("  Cell 3: SFT Training (~30-60 min)")
print("  Cell 4: Chat with Vill! 🎉")
