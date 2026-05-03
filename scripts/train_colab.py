#!/usr/bin/env python3
"""
Vill Training Notebook for Google Colab / Kaggle
--------------------------------------------------
Copy this into a Colab or Kaggle notebook cell to train Vill
on a free T4/P100 GPU (5-10x faster than M1).

Instructions:
  1. Go to colab.research.google.com or kaggle.com/notebooks
  2. Create a new notebook
  3. Set runtime to GPU (T4)
  4. Copy this entire file into a single cell
  5. Run it
"""

# -- Cell 1: Install and clone --
import subprocess, os

subprocess.run(["pip", "install", "-q", "torch", "tokenizers", "datasets", "safetensors", "tqdm", "pyyaml"], check=True)
subprocess.run(["git", "clone", "https://github.com/srivilliamsai/vill.git"], check=True)
os.chdir("vill")

print("Setup complete.")

# -- Cell 2: Download training data --
from scripts.download_data import download_sample
corpus_path = download_sample(output_dir="data/raw", num_samples=50000)
print(f"Corpus downloaded: {corpus_path}")

# -- Cell 3: Train tokenizer --
from vill.tokenizer import VillTokenizer

tokenizer = VillTokenizer.train(
    files=["data/raw/corpus.txt"],
    vocab_size=32000,
    output_dir="tokenizer_model",
)
print(f"Tokenizer trained: vocab_size={tokenizer.vocab_size}")

# Verify tokenizer
test = "Vill is a language model trained from scratch."
ids = tokenizer.encode(test)
print(f"Test encode: '{test}' -> {len(ids)} tokens")
print(f"Test decode: '{tokenizer.decode(ids)}'")

# -- Cell 4: Configure and start training --
import torch
from vill.model.config import get_config
from vill.model.transformer import VillForCausalLM
from vill.training import Trainer, TrainingConfig
from vill.data import TextFileDataset, create_dataloader

# On Colab/Kaggle T4 (16GB VRAM), we can train vill-nano (124M)
# On M1 8GB, use vill-micro instead
device_type = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"

if device_type == "cuda":
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_mem:.1f} GB)")
    if gpu_mem >= 14:
        config_name = "vill-nano"   # 124M, needs ~8GB VRAM
    else:
        config_name = "vill-micro"  # 56M, needs ~3GB VRAM
else:
    config_name = "vill-micro"
    print(f"Device: {device_type}")

print(f"Using config: {config_name}")

config = get_config(config_name)
model = VillForCausalLM(config)
print(f"Model parameters: {model.count_parameters():,}")

# Create dataset
dataset = TextFileDataset(
    file_paths=["data/raw/corpus.txt"],
    tokenizer=tokenizer,
    seq_length=config.max_position_embeddings,
)

# Training configuration -- tuned for T4 GPU
train_config = TrainingConfig(
    learning_rate=3e-4,
    min_learning_rate=3e-5,
    batch_size=8 if device_type == "cuda" else 2,
    gradient_accumulation_steps=4 if device_type == "cuda" else 16,
    max_steps=50000,
    log_interval=50,
    save_interval=2500,
    output_dir="checkpoints",
)

loader = create_dataloader(dataset, batch_size=train_config.batch_size)

# Start training
trainer = Trainer(model, loader, train_config)
trainer.train()

print("Training complete.")
print(f"Checkpoints saved in: checkpoints/")

# -- Cell 5: Test generation --
model.eval()
prompt = "The principles of artificial intelligence"
input_ids = torch.tensor([tokenizer.encode(prompt)], device=trainer.device)

with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.8)

generated = tokenizer.decode(output_ids[0].tolist())
print(f"\nPrompt: {prompt}")
print(f"Generated: {generated}")
