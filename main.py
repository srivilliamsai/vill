#!/usr/bin/env python3
"""
Vill -- Main Entry Point
---------------------------
Command-line interface for training, evaluation, and inference.

Usage:
    python main.py train --config vill-nano
    python main.py generate --checkpoint checkpoints/vill_step_50000.pt --prompt "Hello"
    python main.py info --config vill-nano
"""

import argparse
import logging
import sys

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vill")


def cmd_info(args):
    """Display model architecture information."""
    from vill.model.config import get_config

    config = get_config(args.config)
    model_from_config = None

    print(f"\nVill Model Configuration: {args.config}")
    print("=" * 60)
    print(f"  Vocabulary size:        {config.vocab_size:,}")
    print(f"  Hidden size:            {config.hidden_size:,}")
    print(f"  Layers:                 {config.num_hidden_layers}")
    print(f"  Attention heads:        {config.num_attention_heads}")
    print(f"  KV heads (GQA):         {config.num_key_value_heads}")
    print(f"  Head dimension:         {config.head_dim}")
    print(f"  Intermediate (FFN):     {config.intermediate_size:,}")
    print(f"  Max context:            {config.max_position_embeddings:,}")
    print(f"  RoPE theta:             {config.rope_theta:,.0f}")

    if config.is_moe:
        print(f"  MoE experts:            {config.num_experts}")
        print(f"  Active experts/token:   {config.num_experts_per_tok}")

    print(f"\n  Estimated parameters:   {config.num_parameters_estimate:,}")

    # Instantiate to get exact count
    from vill.model.transformer import VillForCausalLM
    model = VillForCausalLM(config)
    actual = model.count_parameters()
    print(f"  Actual parameters:      {actual:,}")
    print(f"  Model size (float32):   {actual * 4 / 1e9:.2f} GB")
    print(f"  Model size (bfloat16):  {actual * 2 / 1e9:.2f} GB")
    print()


def cmd_train(args):
    """Run pre-training."""
    from vill.model.config import get_config
    from vill.model.transformer import VillForCausalLM
    from vill.training import Trainer, TrainingConfig
    from vill.data import TextFileDataset, PretrainingDataset, create_dataloader

    config = get_config(args.config)
    model = VillForCausalLM(config)
    logger.info("Model created: %s (%d parameters)", args.config, model.count_parameters())

    # Load tokenizer
    from vill.tokenizer import VillTokenizer
    if args.tokenizer:
        tokenizer = VillTokenizer.from_pretrained(args.tokenizer)
    else:
        logger.error("Tokenizer required. Train one first with: python main.py train-tokenizer")
        sys.exit(1)

    # Create dataset
    if args.data_files:
        dataset = TextFileDataset(
            file_paths=args.data_files,
            tokenizer=tokenizer,
            seq_length=config.max_position_embeddings,
        )
    elif args.dataset:
        dataset = PretrainingDataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            seq_length=config.max_position_embeddings,
        )
    else:
        logger.error("Provide --data-files or --dataset")
        sys.exit(1)

    loader = create_dataloader(dataset, batch_size=args.batch_size)

    train_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )

    trainer = Trainer(model, loader, train_config)
    trainer.train()


def cmd_train_tokenizer(args):
    """Train a BPE tokenizer."""
    from vill.tokenizer import VillTokenizer
    VillTokenizer.train(
        files=args.files,
        vocab_size=args.vocab_size,
        output_dir=args.output_dir,
    )


def cmd_generate(args):
    """Generate text from a trained model."""
    from vill.model.config import VillConfig
    from vill.model.transformer import VillForCausalLM
    from vill.tokenizer import VillTokenizer

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = VillConfig(**checkpoint["config"])
    model = VillForCausalLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokenizer = VillTokenizer.from_pretrained(args.tokenizer)
    input_ids = torch.tensor([tokenizer.encode(args.prompt)], device=device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    text = tokenizer.decode(output_ids[0].tolist())
    print(text)


def main():
    parser = argparse.ArgumentParser(description="Vill Language Model")
    sub = parser.add_subparsers(dest="command")

    # info
    p_info = sub.add_parser("info", help="Show model architecture details")
    p_info.add_argument("--config", default="vill-nano", help="Config preset name")

    # train-tokenizer
    p_tok = sub.add_parser("train-tokenizer", help="Train a BPE tokenizer")
    p_tok.add_argument("--files", nargs="+", required=True, help="Text files for training")
    p_tok.add_argument("--vocab-size", type=int, default=32000)
    p_tok.add_argument("--output-dir", default="tokenizer_model")

    # train
    p_train = sub.add_parser("train", help="Pre-train the model")
    p_train.add_argument("--config", default="vill-nano")
    p_train.add_argument("--tokenizer", default="tokenizer_model")
    p_train.add_argument("--data-files", nargs="*", help="Local text files")
    p_train.add_argument("--dataset", help="HuggingFace dataset name")
    p_train.add_argument("--batch-size", type=int, default=4)
    p_train.add_argument("--grad-accum", type=int, default=8)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--max-steps", type=int, default=50000)
    p_train.add_argument("--output-dir", default="checkpoints")
    p_train.add_argument("--resume", help="Resume from checkpoint path")

    # generate
    p_gen = sub.add_parser("generate", help="Generate text")
    p_gen.add_argument("--checkpoint", required=True)
    p_gen.add_argument("--tokenizer", default="tokenizer_model")
    p_gen.add_argument("--prompt", required=True)
    p_gen.add_argument("--max-tokens", type=int, default=256)
    p_gen.add_argument("--temperature", type=float, default=0.8)
    p_gen.add_argument("--top-p", type=float, default=0.95)

    args = parser.parse_args()

    if args.command == "info":
        cmd_info(args)
    elif args.command == "train-tokenizer":
        cmd_train_tokenizer(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "generate":
        cmd_generate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
