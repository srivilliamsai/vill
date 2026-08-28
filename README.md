# Vill

A decoder-only large language model built from scratch using modern Transformer architecture. Vill implements the same core components found in production language models including Grouped-Query Attention, Rotary Positional Embeddings, SwiGLU activation, RMSNorm, and optional Mixture of Experts routing.

## Architecture

Vill follows a decoder-only Transformer design with the following components:

| Component | Description |
|-----------|-------------|
| Positional Encoding | Rotary Positional Embeddings (RoPE) with configurable base frequency |
| Attention | Grouped-Query Attention (GQA) with configurable KV head ratio |
| Feed-Forward | SwiGLU gated activation with three-matrix projection |
| Normalization | RMSNorm (pre-normalization at each sub-layer) |
| Experts | Optional Mixture of Experts with top-K routing and load balancing |

### Model Configurations

| Variant | Parameters | Layers | Heads | KV Heads | Context | Target Hardware |
|---------|-----------|--------|-------|----------|---------|-----------------|
| vill-nano | 150M | 12 | 12 | 4 | 2048 | Apple M1 / Single GPU |
| vill-small | 1.5B | 24 | 16 | 4 | 4096 | Single A100 or T4 |
| vill-medium | 7B | 32 | 32 | 8 | 8192 | 4x A100 |
| vill-large-moe | 70B (13B active) | 48 | 32 | 8 | 32768 | TPU pod or GPU cluster |

## Project Structure

```
vill/
    __init__.py
    model/
        __init__.py
        config.py           Model configuration and presets
        components.py        RMSNorm, RoPE, GQA, SwiGLU, MoE
        transformer.py       TransformerBlock, VillModel, VillForCausalLM
    tokenizer.py             BPE tokenizer (training and inference)
    data.py                  Streaming data pipeline
    training.py              Pre-training loop with mixed precision
    alignment.py             SFT and DPO post-training
    export.py                Model export and Ollama integration
tests/
    test_model.py            Architecture and forward pass tests
main.py                      Command-line interface
```

## Requirements

- Python 3.9 or later
- PyTorch 2.1 or later

Install dependencies:

```
pip install -e .
```

## Usage

### View Model Information

```
python main.py info --config vill-nano
```

Output:

```
Vill Model Configuration: vill-nano
============================================================
  Vocabulary size:        32,000
  Hidden size:            768
  Layers:                 12
  Attention heads:        12
  KV heads (GQA):         4
  Head dimension:         64
  Intermediate (FFN):     2,048
  Max context:            2,048

  Actual parameters:      ~150,000,000
  Model size (bfloat16):  ~0.28 GB
```

### Train a Tokenizer

```
python main.py train-tokenizer --files data/corpus.txt --vocab-size 32000 --output-dir tokenizer_model
```

### Pre-Train the Model

From local text files:

```
python main.py train \
    --config vill-nano \
    --tokenizer tokenizer_model \
    --data-files data/corpus.txt \
    --batch-size 4 \
    --grad-accum 8 \
    --lr 3e-4 \
    --max-steps 50000 \
    --output-dir checkpoints
```

From HuggingFace datasets (streaming, no download required):

```
python main.py train \
    --config vill-nano \
    --tokenizer tokenizer_model \
    --dataset HuggingFaceFW/fineweb-edu-sample \
    --batch-size 4 \
    --max-steps 50000
```

Resume from checkpoint:

```
python main.py train \
    --config vill-nano \
    --tokenizer tokenizer_model \
    --dataset HuggingFaceFW/fineweb-edu-sample \
    --resume checkpoints/vill_step_10000.pt
```

### Generate Text

```
python main.py generate \
    --checkpoint checkpoints/vill_step_50000.pt \
    --tokenizer tokenizer_model \
    --prompt "The fundamental principles of" \
    --max-tokens 256 \
    --temperature 0.8
```

### Deploy with Ollama

After exporting to GGUF format:

```
ollama create vill -f Modelfile
ollama run vill
```

## Training Pipeline

The full training pipeline consists of six stages:

1. **Tokenizer Training**: Train a Byte Pair Encoding tokenizer on a representative text corpus. Produces a vocabulary of subword tokens that balances compression efficiency with coverage of rare words.

2. **Architecture Instantiation**: Configure the Transformer model at the target scale. All configurations use identical code; only the hyperparameters differ between a 150M and a 70B model.

3. **Data Preparation**: Stream training data from HuggingFace Hub (FineWeb, SlimPajama, or custom datasets). Data is tokenized on-the-fly and packed into fixed-length sequences.

4. **Pre-Training**: Train the model on next-token prediction using AdamW with cosine learning rate decay, gradient accumulation, mixed precision (bfloat16), and gradient clipping. Checkpoints are saved at configurable intervals.

5. **Alignment**: Supervised Fine-Tuning (SFT) on instruction-response pairs followed by Direct Preference Optimization (DPO) to align the model with human preferences.

6. **Export and Deployment**: Convert the trained model to GGUF format for inference with Ollama or llama.cpp. Supports 4-bit and 8-bit quantization for efficient serving on consumer hardware.

## Training Data Sources

All datasets listed below are freely available:

| Dataset | Tokens | Description |
|---------|--------|-------------|
| FineWeb-Edu | 1.5T | High-quality educational web content |
| SlimPajama | 627B | Deduplicated web, books, code, Wikipedia |
| StarCoder | 250B | Permissively licensed source code |
| Common Corpus | 2T | Uncopyrighted multilingual text |

## Hardware Requirements by Scale

| Scale | VRAM Required | Estimated Training Time | Recommended Platform |
|-------|--------------|------------------------|---------------------|
| 150M (nano) | 4 GB | 2-4 days | Apple M1/M2, single consumer GPU |
| 1.5B (small) | 16 GB | 1-2 weeks | Google Colab T4, Kaggle |
| 7B (medium) | 80 GB (4x A100) | 2-4 weeks | Google Cloud (free credits) |
| 70B MoE (large) | 640 GB (8x A100) | 4-8 weeks | Google TPU Research Cloud |

## Running Tests

```
pip install -e ".[dev]"
pytest tests/ -v
```

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Shazeer, "GLU Variants Improve Transformer" (2020)
- Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models" (2022)
- Rafailov et al., "Direct Preference Optimization" (2023)

## License

Apache 2.0