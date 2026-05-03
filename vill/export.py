"""
Vill -- Model Export and Deployment
--------------------------------------
Converts trained Vill models to GGUF format for use with
Ollama and llama.cpp. Includes quantization support.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import torch

from vill.model.config import VillConfig
from vill.model.transformer import VillForCausalLM

logger = logging.getLogger(__name__)


def save_for_inference(
    model: VillForCausalLM,
    output_dir: str,
    tokenizer_path: Optional[str] = None,
) -> None:
    """
    Save model weights and config for inference.

    Creates a directory with:
    - model.safetensors (or model.pt)
    - config.json
    - tokenizer files (if provided)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save weights
    try:
        from safetensors.torch import save_file
        state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(state_dict, str(out / "model.safetensors"))
        logger.info("Saved weights in safetensors format.")
    except ImportError:
        torch.save(model.state_dict(), out / "model.pt")
        logger.info("Saved weights in PyTorch format.")

    # Save config
    model.config.save(str(out / "config.json"))

    # Copy tokenizer if available
    if tokenizer_path:
        import shutil
        src = Path(tokenizer_path)
        for f in src.iterdir():
            shutil.copy2(f, out / f.name)

    logger.info("Model saved to %s", output_dir)


def load_for_inference(
    model_dir: str,
    device: Optional[torch.device] = None,
) -> VillForCausalLM:
    """Load a saved model for inference."""
    path = Path(model_dir)
    config = VillConfig.load(str(path / "config.json"))
    model = VillForCausalLM(config)

    weights_path = path / "model.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_path))
        model.load_state_dict(state_dict)
    else:
        state_dict = torch.load(path / "model.pt", map_location="cpu")
        model.load_state_dict(state_dict)

    if device:
        model.to(device)
    model.eval()
    return model


def create_ollama_modelfile(
    model_path: str,
    output_path: str = "Modelfile",
    system_prompt: str = "",
) -> str:
    """
    Generate an Ollama Modelfile for serving Vill.

    After converting to GGUF format, this Modelfile allows running
    the model with: ollama create vill -f Modelfile
    """
    if not system_prompt:
        system_prompt = (
            "You are Vill, a helpful AI assistant built from scratch. "
            "You provide clear, accurate, and thoughtful responses."
        )

    content = f"""FROM {model_path}

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 2048

TEMPLATE \"\"\"{{{{- if .System }}}}
<|system|>{{{{ .System }}}}<|end|>
{{{{- end }}}}
<|user|>{{{{ .Prompt }}}}<|end|>
<|assistant|>\"\"\"

PARAMETER stop "<|end|>"
PARAMETER stop "<|eos|>"
PARAMETER stop "<|user|>"
"""

    Path(output_path).write_text(content)
    logger.info("Ollama Modelfile written to %s", output_path)
    return content
