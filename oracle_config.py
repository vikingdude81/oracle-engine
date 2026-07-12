"""
Oracle Engine Configuration
===========================

Central configuration for model paths. Override with environment variables
so the repo works on any machine:

    export ORACLE_CUSTOM_MODEL_PATH=/path/to/your/finetuned/model
    export ORACLE_BASE_MODEL=unsloth/Qwen2.5-32B-Instruct-bnb-4bit

Defaults:
- ORACLE_CUSTOM_MODEL_PATH falls back to the HF Hub LoRA if unset and the
  local path does not exist.
- ORACLE_BASE_MODEL defaults to the 4-bit quantized Qwen2.5-32B-Instruct.
"""

import os
from pathlib import Path

# Local fine-tuned model (LoRA final checkpoint). This default matches the
# original training machine (WSL2); set ORACLE_CUSTOM_MODEL_PATH elsewhere.
_DEFAULT_LOCAL_MODEL = "/home/akbon/unsloth_train/outputs_stage3_code/final"

# Published LoRA on the HF Hub — used when no local checkpoint is available.
HUB_LORA_ID = "Vikingdude81/oracle-engine-32b-lora"

CUSTOM_MODEL_PATH = os.environ.get("ORACLE_CUSTOM_MODEL_PATH", _DEFAULT_LOCAL_MODEL)
BASE_MODEL_NAME = os.environ.get("ORACLE_BASE_MODEL", "unsloth/Qwen2.5-32B-Instruct-bnb-4bit")


def resolve_custom_model_path() -> str:
    """Return the custom model path, falling back to the HF Hub LoRA when the
    configured local path does not exist on this machine."""
    if Path(CUSTOM_MODEL_PATH).exists():
        return CUSTOM_MODEL_PATH
    return HUB_LORA_ID
