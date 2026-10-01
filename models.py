"""
models.py
=========
Tokenizer loading, 4-bit NF4 quantized base model loading, and LoRA/PEFT
configuration shared across all training stages.
"""

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import get_attn_impl

# LoRA target module names are NOT consistent across architectures -- Phi
# names its output projection "dense", while Qwen2/Llama-family models name
# the equivalent layer "o_proj". Hardcoding Phi's names here would silently
# skip that layer for any other backbone (peft does not error on a
# target_modules entry that matches nothing; it just adapts fewer layers
# than intended), which would confound a cross-model comparison (review
# comment #2) rather than isolate FC-KD's effect. Add new architectures here
# as they're evaluated.
LORA_TARGET_MODULES_BY_ARCHITECTURE = {
    "phi": ["q_proj", "k_proj", "v_proj", "dense"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj"],
}


def _infer_architecture_family(model_name_or_path: str) -> str:
    name = model_name_or_path.lower()
    if "phi" in name:
        return "phi"
    if "qwen" in name:
        return "qwen2"
    if "llama" in name or "tinyllama" in name:
        return "llama"
    raise ValueError(
        f"Unrecognized model family for '{model_name_or_path}' -- add its LoRA "
        f"target_modules to LORA_TARGET_MODULES_BY_ARCHITECTURE in models.py "
        f"before training on it. Guessing wrong here silently under-adapts the "
        f"model rather than raising an error, which is worse than failing loudly."
    )


def get_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_quantized_base_model(model_name: str):
    """Loads a base causal LM in 4-bit NF4 precision with bf16 compute dtype.

    Note on attention implementation: flash_attention_2 is selected
    automatically when the host GPU supports it (compute capability >= 8.0),
    falling back to eager attention otherwise. This affects throughput only,
    not numerical results, and is applied consistently across all three
    training stages.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=get_attn_impl(),
        trust_remote_code=True,
    )
    return model


def setup_peft_model(base_model, model_name_or_path: str = None):
    """
    model_name_or_path: REQUIRED in practice (kept optional only for backward
        compatibility with any external callers) -- used to select the
        correct target_modules for this architecture. If omitted, falls back
        to Phi's naming, which is only correct for the original phi-1_5
        student; passing it explicitly is strongly recommended for any
        non-Phi backbone (review comment #2's cross-model validation).
    """
    base_model = prepare_model_for_kbit_training(base_model)
    if model_name_or_path is not None:
        family = _infer_architecture_family(model_name_or_path)
        target_modules = LORA_TARGET_MODULES_BY_ARCHITECTURE[family]
    else:
        target_modules = LORA_TARGET_MODULES_BY_ARCHITECTURE["phi"]
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
        bias="none",
    )
    return get_peft_model(base_model, peft_config)
