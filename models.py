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


def setup_peft_model(base_model):
    base_model = prepare_model_for_kbit_training(base_model)
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "dense"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    return get_peft_model(base_model, peft_config)
