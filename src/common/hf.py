from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ModelSpec:
    model_name: str
    adapter_path: Optional[str]
    use_qlora: bool


def _infer_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    model_name: str,
    use_qlora: bool,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: str = "auto",
):
    dtype = torch_dtype or _infer_torch_dtype()
    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=dtype,
        quantization_config=quant_config,
    )


def apply_lora(
    model,
    target_modules: List[str],
    r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, config)


def prepare_for_training(model, use_qlora: bool):
    if use_qlora:
        return prepare_model_for_kbit_training(model)
    return model


def load_model_for_inference(spec: ModelSpec):
    model = load_base_model(spec.model_name, use_qlora=spec.use_qlora)
    if spec.adapter_path:
        model = PeftModel.from_pretrained(model, spec.adapter_path)
    model.eval()
    return model
