from __future__ import annotations

import json
from typing import Any, Dict

import torch


def extract_json(text: str) -> Dict[str, Any]:
    if "```" in text:
        for fence in ("```json", "```"):
            start = text.find(fence)
            if start == -1:
                continue
            start = text.find("\n", start)
            if start == -1:
                continue
            end = text.find("```", start)
            if end == -1:
                continue
            snippet = text[start:end].strip()
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                break

    start = text.find("{")
    if start == -1:
        return {"raw_text": text}

    depth = 0
    end = None
    for idx, char in enumerate(text[start:], start=start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break

    if end is None:
        end = text.rfind("}")
        if end == -1 or end <= start:
            return {"raw_text": text}

    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return {"raw_text": text}


def generate_json(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> Dict[str, Any]:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=getattr(model.config, "max_position_embeddings", 2048),
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]
    do_sample = temperature > 0
    gen_kwargs = {}
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
    completion_ids = generated[0][prompt_len:]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return extract_json(text)
