from __future__ import annotations

from typing import Any, Dict, List


def build_causal_lm_collator(tokenizer):
    def collate(features: List[Dict[str, Any]]):
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        if "labels" in batch:
            labels = batch["labels"].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    return collate
