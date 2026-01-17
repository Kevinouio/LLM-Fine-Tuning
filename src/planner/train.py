import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from src.common.config import TrainConfig
from src.common.collators import build_causal_lm_collator
from src.common.hf import apply_lora, load_base_model, load_tokenizer, prepare_for_training
from src.common.io import ensure_dir, read_jsonl, save_json
from src.common.logging import setup_logging
from src.common.seeds import set_seed
from src.planner.prompts import build_prompt


logger = setup_logging("planner.train")


REQUIRED_KEYS = {"id", "goal", "state", "tools", "output_json"}


def validate_records(records: List[Dict[str, Any]]) -> None:
    for index, record in enumerate(records):
        missing = REQUIRED_KEYS - set(record.keys())
        if missing:
            raise ValueError(f"Record {index} missing keys: {sorted(missing)}")


def dataset_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths = [len(str(r.get("goal", ""))) for r in records]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    return {"count": len(records), "avg_goal_length": avg_len}


def _format_example(record: Dict[str, Any]) -> Dict[str, str]:
    goal = record.get("goal", "")
    state = record.get("state", "")
    tools = record.get("tools", []) or []
    prompt = build_prompt(goal, state, tools)
    output_json = record.get("output_json", {})
    completion = json.dumps(output_json, ensure_ascii=True)
    return {"prompt": prompt, "completion": completion}


def _tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    max_seq_len: int,
) -> Dataset:
    def tokenize_row(row: Dict[str, str]) -> Dict[str, Any]:
        prompt = row["prompt"]
        completion = row["completion"]
        full_text = prompt + completion

        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )["input_ids"]
        full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=False,
        )

        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return dataset.map(tokenize_row, remove_columns=dataset.column_names)


def _default_target_modules(model_name: str) -> List[str]:
    if "gemma" in model_name.lower():
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    return ["q_proj", "v_proj"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train planner.")
    parser.add_argument("--train_path", default="data/planner/train.jsonl")
    parser.add_argument("--val_path", default="data/planner/val.jsonl")
    parser.add_argument("--output_dir", default="results/planner")
    parser.add_argument("--model_name", default="google/gemma-3-4b")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA adapters.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    args = parser.parse_args()

    set_seed(args.seed)

    train_records = read_jsonl(args.train_path)
    val_records = read_jsonl(args.val_path) if args.val_path else []
    validate_records(train_records)
    if val_records:
        validate_records(val_records)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_id
    ensure_dir(run_dir)

    use_lora = not args.no_lora
    target_modules: Optional[List[str]] = None
    if use_lora:
        if args.lora_target_modules:
            target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
        else:
            target_modules = _default_target_modules(args.model_name)

    config = TrainConfig(
        project="planner",
        model_name=args.model_name,
        train_path=args.train_path,
        val_path=args.val_path,
        output_dir=str(run_dir),
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        gradient_accumulation=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        use_qlora=args.use_qlora,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=",".join(target_modules) if target_modules else None,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
    )
    config.save(run_dir / "config.json")

    stats = {
        "train": dataset_stats(train_records),
        "val": dataset_stats(val_records),
    }
    save_json(run_dir / "dataset_stats.json", stats)

    tokenizer = load_tokenizer(args.model_name)
    model = load_base_model(args.model_name, use_qlora=args.use_qlora)
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_for_training(model, use_qlora=args.use_qlora)
    if use_lora and target_modules:
        model = apply_lora(
            model,
            target_modules=target_modules,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )

    train_dataset = Dataset.from_list([_format_example(r) for r in train_records])
    val_dataset = Dataset.from_list([_format_example(r) for r in val_records]) if val_records else None
    train_dataset = _tokenize_dataset(train_dataset, tokenizer, args.max_seq_len)
    if val_dataset is not None:
        val_dataset = _tokenize_dataset(val_dataset, tokenizer, args.max_seq_len)

    data_collator = build_causal_lm_collator(tokenizer)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=10,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if val_dataset is not None else None,
        evaluation_strategy="steps" if val_dataset is not None else "no",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting training run: %s", run_dir)
    trainer.train()
    trainer.save_model(run_dir / "adapter")
    tokenizer.save_pretrained(run_dir / "adapter")
    logger.info("Training complete. Adapter saved to %s", run_dir / "adapter")


if __name__ == "__main__":
    main()

