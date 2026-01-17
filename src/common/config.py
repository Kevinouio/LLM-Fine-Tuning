from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    project: str
    model_name: str
    train_path: str
    val_path: Optional[str]
    output_dir: str
    seed: int = 42
    max_seq_len: int = 2048
    batch_size: int = 1
    gradient_accumulation: int = 1
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    use_qlora: bool = False
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[str] = None
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


@dataclass
class EvalConfig:
    project: str
    data_path: str
    predictions_path: Optional[str]
    schema_path: str
    output_path: str

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
