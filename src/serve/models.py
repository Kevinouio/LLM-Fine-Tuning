from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.common.generation import generate_json
from src.common.hf import ModelSpec, load_model_for_inference, load_tokenizer
from src.paper_parser.prompts import build_prompt as build_paper_prompt
from src.planner.prompts import build_prompt as build_plan_prompt


@dataclass
class ModelBundle:
    project: str
    model_name: str
    adapter_path: Optional[str]
    model: Any
    tokenizer: Any

    def parse_paper(self, input_text: str) -> Dict[str, Any]:
        prompt = build_paper_prompt(input_text)
        return generate_json(self.model, self.tokenizer, prompt)

    def plan(self, goal: str, state: str, tools: List[str]) -> Dict[str, Any]:
        prompt = build_plan_prompt(goal, state, tools)
        return generate_json(self.model, self.tokenizer, prompt, max_new_tokens=128)


def load_model(project: str, model_name: str, adapter_path: Optional[str]) -> ModelBundle:
    tokenizer = load_tokenizer(model_name)
    model = load_model_for_inference(
        ModelSpec(model_name=model_name, adapter_path=adapter_path, use_qlora=False)
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    return ModelBundle(
        project=project,
        model_name=model_name,
        adapter_path=adapter_path,
        model=model,
        tokenizer=tokenizer,
    )
