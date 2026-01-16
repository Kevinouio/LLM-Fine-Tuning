from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.paper_parser.infer import generate_stub_output as parse_stub
from src.planner.infer import generate_stub_output as plan_stub


@dataclass
class ModelBundle:
    project: str
    model_name: str
    adapter_path: Optional[str]

    def parse_paper(self, input_text: str) -> Dict[str, Any]:
        return parse_stub(input_text)

    def plan(self, goal: str, state: str, tools: List[str]) -> Dict[str, Any]:
        return plan_stub(goal, state, tools)


def load_model(project: str, model_name: str, adapter_path: Optional[str]) -> ModelBundle:
    return ModelBundle(project=project, model_name=model_name, adapter_path=adapter_path)
