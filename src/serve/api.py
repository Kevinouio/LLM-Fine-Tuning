import argparse
import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from src.serve.models import load_model


class ParseRequest(BaseModel):
    input_text: str


class ParseResponse(BaseModel):
    output_json: Dict[str, Any]


class PlanRequest(BaseModel):
    goal: str
    state: str = ""
    tools: List[str] = []


class PlanResponse(BaseModel):
    output_json: Dict[str, Any]


def create_app(project: str, model_name: str, adapter_path: Optional[str]) -> FastAPI:
    app = FastAPI()
    bundle = load_model(project=project, model_name=model_name, adapter_path=adapter_path)

    @app.post("/parse", response_model=ParseResponse)
    def parse(request: ParseRequest) -> ParseResponse:
        output = bundle.parse_paper(request.input_text)
        return ParseResponse(output_json=output)

    @app.post("/plan", response_model=PlanResponse)
    def plan(request: PlanRequest) -> PlanResponse:
        output = bundle.plan(request.goal, request.state, request.tools)
        return PlanResponse(output_json=output)

    return app


def _env(name: str, default: Optional[str]) -> Optional[str]:
    value = os.getenv(name)
    return value if value is not None else default


app = create_app(
    project=_env("PROJECT", "paper_parser"),
    model_name=_env("MODEL_NAME", "google/gemma-2b"),
    adapter_path=_env("ADAPTER_PATH", None),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run API server.")
    parser.add_argument("--project", default="paper_parser")
    parser.add_argument("--model_name", default="google/gemma-2b")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    local_app = create_app(args.project, args.model_name, args.adapter_path)
    uvicorn.run(local_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
