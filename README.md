# LLM-Fine-Tuning
This will be my own experimentation in Fine-Tuning LLM models, more specifically the Gemma model for now. Feel Free to use the code if you plan to Fine tune your own personal model!

A local, reproducible repo for fine-tuning **Gemma 3 4B** (LoRA/QLoRA) on targeted behaviors and evaluating results.

This repo contains **two projects**:

1. **Research Paper Parsing** — Paper text → quality check + structured JSON digest + high-level simplified explanation.
2. **Clarification-First Planner** — Goal + state → either a structured plan *or* a clarifying question when required info is missing.

> Primary goal: learn end-to-end fine-tuning workflows (data → train → eval → serve)

---

## Features

* ✅ Local training + inference (WSL/Linux)
* ✅ LoRA/QLoRA fine-tuning for Gemma 3 4B
* ✅ Deterministic evaluation for structured outputs (schema validation, grounding checks, execution checks where applicable)
* ✅ Simple API server for model inference
* ✅ Optional JavaScript frontend for interactive use

---

## Repo Layout

```
.
├─ docs/
│  ├─ design_research_paper_parser.md
│  ├─ design_clarification_first_planner.md
│  └─ notes.md
├─ data/
│  ├─ paper_parser/
│  │  ├─ raw/
│  │  ├─ processed/
│  │  ├─ train.jsonl
│  │  ├─ val.jsonl
│  │  └─ test.jsonl
│  └─ planner/
│     ├─ raw/
│     ├─ processed/
│     ├─ train.jsonl
│     ├─ val.jsonl
│     └─ test.jsonl
├─ schemas/
│  ├─ paper_digest.schema.json
│  └─ planner_output.schema.json
├─ src/
│  ├─ common/
│  │  ├─ config.py
│  │  ├─ io.py
│  │  ├─ logging.py
│  │  └─ seeds.py
│  ├─ paper_parser/
│  │  ├─ prompts.py
│  │  ├─ preprocess.py
│  │  ├─ train.py
│  │  ├─ infer.py
│  │  └─ eval.py
│  ├─ planner/
│  │  ├─ prompts.py
│  │  ├─ preprocess.py
│  │  ├─ train.py
│  │  ├─ infer.py
│  │  └─ eval.py
│  └─ serve/
│     ├─ api.py
│     └─ models.py
├─ scripts/
│  ├─ setup_wsl.sh
│  ├─ train_paper_parser.sh
│  ├─ eval_paper_parser.sh
│  ├─ train_planner.sh
│  ├─ eval_planner.sh
│  └─ serve.sh
├─ frontend/
│  ├─ (optional JS UI)
│  └─ README.md
├─ results/
│  ├─ paper_parser/
│  └─ planner/
├─ requirements.txt
└─ README.md
```

---

## Quick Start

### 1) Environment Setup (WSL/Linux)

* Install NVIDIA drivers on Windows and ensure WSL GPU support is working.
* Use Python 3.10+.

Create a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Tip: verify GPU is visible:

```bash
nvidia-smi
```

---

## Data Format

Both projects use **JSONL** (one example per line).

### Research Paper Parsing (paper_parser/*.jsonl)

Each line contains:

* `id`: unique example id
* `input_text`: paper text or sectioned text
* `output_json`: the structured digest JSON (must validate against `schemas/paper_digest.schema.json`) including `quality_summary` and `quality_flags`

Example:

```json
{"id":"paper_001","input_text":"...paper text...","output_json":{"quality_summary":"...","quality_flags":["..."],"paper_type":"theoretical","title":"...",...}}
```

### Clarification-First Planner (planner/*.jsonl)

Each line contains:

* `id`
* `goal`
* `state`: structured state description (text or JSON-as-string)
* `tools`: allowed skills/actions
* `output_json`: either a plan or a clarifying question (validates against `schemas/planner_output.schema.json`)

---

## Training

This repo is designed around **LoRA/QLoRA** so you can train on consumer GPUs.

### Research Paper Parser

```bash
bash scripts/train_paper_parser.sh
```

### Clarification-First Planner

```bash
bash scripts/train_planner.sh
```

Trained adapters and logs are saved under:

* `results/paper_parser/<run_id>/`
* `results/planner/<run_id>/`

---

## Evaluation

### Research Paper Parser

Primary metrics:

* Schema validity rate
* Repair rate (if you implement JSON repair)
* Evidence grounding checks (quotes appear in input)
* Completeness of required fields

Run:

```bash
bash scripts/eval_paper_parser.sh
```

### Clarification-First Planner

Primary metrics:

* Output schema validity
* Correct decision rate: plan vs clarify
* Constraint compliance (only allowed tools, args type checks)

Run:

```bash
bash scripts/eval_planner.sh
```

---

## Serving (API)

Run a local API server that loads the base model + selected adapter.

```bash
bash scripts/serve.sh --project paper_parser --adapter results/paper_parser/<run_id>/adapter
```

Example endpoints:

* `POST /parse` (paper parser)
* `POST /plan` (planner)

---

## Optional Frontend

A simple JavaScript UI can live in `frontend/`.

Typical features:

* Upload/paste text
* Select reading level / output length
* Display summary + JSON + evidence

---

## Roadmap

* [ ] V0 prompt-only baselines for both projects
* [ ] V1 LoRA/QLoRA training + deterministic eval
* [ ] V2 UI + report generation
* [ ] V3 improved robustness (chunking, better schema grounding)

---

## License / Notes

* Ensure you comply with the model license and any dataset licenses you use.
* This repo is for **learning and research**. Avoid using or sharing tooling intended for harmful misuse.

