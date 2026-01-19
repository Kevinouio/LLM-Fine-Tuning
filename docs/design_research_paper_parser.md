# Research Paper Parser - Design

## Goal
Turn research paper text into a structured JSON digest, plus a quality summary, rubric scores, and flags.

## Inputs
- input_text: raw paper text or sectioned text.
- id: unique identifier.

## Outputs
- output_json that validates against schemas/paper_digest.schema.json, including quality_summary, quality_scores, and quality_flags.

## Data pipeline
1. Place raw text in data/paper_parser/raw/.
2. Run preprocess to normalize and chunk (optional).
3. Write JSONL to data/paper_parser/processed/ and train/val/test.

## Prompting strategy
- Extract only what appears in the input.
- Keep outputs concise and JSON-only.
- Provide evidence quotes when possible.

## Evaluation
- Schema validity rate.
- Evidence grounding: quotes should appear in input_text.
- Completeness of required fields.

## TODO
- Add chunking for long papers.
- Add JSON repair for near-valid outputs.
- Add automatic citation extraction.
