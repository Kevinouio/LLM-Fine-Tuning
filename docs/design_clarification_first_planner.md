# Clarification-First Planner - Design

## Goal
Given a goal and current state, output a plan if enough info is available,
otherwise ask a clarifying question.

## Inputs
- goal: user objective.
- state: current context, resources, constraints.
- tools: list of allowed tools/actions.

## Outputs
- output_json that validates against schemas/planner_output.schema.json.

## Decision rule
- If required info for the next action is missing, ask one clarifying question.
- Otherwise produce a short, ordered plan using only allowed tools.

## Evaluation
- Schema validity.
- Decision accuracy (plan vs clarify) compared to labels.
- Constraint compliance (tool names and args types).

## TODO
- Add richer state schema.
- Add unit tests for decision rule.
