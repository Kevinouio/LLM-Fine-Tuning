SYSTEM_PROMPT = (
    "You are a planner that either returns a plan or a clarifying question. "
    "Return JSON only and use allowed tools."
)


def build_prompt(goal: str, state: str, tools: list[str]) -> str:
    tools_text = ", ".join(tools) if tools else "(none)"
    return (
        f"{SYSTEM_PROMPT}

"
        f"GOAL: {goal}
"
        f"STATE: {state}
"
        f"TOOLS: {tools_text}

"
        "OUTPUT JSON:"
    )
