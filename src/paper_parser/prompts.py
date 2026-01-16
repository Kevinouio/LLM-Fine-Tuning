SYSTEM_PROMPT = (
    "You are a research paper parser. "
    "Return JSON only with the required keys. "
    "Use concise language and avoid speculation."
)


def build_prompt(input_text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}

"
        "INPUT:
"
        f"{input_text}

"
        "OUTPUT JSON:"
    )
