SYSTEM_PROMPT = (
    "You are a research paper parser. "
    "Return JSON only with the required keys. "
    "Include a quality_summary, quality_scores, and quality_flags before the rest of the digest. "
    "Use concise language and avoid speculation."
)


def build_prompt(input_text: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "INPUT:\n"
        f"{input_text}\n\n"
        "OUTPUT JSON:"
    )
