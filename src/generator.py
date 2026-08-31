from typing import Any
from google import genai
from google.genai import types
from src.config import DEFAULT_TEMPERATURE, GENERATION_MODEL

SYSTEM_INSTRUCTION = (
    "You are Seminar Prep Bot. Answer clearly for a student preparing a seminar. "
    "Use the supplied paper excerpts as the primary evidence. "
    "Do not invent claims or citations. If the papers do not contain the answer, "
    "say so explicitly. Distinguish any web-based information under a heading "
    "called 'Web context'. End with a short 'Sources from papers' list using the "
    "paper name and page number provided in the context."
)


def format_context_excerpts(retrieved_chunks: list[dict[str, Any]]) -> str:
    return "\n\n---\n\n".join(
        f"[Paper: {item['source']}, page: {item['page']}]\n{item['text']}"
        for item in retrieved_chunks
    )


def generate_grounded_answer(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    client: genai.Client,
    use_web: bool = False,
    model: str = GENERATION_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    context = format_context_excerpts(retrieved_chunks)

    prompt = f"""Research-paper excerpts:
{context}

User question:
{query}

Give a concise but useful answer. Explain technical terms in simple language when needed."""

    config_kwargs: dict[str, Any] = {
        "system_instruction": SYSTEM_INSTRUCTION,
        "temperature": temperature,
    }

    if use_web:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )

    return response.text or "No response generated."
