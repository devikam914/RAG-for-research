import os
from typing import Optional
from google import genai


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please set the GEMINI_API_KEY environment "
            "variable or supply a valid key."
        )
    return genai.Client(api_key=key)
