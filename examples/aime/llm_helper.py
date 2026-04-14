"""Thin wrapper for calling the local vLLM server."""

import os

from openai import OpenAI


def make_llm_fn(
    api_base: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 4096,
    temperature: float = 0.7,
):
    """Create a callable that sends a prompt to vLLM and returns the response.

    Reads from environment variables if not provided:
      VLLM_BASE_URL, VLLM_MODEL, VLLM_API_KEY
    """
    api_base = api_base or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
    model = model or os.environ.get("VLLM_MODEL", "Qwen/Qwen3-8B")
    api_key = api_key or os.environ.get("VLLM_API_KEY", "local-key")

    client = OpenAI(base_url=api_base, api_key=api_key)

    def llm_fn(prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"ERROR: {e}"

    return llm_fn
