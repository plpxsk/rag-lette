"""LLM generation for RAG answers."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMAdapter(ABC):
    """Adapter contract for provider-specific text generation."""

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Return a complete generation result for prompt."""

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        """Yield streamed generation chunks for prompt."""


class MistralLLMAdapter(LLMAdapter):
    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        return _call_mistral(prompt, model=self.model, temperature=temperature, max_tokens=max_tokens)

    def stream(self, prompt: str, *, temperature: float, max_tokens: int) -> Iterator[str]:
        yield from _stream_mistral(prompt, model=self.model, temperature=temperature, max_tokens=max_tokens)


class AnthropicLLMAdapter(LLMAdapter):
    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        return _call_anthropic(prompt, model=self.model, temperature=temperature, max_tokens=max_tokens)

    def stream(self, prompt: str, *, temperature: float, max_tokens: int) -> Iterator[str]:
        yield from _stream_anthropic(prompt, model=self.model, temperature=temperature, max_tokens=max_tokens)


class OpenAILLMAdapter(LLMAdapter):
    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        return _call_openai(prompt, model=self.model, temperature=temperature, max_tokens=max_tokens)

    def stream(self, prompt: str, *, temperature: float, max_tokens: int) -> Iterator[str]:
        yield from _stream_openai(prompt, model=self.model, temperature=temperature, max_tokens=max_tokens)


class GoogleLLMAdapter(LLMAdapter):
    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        return _call_google(prompt, model=self.model, temperature=temperature, max_tokens=max_tokens)

    def stream(self, prompt: str, *, temperature: float, max_tokens: int) -> Iterator[str]:
        yield from _stream_google(prompt, model=self.model, temperature=temperature, max_tokens=max_tokens)


def get_llm_adapter(provider: str, model: str) -> LLMAdapter:
    """Instantiate an LLM adapter from provider + model strings."""
    match provider:
        case "mistral":
            return MistralLLMAdapter(model)
        case "anthropic":
            return AnthropicLLMAdapter(model)
        case "openai":
            return OpenAILLMAdapter(model)
        case "google":
            return GoogleLLMAdapter(model)
        case _:
            raise ValueError(f"Unknown LLM provider: {provider!r}")


def generate(
    query: str,
    context: list[str],
    provider: str = "mistral",
    model: str = "ministral-3b-2512",
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """Generate an answer and return the full response as a string."""
    prompt = _build_prompt(query, context)
    adapter = get_llm_adapter(provider, model)
    return adapter.generate(prompt, temperature=temperature, max_tokens=max_tokens)


def generate_stream(
    query: str,
    context: list[str],
    provider: str = "mistral",
    model: str = "ministral-3b-2512",
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """Generate an answer and yield text chunks as they arrive from the LLM.

    Use this for streaming output to the terminal. Note that chunks are plain
    text — markdown rendering is not available during streaming.
    """
    prompt = _build_prompt(query, context)
    adapter = get_llm_adapter(provider, model)
    yield from adapter.stream(prompt, temperature=temperature, max_tokens=max_tokens)


def _build_prompt(query: str, context: list[str]) -> str:
    ctx = "\n\n---\n\n".join(context)
    return (
        "Context information is below.\n"
        "---------------------\n"
        f"{ctx}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        f"Query: {query}\n"
        "Answer:"
    )


def _call_mistral(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    from mistralai import Mistral
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY environment variable is not set")
    client = Mistral(api_key=api_key)
    response = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_anthropic(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.content[0].text


def _call_openai(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_google(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    from google import genai
    from google.genai import types
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return response.text or ""


def _stream_mistral(prompt: str, model: str, temperature: float, max_tokens: int) -> Iterator[str]:
    from mistralai import Mistral
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY environment variable is not set")
    client = Mistral(api_key=api_key)
    with client.chat.stream(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    ) as stream:
        for event in stream:
            delta = event.data.choices[0].delta.content
            if delta:
                yield delta


def _stream_anthropic(prompt: str, model: str, temperature: float, max_tokens: int) -> Iterator[str]:
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    client = Anthropic(api_key=api_key)
    with client.messages.stream(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    ) as stream:
        yield from stream.text_stream


def _stream_openai(prompt: str, model: str, temperature: float, max_tokens: int) -> Iterator[str]:
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    client = OpenAI(api_key=api_key)
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _stream_google(prompt: str, model: str, temperature: float, max_tokens: int) -> Iterator[str]:
    from google import genai
    from google.genai import types
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    ):
        if chunk.text:
            yield chunk.text
