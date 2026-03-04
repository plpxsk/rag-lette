"""
Embedding adapter interface and placeholder implementations.

To add a new embedder:
  1. Subclass EmbedAdapter and implement embed() and dim.
  2. Add a match in get_embed_adapter().
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class EmbedAdapter(ABC):

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding vector dimensionality."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class VoyageAIEmbedder(EmbedAdapter):
    """VoyageAI embeddings.  Models: voyage-3.5-lite, voyage-3.5, ...

    Requires: pip install 'rag[voyageai]'
    """

    def __init__(self, model: str = "voyage-3.5-lite") -> None:
        self.model = model

    @property
    def dim(self) -> int:
        return 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        import os
        try:
            import voyageai
        except ImportError as exc:
            raise ImportError(
                "The 'voyageai' package is required for VoyageAI embeddings.\n"
                "Install it with:  pip install \"rag[voyageai]\"\n"
                f"Original error: {exc}"
            ) from exc
        client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        result: list[list[float]] = []
        for i in range(0, len(texts), 128):
            batch = texts[i : i + 128]
            response = client.embed(batch, model=self.model)
            result.extend(response.embeddings)
        return result


class OpenAIEmbedder(EmbedAdapter):
    """OpenAI embeddings.  Models: text-embedding-3-small, text-embedding-3-large, ...

    Requires: pip install 'rag[openai]'
    """

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model

    @property
    def dim(self) -> int:
        return 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        import os
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAI embeddings.\n"
                "Install it with:  pip install \"rag[openai]\"\n"
                f"Original error: {exc}"
            ) from exc
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        client = OpenAI(api_key=api_key)
        result: list[list[float]] = []
        for i in range(0, len(texts), 512):
            batch = texts[i : i + 512]
            response = client.embeddings.create(model=self.model, input=batch)
            result.extend(e.embedding for e in response.data)
        return result


class GeminiEmbedder(EmbedAdapter):
    """Google Gemini embeddings.  Model: gemini-embedding-001 (text-embedding-004 is deprecated in v1beta).

    Requires: pip install 'rag[gemini]'  and GEMINI_API_KEY.
    """

    def __init__(self, model: str = "gemini-embedding-001") -> None:
        self.model = model

    @property
    def dim(self) -> int:
        return 3072

    def embed(self, texts: list[str]) -> list[list[float]]:
        import os
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "The 'google-genai' package is required for Gemini embeddings.\n"
                "Install it with:  pip install \"rag[gemini]\"\n"
                f"Original error: {exc}"
            ) from exc
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        client = genai.Client(api_key=api_key)
        result: list[list[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            response = client.models.embed_content(
                model=self.model,
                contents=batch,
            )
            for emb in response.embeddings:
                vec = getattr(emb, "values", None) or getattr(emb, "embedding", None)
                if vec is not None:
                    result.append(list(vec))
        return result


class MistralEmbedder(EmbedAdapter):
    """Mistral embeddings.  Models: mistral-embed, ..."""

    def __init__(self, model: str = "mistral-embed") -> None:
        self.model = model

    @property
    def dim(self) -> int:
        return 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        import os
        from mistralai import Mistral
        client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        result: list[list[float]] = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            response = client.embeddings.create(model=self.model, inputs=batch)
            result.extend(e.embedding for e in response.data)
        return result


class BedrockEmbedAdapter(EmbedAdapter):
    """AWS Bedrock embeddings.  Models: cohere.embed-english-v3, amazon.titan-embed-text-v2:0, ...

    Requires: pip install 'rag[bedrock]'  and standard AWS credential chain.
    """

    def __init__(self, model: str = "cohere.embed-english-v3") -> None:
        self.model = model

    @property
    def dim(self) -> int:
        return 1024

    def embed(self, texts: list[str]) -> list[list[float]]:
        import json
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "The 'boto3' package is required for Bedrock embeddings.\n"
                "Install it with:  pip install \"rag[bedrock]\"\n"
                f"Original error: {exc}"
            ) from exc
        client = boto3.client("bedrock-runtime")
        result: list[list[float]] = []
        if self.model.startswith("cohere"):
            # Cohere batch API: up to 96 texts per call
            for i in range(0, len(texts), 96):
                batch = texts[i : i + 96]
                body = json.dumps({"texts": batch, "input_type": "search_document"})
                response = client.invoke_model(modelId=self.model, body=body)
                result.extend(json.loads(response["body"].read())["embeddings"])
        else:
            # Titan: one text per call
            for text in texts:
                body = json.dumps({"inputText": text})
                response = client.invoke_model(modelId=self.model, body=body)
                result.append(json.loads(response["body"].read())["embedding"])
        return result


def get_embed_adapter(provider: str, model: str) -> EmbedAdapter:
    """Instantiate the correct EmbedAdapter from provider + model strings."""
    match provider:
        case "voyageai":
            return VoyageAIEmbedder(model)
        case "openai":
            return OpenAIEmbedder(model)
        case "google" | "gemini":
            return GeminiEmbedder(model)
        case "mistral":
            return MistralEmbedder(model)
        case "bedrock":
            return BedrockEmbedAdapter(model)
        case _:
            raise ValueError(f"Unknown embedding provider: {provider!r}")
