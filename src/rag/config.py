"""
Config loading: merges ~/.rag.toml -> ./rag.toml -> --profile -> CLI flags -> env vars.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_EMBED_PROVIDER = "mistral"
DEFAULT_EMBED_MODEL = "mistral-embed"
DEFAULT_LLM_PROVIDER = "mistral"
DEFAULT_LLM_MODEL = "ministral-3b-2512"

LLM_ALIASES: dict[str, tuple[str, str]] = {
    "mistral":         ("mistral",   DEFAULT_LLM_MODEL),
    "mistral-small":   ("mistral",   DEFAULT_LLM_MODEL),
    "mistral-large":   ("mistral",   "mistral-large-2512"),
    "claude":          ("anthropic", "claude-haiku-4-5"),
    "anthropic":       ("anthropic", "claude-haiku-4-5"),
    "claude-sonnet-4-5": ("anthropic", "claude-sonnet-4-5"),
    "gpt-4o":          ("openai",    "gpt-4o"),
    "openai":          ("openai",    "gpt-4o"),
    "gpt-5.4":         ("openai",    "gpt-5.4"),
    "gpt-5.4-mini":    ("openai",    "gpt-5-mini-2025-08-07"),
    "gpt-5.4-nano":    ("openai",    "gpt-5-nano-2025-08-07"),
    "gpt-5-mini":      ("openai",    "gpt-5-mini-2025-08-07"),
    "gpt-5-nano":      ("openai",    "gpt-5-nano-2025-08-07"),
    "gemini":          ("google",    "gemini-2.5-flash"),
    "bedrock":         ("bedrock",   "amazon.nova-lite-v1:0"),
}

EMBED_ALIASES: dict[str, tuple[str, str]] = {
    "mistral":   (DEFAULT_EMBED_PROVIDER, DEFAULT_EMBED_MODEL),
    "voyageai":  ("voyageai",  "voyage-3.5-lite"),
    "voyage":    ("voyageai",  "voyage-3.5-lite"),
    "openai":    ("openai",    "text-embedding-3-small"),
    "gemini":    ("google",    "gemini-embedding-001"),
    "bedrock":   ("bedrock",   "cohere.embed-english-v3"),
}


class RagConfig(BaseModel):
    """Fully-resolved configuration for a single command invocation."""

    db: str = Field(default="./db", description="Database URI")
    table: str = Field(default="embeddings", description="Table/collection name")

    embed_provider: str = Field(default=DEFAULT_EMBED_PROVIDER)
    embed_model: str = Field(default=DEFAULT_EMBED_MODEL)

    llm_provider: str = Field(default=DEFAULT_LLM_PROVIDER)
    llm_model: str = Field(default=DEFAULT_LLM_MODEL)

    top_k: int = Field(default=5)
    max_tokens: int = Field(default=4096)

    chunk_method: str = Field(default="basic")
    chunk_size: int = Field(default=1000)
    chunk_overlap: int = Field(default=200)
    pdf_strategy: str = Field(default="fast")


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _get_profile(data: dict[str, Any], profile: str | None) -> dict[str, Any]:
    """Return [defaults] merged with [profiles.<name>]."""
    base: dict[str, Any] = dict(data.get("defaults", {}))
    if profile:
        base.update(data.get("profiles", {}).get(profile, {}))
    return base


def _parse_provider_model(
    value: str,
    aliases: dict[str, tuple[str, str]],
    default_provider: str,
    default_model: str,
) -> tuple[str, str]:
    if "/" in value:
        provider, model = value.split("/", 1)
        return provider.strip(), model.strip()
    if value in aliases:
        return aliases[value]
    return _resolve_model_only(value, aliases, default_provider)


def _resolve_model_only(
    model: str,
    aliases: dict[str, tuple[str, str]],
    default_provider: str,
) -> tuple[str, str]:
    if model in aliases:
        return aliases[model]
    matches = {provider for provider, alias_model in aliases.values() if alias_model == model}
    if not matches:
        return default_provider, model
    if default_provider in matches:
        return default_provider, model
    provider = sorted(matches)[0]
    return provider, model


def resolve_provider_model(
    value: str,
    aliases: dict[str, tuple[str, str]],
    default_provider: str,
    default_model: str,
) -> tuple[str, str]:
    """Resolve a provider/model shorthand into canonical provider and model strings."""
    return _parse_provider_model(value, aliases, default_provider, default_model)


def _flatten(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalise raw TOML/CLI keys into RagConfig field names."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "embed":
            p, m = _parse_provider_model(str(value), EMBED_ALIASES, DEFAULT_EMBED_PROVIDER, DEFAULT_EMBED_MODEL)
            out.setdefault("embed_provider", p)
            out.setdefault("embed_model", m)
        elif key == "llm":
            p, m = _parse_provider_model(str(value), LLM_ALIASES, DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL)
            out.setdefault("llm_provider", p)
            out.setdefault("llm_model", m)
        else:
            out[key.replace("-", "_")] = value
    return out


def load_config(
    profile: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> RagConfig:
    """
    Build a RagConfig (lowest -> highest priority):
      ~/.rag.toml < ./rag.toml < --profile < CLI overrides < env vars
    """
    global_data = _load_toml(Path.home() / ".rag.toml")
    local_data  = _load_toml(Path("rag.toml"))

    merged: dict[str, Any] = {
        **_flatten(_get_profile(global_data, profile)),
        **_flatten(_get_profile(local_data,  profile)),
        **_flatten(overrides or {}),
    }

    env_map: dict[str, str] = {
        "RAG_DB":            "db",
        "RAG_TABLE":         "table",
        "RAG_LLM":           "llm",
        "RAG_LLM_PROVIDER":  "llm_provider",
        "RAG_LLM_MODEL":     "llm_model",
        "RAG_EMBED":         "embed",
        "RAG_EMBED_PROVIDER": "embed_provider",
        "RAG_EMBED_MODEL":   "embed_model",
        "RAG_TOP_K":         "top_k",
        "RAG_MAX_TOKENS":    "max_tokens",
    }
    for env_key, cfg_key in env_map.items():
        if val := os.environ.get(env_key):
            merged.update(_flatten({cfg_key: val}))

    if "llm_model" in merged and not merged.get("llm_provider"):
        merged["llm_provider"], merged["llm_model"] = _resolve_model_only(
            str(merged["llm_model"]),
            LLM_ALIASES,
            DEFAULT_LLM_PROVIDER,
        )
    if "embed_model" in merged and not merged.get("embed_provider"):
        merged["embed_provider"], merged["embed_model"] = _resolve_model_only(
            str(merged["embed_model"]),
            EMBED_ALIASES,
            DEFAULT_EMBED_PROVIDER,
        )

    valid = {k: v for k, v in merged.items() if k in RagConfig.model_fields}
    return RagConfig(**valid)
