# Provider-agnostic (bring-your-own-key) LLM abstraction for ODDA.
#
# This module decouples ODDA's chat-completion and text-embedding calls from any
# single vendor. Callers use two entry points:
#
#   * complete_json(...) -> CompletionResult   (chat completion returning parsed JSON)
#   * embed(...)         -> EmbeddingResult     (one or many embedding vectors)
#
# Chat and embedding providers are configured INDEPENDENTLY because some chat
# providers (e.g. Anthropic Claude) cannot produce embeddings. A typical setup is
# chat = Claude-hosted-on-Azure and embedding = Azure-OpenAI text-embedding-3-small.
#
# Supported providers:
#   chat:      azure_openai | azure_claude | openai | anthropic | ollama
#   embedding: azure_openai | openai | ollama
#
# There is NO hard-coded default provider. Configuration is resolved, in order:
#   1. Environment variables (highest precedence, per field):
#        ODDA_CHAT_PROVIDER / ODDA_CHAT_MODEL / ODDA_CHAT_ENDPOINT /
#        ODDA_CHAT_BASE_URL / ODDA_CHAT_RESOURCE / ODDA_CHAT_API_KEY /
#        ODDA_CHAT_API_VERSION
#        ODDA_EMBEDDING_PROVIDER / ODDA_EMBEDDING_MODEL / ODDA_EMBEDDING_ENDPOINT /
#        ODDA_EMBEDDING_BASE_URL / ODDA_EMBEDDING_API_KEY / ODDA_EMBEDDING_API_VERSION
#   2. A JSON config file (default: .claude/model.config; override with the
#      ODDA_MODEL_CONFIG env var or the config_file argument).
#   3. Legacy fallback: if no provider is configured but Azure OpenAI credentials
#      are available (passed in by a caller, or via AZURE_OPENAI_ENDPOINT /
#      AZURE_OPENAI_API_KEY, or the .claude/azure.endpoint / .claude/azure.key
#      files), the provider is inferred to be azure_openai. This preserves the
#      original Azure-OpenAI-only behaviour for existing deployments.
#
# If nothing can be resolved, a ModelConfigError is raised with an actionable
# message telling the user to configure a provider/key (a separate /setup skill
# is expected to call into this later).
#
# .claude/model.config format (JSON)::
#
#   {
#     "chat": {
#       "provider": "azure_claude",
#       "model": "claude-opus-4-8",
#       "resource": "my-foundry-resource",        // OR "base_url"/"endpoint"
#       "api_key_file": ".claude/azure_claude.key" // OR "api_key"/"api_key_env"
#     },
#     "embedding": {
#       "provider": "azure_openai",
#       "model": "text-embedding-3-small",
#       "endpoint_file": ".claude/azure.endpoint", // OR "endpoint"/"endpoint_env"
#       "api_key_file": ".claude/azure.key",       // OR "api_key"/"api_key_env"
#       "api_version": "2024-02-01"
#     }
#   }
#
# Legacy azure hints: complete_json / embed accept endpoint / api_key / model /
# endpoint_file / api_key_file arguments. These are honoured ONLY when the
# resolved provider is azure_openai (they are Azure-OpenAI-shaped and are what the
# original call sites pass); for any other provider they are ignored so that, for
# example, Azure-OpenAI credentials are never sent to a Claude endpoint.
#
# Provenance: CompletionResult and EmbeddingResult carry the provider id and the
# exact model id actually used. describe_config() / active_chat_model() /
# active_embedding_model() expose the resolved provider+model without making a
# request, so a later step can persist provenance to the database.

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = Path(".claude/model.config")

CHAT_PROVIDERS = frozenset(
    {"azure_openai", "azure_claude", "openai", "anthropic", "ollama"}
)
EMBEDDING_PROVIDERS = frozenset({"azure_openai", "openai", "ollama"})

# Providers that speak the OpenAI chat/embeddings wire protocol.
_OPENAI_FAMILY = frozenset({"azure_openai", "openai", "ollama"})
# Providers that speak the Anthropic Messages protocol.
_ANTHROPIC_FAMILY = frozenset({"azure_claude", "anthropic"})

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_AZURE_API_VERSION = "2024-02-01"


class ModelConfigError(Exception):
    """Raised when no usable model/provider configuration can be resolved."""


class LLMProviderError(Exception):
    """Raised when a configured provider cannot service a request."""


@dataclass
class ProviderConfig:
    """Resolved configuration for a single role (chat or embedding).

    Attributes
    ----------
    role : str
        Either ``"chat"`` or ``"embedding"``.
    provider : str
        The resolved provider id (e.g. ``"azure_openai"``).
    model : str or None
        The model / deployment id to use.
    endpoint : str or None
        Endpoint URL (Azure OpenAI) when applicable.
    base_url : str or None
        Base URL for OpenAI-compatible or Anthropic-compatible endpoints.
    resource : str or None
        Azure AI Foundry resource name (for azure_claude).
    api_key : str or None
        API key / token for the provider.
    api_version : str or None
        API version (Azure OpenAI).
    """

    role: str
    provider: str
    model: str | None = None
    endpoint: str | None = None
    base_url: str | None = None
    resource: str | None = None
    api_key: str | None = None
    api_version: str | None = None


@dataclass
class CompletionResult:
    """Result of a chat completion.

    Attributes
    ----------
    text : str
        The raw response text (expected to be a JSON document).
    data : dict or None
        The parsed JSON object, or None if parsing failed.
    provider : str
        The provider that served the request.
    model : str
        The exact model id that produced the response.
    """

    text: str
    data: dict | None
    provider: str
    model: str


@dataclass
class EmbeddingResult:
    """Result of an embedding request (one or more input strings).

    Attributes
    ----------
    vectors : list[list[float]]
        One embedding vector per input string, in input order.
    provider : str
        The provider that served the request.
    model : str
        The exact embedding model id used.
    """

    vectors: list[list[float]] = field(default_factory=list)
    provider: str = ""
    model: str = ""

    @property
    def vector(self) -> list[float]:
        """Return the single embedding vector (first input)."""
        if not self.vectors:
            raise LLMProviderError("Embedding response contained no vectors")
        return self.vectors[0]


# ---------------------------------------------------------------------------
# Configuration loading and resolution
# ---------------------------------------------------------------------------


def _load_config_file(config_file: str | Path | None) -> dict:
    """Read and parse the JSON model config file, if present.

    Parameters
    ----------
    config_file : str or Path or None
        Explicit config path. If None, the ODDA_MODEL_CONFIG environment
        variable is consulted, then the default .claude/model.config path.

    Returns
    -------
    dict
        The parsed config mapping, or an empty dict if no file exists.

    Raises
    ------
    ModelConfigError
        If the file exists but cannot be parsed as JSON.
    """
    if config_file is None:
        config_file = os.environ.get("ODDA_MODEL_CONFIG")
    path = Path(config_file).expanduser() if config_file else DEFAULT_CONFIG_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ModelConfigError(f"Failed to read model config {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelConfigError(
            f"Model config {path} must contain a JSON object at the top level"
        )
    return data


def _read_secret_field(block: dict, base: str) -> str | None:
    """Resolve a secret-ish field that may be inline, in a file, or in an env var.

    Looks for ``<base>`` (inline value), ``<base>_env`` (environment variable
    name), and ``<base>_file`` (path to a file whose stripped contents are used),
    in that order.

    Parameters
    ----------
    block : dict
        A config block (e.g. the "chat" mapping).
    base : str
        The base field name (e.g. "api_key" or "endpoint").

    Returns
    -------
    str or None
        The resolved value, or None if unset.
    """
    if block.get(base):
        return str(block[base]).strip()
    env_name = block.get(f"{base}_env")
    if env_name and os.environ.get(env_name):
        return os.environ[env_name].strip()
    file_name = block.get(f"{base}_file")
    if file_name:
        path = Path(file_name).expanduser()
        if path.exists():
            return path.read_text().strip()
    return None


def _merge_block(role: str, file_cfg: dict) -> dict:
    """Merge a config-file block for a role with environment-variable overrides.

    Parameters
    ----------
    role : str
        Either "chat" or "embedding".
    file_cfg : dict
        The full parsed config-file mapping.

    Returns
    -------
    dict
        A normalized block with keys: provider, model, endpoint, base_url,
        resource, api_key, api_version (values may be None).
    """
    block = dict(file_cfg.get(role) or {})
    env_prefix = "ODDA_CHAT_" if role == "chat" else "ODDA_EMBEDDING_"

    def env(name: str) -> str | None:
        value = os.environ.get(env_prefix + name)
        return value.strip() if value else None

    return {
        "provider": env("PROVIDER") or block.get("provider"),
        "model": env("MODEL") or block.get("model"),
        "endpoint": env("ENDPOINT") or _read_secret_field(block, "endpoint"),
        "base_url": env("BASE_URL") or block.get("base_url"),
        "resource": env("RESOURCE") or block.get("resource"),
        "api_key": env("API_KEY") or _read_secret_field(block, "api_key"),
        "api_version": env("API_VERSION") or block.get("api_version"),
    }


def _azure_credentials(
    endpoint: str | None,
    api_key: str | None,
    endpoint_file: str | Path | None,
    api_key_file: str | Path | None,
) -> tuple[str | None, str | None]:
    """Resolve Azure OpenAI endpoint/key from explicit values, files, or env.

    Precedence: explicit endpoint/api_key > *_file > utils.get_azure_credentials
    (which itself falls back to AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY).

    Returns
    -------
    tuple of (str or None, str or None)
        The resolved (endpoint, api_key); either element may be None if it
        could not be resolved.
    """
    if endpoint and api_key:
        return endpoint, api_key
    # Import lazily to avoid a circular import with odda_utils.utils.
    from odda_utils.utils import AzureCredentialsError, get_azure_credentials

    try:
        resolved_endpoint, resolved_key = get_azure_credentials(
            endpoint_file, api_key_file
        )
    except AzureCredentialsError:
        return endpoint, api_key
    return endpoint or resolved_endpoint, api_key or resolved_key


def _config_error(role: str) -> ModelConfigError:
    """Build an actionable ModelConfigError for an unconfigured role."""
    prefix = "ODDA_CHAT_" if role == "chat" else "ODDA_EMBEDDING_"
    return ModelConfigError(
        f"No {role} model provider is configured. ODDA uses a bring-your-own-key "
        "model layer with no default provider. Configure one by either creating "
        f"{DEFAULT_CONFIG_FILE} (JSON with a '{role}' block naming a provider, "
        "model and key) or setting the environment variables "
        f"{prefix}PROVIDER / {prefix}MODEL / {prefix}API_KEY (and, for Azure, "
        f"{prefix}ENDPOINT). Run the /setup skill to configure this."
    )


def resolve_chat_config(
    *,
    config_file: str | Path | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    api_version: str | None = None,
) -> ProviderConfig:
    """Resolve the effective chat-completion provider configuration.

    The endpoint / api_key / model / api_version arguments are Azure-OpenAI
    legacy hints, honoured only when the resolved provider is azure_openai.

    Parameters
    ----------
    config_file : str or Path or None
        Optional override for the config-file path.
    endpoint, api_key, model, api_version : str or None
        Azure-OpenAI legacy hints from existing call sites.

    Returns
    -------
    ProviderConfig
        The resolved chat configuration.

    Raises
    ------
    ModelConfigError
        If no chat provider can be resolved, or the provider is unknown.
    """
    block = _merge_block("chat", _load_config_file(config_file))
    provider = block["provider"]

    if provider is None:
        # Legacy fallback: infer azure_openai only if Azure creds are available.
        eff_endpoint, eff_key = _azure_credentials(endpoint, api_key, None, None)
        if eff_endpoint and eff_key:
            return ProviderConfig(
                role="chat",
                provider="azure_openai",
                model=model or "gpt-5",
                endpoint=eff_endpoint,
                api_key=eff_key,
                api_version=api_version or _DEFAULT_AZURE_API_VERSION,
            )
        raise _config_error("chat")

    if provider not in CHAT_PROVIDERS:
        raise ModelConfigError(
            f"Unknown chat provider '{provider}'. Valid options: "
            f"{', '.join(sorted(CHAT_PROVIDERS))}."
        )

    cfg = ProviderConfig(
        role="chat",
        provider=provider,
        model=block["model"],
        endpoint=block["endpoint"],
        base_url=block["base_url"],
        resource=block["resource"],
        api_key=block["api_key"],
        api_version=block["api_version"] or _DEFAULT_AZURE_API_VERSION,
    )

    if provider == "azure_openai":
        # Honour legacy hints and fall back to env-based Azure credentials.
        eff_endpoint, eff_key = _azure_credentials(
            endpoint or cfg.endpoint, api_key or cfg.api_key, None, None
        )
        cfg.endpoint = eff_endpoint
        cfg.api_key = eff_key
        cfg.model = model or cfg.model or "gpt-5"
        if api_version:
            cfg.api_version = api_version
    elif endpoint or api_key or model:
        logger.debug(
            "Ignoring Azure-OpenAI legacy hints for chat provider '%s'; using "
            "configured values instead.",
            provider,
        )
    return cfg


def resolve_embedding_config(
    *,
    config_file: str | Path | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    model: str | None = None,
    api_version: str | None = None,
) -> ProviderConfig:
    """Resolve the effective embedding provider configuration.

    The endpoint / api_key / endpoint_file / api_key_file / model / api_version
    arguments are Azure-OpenAI legacy hints, honoured only when the resolved
    provider is azure_openai.

    Returns
    -------
    ProviderConfig
        The resolved embedding configuration.

    Raises
    ------
    ModelConfigError
        If no embedding provider can be resolved, or the provider is unknown.
    """
    block = _merge_block("embedding", _load_config_file(config_file))
    provider = block["provider"]

    if provider is None:
        eff_endpoint, eff_key = _azure_credentials(
            endpoint, api_key, endpoint_file, api_key_file
        )
        if eff_endpoint and eff_key:
            return ProviderConfig(
                role="embedding",
                provider="azure_openai",
                model=model or "text-embedding-3-small",
                endpoint=eff_endpoint,
                api_key=eff_key,
                api_version=api_version or _DEFAULT_AZURE_API_VERSION,
            )
        raise _config_error("embedding")

    if provider not in EMBEDDING_PROVIDERS:
        raise ModelConfigError(
            f"Unknown or unsupported embedding provider '{provider}'. Valid "
            f"options: {', '.join(sorted(EMBEDDING_PROVIDERS))}. Note that chat-"
            "only providers such as Claude cannot produce embeddings; configure a "
            "separate embedding provider."
        )

    cfg = ProviderConfig(
        role="embedding",
        provider=provider,
        model=block["model"],
        endpoint=block["endpoint"],
        base_url=block["base_url"],
        api_key=block["api_key"],
        api_version=block["api_version"] or _DEFAULT_AZURE_API_VERSION,
    )

    if provider == "azure_openai":
        eff_endpoint, eff_key = _azure_credentials(
            endpoint or cfg.endpoint,
            api_key or cfg.api_key,
            endpoint_file,
            api_key_file,
        )
        cfg.endpoint = eff_endpoint
        cfg.api_key = eff_key
        cfg.model = model or cfg.model or "text-embedding-3-small"
        if api_version:
            cfg.api_version = api_version
    elif endpoint or api_key or endpoint_file or api_key_file or model:
        logger.debug(
            "Ignoring Azure-OpenAI legacy hints for embedding provider '%s'; "
            "using configured values instead.",
            provider,
        )
    return cfg


# ---------------------------------------------------------------------------
# Provenance helpers (retrievable without making a request)
# ---------------------------------------------------------------------------


def active_chat_model(config_file: str | Path | None = None) -> tuple[str, str | None]:
    """Return the (provider, model) that chat completions would use.

    Returns
    -------
    tuple of (str, str or None)
        The resolved chat provider id and model id.
    """
    cfg = resolve_chat_config(config_file=config_file)
    return cfg.provider, cfg.model


def active_embedding_model(
    config_file: str | Path | None = None,
) -> tuple[str, str | None]:
    """Return the (provider, model) that embeddings would use.

    Returns
    -------
    tuple of (str, str or None)
        The resolved embedding provider id and model id.
    """
    cfg = resolve_embedding_config(config_file=config_file)
    return cfg.provider, cfg.model


def describe_config(config_file: str | Path | None = None) -> dict:
    """Describe the resolved chat and embedding providers (no secrets).

    Useful for persisting provenance. Each entry is either
    ``{"provider": ..., "model": ...}`` or ``{"error": ...}`` if that role is
    not configured.

    Returns
    -------
    dict
        A mapping with "chat" and "embedding" entries.
    """
    result: dict[str, Any] = {}
    for role, resolver in (
        ("chat", resolve_chat_config),
        ("embedding", resolve_embedding_config),
    ):
        try:
            cfg = resolver(config_file=config_file)
            result[role] = {"provider": cfg.provider, "model": cfg.model}
        except ModelConfigError as exc:
            result[role] = {"error": str(exc)}
    return result


# ---------------------------------------------------------------------------
# Provider-specific request handlers
# ---------------------------------------------------------------------------

_JSON_SYSTEM_SUFFIX = (
    " Respond with a single valid JSON object and nothing else. Do not wrap the "
    "JSON in Markdown code fences or add commentary."
)


def _strip_json_text(text: str) -> str:
    """Strip Markdown code fences that some models wrap around JSON."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence line (``` or ```json) and the trailing fence.
        newline = stripped.find("\n")
        if newline != -1:
            stripped = stripped[newline + 1 :]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -3]
    return stripped.strip()


def _openai_chat_text(
    client: Any,
    model: str,
    system: str | None,
    prompt: str,
    max_tokens: int,
    temperature: float | None,
) -> tuple[str, str]:
    """Call an OpenAI-compatible chat endpoint and return (text, model_id).

    Mirrors the historical behaviour: request JSON object mode, prefer
    ``max_completion_tokens`` and fall back to ``max_tokens`` for older models.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    base_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if temperature is not None:
        base_kwargs["temperature"] = temperature

    try:
        response = client.chat.completions.create(
            max_completion_tokens=max_tokens, **base_kwargs
        )
    except Exception as exc:  # noqa: BLE001 - fall back for older models
        if "max_completion_tokens" in str(exc) or "unsupported_parameter" in str(exc):
            response = client.chat.completions.create(
                max_tokens=max_tokens, **base_kwargs
            )
        else:
            raise
    return response.choices[0].message.content, response.model


def _anthropic_message_text(
    client: Any,
    model: str,
    system: str | None,
    prompt: str,
    max_tokens: int,
) -> tuple[str, str]:
    """Call an Anthropic Messages endpoint and return (text, model_id)."""
    system_prompt = (system or "").strip()
    system_prompt = (system_prompt + _JSON_SYSTEM_SUFFIX).strip()
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    response = client.messages.create(**kwargs)
    parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    return "".join(parts), getattr(response, "model", model)


def _build_openai_client(cfg: ProviderConfig) -> tuple[Any, str]:
    """Construct an OpenAI-compatible client for the given config.

    Returns
    -------
    tuple of (client, model)
        The instantiated client and the model id to use.
    """
    if cfg.provider == "azure_openai":
        from openai import AzureOpenAI

        if not cfg.endpoint or not cfg.api_key:
            raise ModelConfigError(
                "azure_openai provider requires an endpoint and api_key."
            )
        client = AzureOpenAI(
            azure_endpoint=cfg.endpoint,
            api_key=cfg.api_key,
            api_version=cfg.api_version or _DEFAULT_AZURE_API_VERSION,
        )
        return client, cfg.model
    if cfg.provider == "openai":
        from openai import OpenAI

        if not cfg.api_key:
            raise ModelConfigError("openai provider requires an api_key.")
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None)
        return client, cfg.model
    if cfg.provider == "ollama":
        from openai import OpenAI

        client = OpenAI(
            api_key=cfg.api_key or "ollama",
            base_url=cfg.base_url or cfg.endpoint or _DEFAULT_OLLAMA_BASE_URL,
        )
        return client, cfg.model
    raise LLMProviderError(f"Provider '{cfg.provider}' is not OpenAI-compatible.")


def _build_anthropic_client(cfg: ProviderConfig) -> tuple[Any, str]:
    """Construct an Anthropic-compatible client for the given config."""
    if cfg.provider == "anthropic":
        from anthropic import Anthropic

        client = Anthropic(api_key=cfg.api_key or None, base_url=cfg.base_url or None)
        return client, cfg.model
    if cfg.provider == "azure_claude":
        from anthropic import AnthropicFoundry

        if not cfg.resource and not (cfg.base_url or cfg.endpoint):
            raise ModelConfigError(
                "azure_claude provider requires a 'resource' or 'base_url'."
            )
        client = AnthropicFoundry(
            resource=cfg.resource or None,
            base_url=cfg.base_url or cfg.endpoint or None,
            api_key=cfg.api_key or None,
        )
        return client, cfg.model
    raise LLMProviderError(f"Provider '{cfg.provider}' is not Anthropic-compatible.")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def complete_json(
    prompt: str,
    *,
    system: str | None = None,
    config_file: str | Path | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    api_version: str | None = None,
    max_tokens: int = 16384,
    temperature: float | None = None,
) -> CompletionResult:
    """Run a chat completion via the configured chat provider, returning JSON.

    Parameters
    ----------
    prompt : str
        The user prompt.
    system : str or None
        Optional system prompt.
    config_file : str or Path or None
        Optional override for the model config path.
    endpoint, api_key, model, api_version : str or None
        Azure-OpenAI legacy hints (honoured only when the resolved provider is
        azure_openai).
    max_tokens : int
        Maximum tokens in the response.
    temperature : float or None
        Sampling temperature for OpenAI-family providers. Ignored (and never
        sent) for Anthropic-family providers, which reject it.

    Returns
    -------
    CompletionResult
        The response text, parsed JSON (if valid), and provider/model used.

    Raises
    ------
    ModelConfigError
        If no chat provider is configured.
    LLMProviderError
        If the provider call fails.
    """
    cfg = resolve_chat_config(
        config_file=config_file,
        endpoint=endpoint,
        api_key=api_key,
        model=model,
        api_version=api_version,
    )
    if not cfg.model:
        raise ModelConfigError(
            f"No chat model id is configured for provider '{cfg.provider}'."
        )

    try:
        if cfg.provider in _OPENAI_FAMILY:
            client, resolved_model = _build_openai_client(cfg)
            text, used_model = _openai_chat_text(
                client, resolved_model, system, prompt, max_tokens, temperature
            )
        elif cfg.provider in _ANTHROPIC_FAMILY:
            client, resolved_model = _build_anthropic_client(cfg)
            text, used_model = _anthropic_message_text(
                client, resolved_model, system, prompt, max_tokens
            )
        else:  # pragma: no cover - guarded by resolve_chat_config
            raise LLMProviderError(f"Unhandled chat provider '{cfg.provider}'.")
    except (ModelConfigError, LLMProviderError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMProviderError(
            f"Chat completion failed for provider '{cfg.provider}': {exc}"
        ) from exc

    data: dict | None = None
    if text:
        try:
            parsed = json.loads(_strip_json_text(text))
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            data = None

    return CompletionResult(
        text=text or "",
        data=data,
        provider=cfg.provider,
        model=used_model or cfg.model,
    )


def embed(
    text: str | list[str],
    *,
    config_file: str | Path | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    endpoint_file: str | Path | None = None,
    api_key_file: str | Path | None = None,
    model: str | None = None,
    api_version: str | None = None,
) -> EmbeddingResult:
    """Produce embedding vectors for one or more strings.

    Parameters
    ----------
    text : str or list of str
        A single string or a list of strings to embed.
    config_file : str or Path or None
        Optional override for the model config path.
    endpoint, api_key, endpoint_file, api_key_file, model, api_version : optional
        Azure-OpenAI legacy hints (honoured only when the resolved provider is
        azure_openai).

    Returns
    -------
    EmbeddingResult
        The embedding vector(s) and the provider/model used.

    Raises
    ------
    ModelConfigError
        If no embedding provider is configured.
    LLMProviderError
        If the provider call fails.
    """
    cfg = resolve_embedding_config(
        config_file=config_file,
        endpoint=endpoint,
        api_key=api_key,
        endpoint_file=endpoint_file,
        api_key_file=api_key_file,
        model=model,
        api_version=api_version,
    )
    if not cfg.model:
        raise ModelConfigError(
            f"No embedding model id is configured for provider '{cfg.provider}'."
        )
    if cfg.provider not in _OPENAI_FAMILY:  # pragma: no cover - guarded above
        raise LLMProviderError(
            f"Provider '{cfg.provider}' cannot produce embeddings."
        )

    inputs = [text] if isinstance(text, str) else list(text)

    try:
        client, resolved_model = _build_openai_client(cfg)
        response = client.embeddings.create(input=inputs, model=resolved_model)
    except (ModelConfigError, LLMProviderError):
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMProviderError(
            f"Embedding request failed for provider '{cfg.provider}': {exc}"
        ) from exc

    vectors = [list(item.embedding) for item in response.data]
    used_model = getattr(response, "model", None) or cfg.model
    return EmbeddingResult(vectors=vectors, provider=cfg.provider, model=used_model)
