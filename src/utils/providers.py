"""LLM providers the rule generator can talk to, and helpers for importing API keys."""

import re
from typing import Dict, List, Optional

CUSTOM = "Custom (OpenAI-compatible)"

# kind: "gemini" uses the google-genai SDK, "anthropic" uses the Messages API,
# "openai" uses any OpenAI-compatible /chat/completions endpoint.
PROVIDERS: Dict[str, Dict[str, object]] = {
    "Google Gemini": {"kind": "gemini", "env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "model": "gemini-2.5-flash", "base_url": ""},
    "Anthropic": {"kind": "anthropic", "env": ["ANTHROPIC_API_KEY"], "model": "claude-sonnet-5-5", "base_url": "https://api.anthropic.com"},
    "OpenAI": {"kind": "openai", "env": ["OPENAI_API_KEY"], "model": "gpt-4o-mini", "base_url": "https://api.openai.com/v1"},
    "OpenRouter": {"kind": "openai", "env": ["OPENROUTER_API_KEY"], "model": "openai/gpt-4o-mini", "base_url": "https://openrouter.ai/api/v1"},
    "Groq": {"kind": "openai", "env": ["GROQ_API_KEY"], "model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"},
    "Mistral": {"kind": "openai", "env": ["MISTRAL_API_KEY"], "model": "mistral-small-latest", "base_url": "https://api.mistral.ai/v1"},
    "Together AI": {"kind": "openai", "env": ["TOGETHER_API_KEY"], "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "base_url": "https://api.together.xyz/v1"},
    "DeepSeek": {"kind": "openai", "env": ["DEEPSEEK_API_KEY"], "model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1"},
    "xAI": {"kind": "openai", "env": ["XAI_API_KEY"], "model": "grok-3", "base_url": "https://api.x.ai/v1"},
    "Cohere": {"kind": "openai", "env": ["COHERE_API_KEY"], "model": "command-r-plus", "base_url": "https://api.cohere.ai/compatibility/v1"},
    "Hugging Face": {"kind": "openai", "env": ["HF_TOKEN", "HUGGINGFACE_API_KEY"], "model": "meta-llama/Llama-3.3-70B-Instruct", "base_url": "https://router.huggingface.co/v1"},
    "Perplexity": {"kind": "openai", "env": ["PERPLEXITY_API_KEY"], "model": "sonar", "base_url": "https://api.perplexity.ai"},
    CUSTOM: {"kind": "openai", "env": ["LLM_API_KEY", "AZURE_OPENAI_API_KEY"], "model": "", "base_url": ""},
}

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$")


def env_names(provider: str) -> List[str]:
    return list(PROVIDERS[provider]["env"])  # type: ignore[arg-type]


def parse_env_text(text: str) -> Dict[str, str]:
    """Read KEY=value lines (a .env file) and return {provider name: key} for known providers."""
    found: Dict[str, str] = {}
    values: Dict[str, str] = {}
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        m = _LINE.match(line)
        if m:
            values[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    for name in PROVIDERS:
        for var in env_names(name):
            if values.get(var):
                found[name] = values[var]
                break
    return found


def key_from_environment(provider: str, environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    import os
    env = environ if environ is not None else os.environ
    for var in env_names(provider):
        if env.get(var):
            return env[var]
    return None
