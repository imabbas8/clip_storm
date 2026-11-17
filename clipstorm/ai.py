"""The only file that talks to an AI service. Bring your own key; every service is asked for the same thing:
a JSON object that follows a small schema.

Claude goes through the official Anthropic SDK. Gemini has its own JSON-schema mode. Everything else (OpenAI, Groq,
OpenRouter, Grok, Ollama, any custom address) speaks the OpenAI chat format."""
import json
import re

import httpx

from . import config

# name -> label, address, dialect, where to get a key, suggested models (first is the default)
PROVIDERS = {
    "anthropic": ("Claude (Anthropic)", "", "anthropic", "https://console.anthropic.com/settings/keys",
                  ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"]),
    "gemini": ("Google Gemini", "https://generativelanguage.googleapis.com/v1beta", "gemini",
               "https://aistudio.google.com/apikey", ["gemini-2.5-flash"]),
    "openai": ("OpenAI", "https://api.openai.com/v1", "openai", "https://platform.openai.com/api-keys", ["gpt-4o-mini"]),
    "groq": ("Groq (fast, free tier)", "https://api.groq.com/openai/v1", "openai", "https://console.groq.com/keys",
             ["llama-3.3-70b-versatile"]),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1", "openai", "https://openrouter.ai/keys",
                   ["anthropic/claude-haiku-5.5"]),
    "xai": ("Grok (xAI)", "https://api.x.ai/v1", "openai", "https://console.x.ai", []),
    "ollama": ("Ollama (on this machine, no key)", "http://127.0.0.1:11434/v1", "openai", "https://ollama.com", ["llama3.1"]),
    "custom": ("Custom (OpenAI-compatible address)", "", "openai", "", []),
}
NO_KEY = ("ollama", "custom")
# Claude models that can hand a declined request to another model inside the same call
_FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5")
TIMEOUT = httpx.Timeout(120.0, connect=15.0)


class AIError(Exception):
    """A message a person can act on: what went wrong with the key, the model or the service."""


def _cfg(over=None) -> dict:
    saved = config.get()
    over = {k: v for k, v in (over or {}).items() if v not in (None, "") and set(str(v)) != {"•"}}
    s = {**saved, **over}
    p = s["ai_provider"] if s["ai_provider"] in PROVIDERS else "anthropic"
    label, base, dialect, _, suggested = PROVIDERS[p]
    if p != saved["ai_provider"]:                    # trying another service: the saved model and key belong to the old one
        s["ai_model"] = over.get("ai_model", "")
        s["ai_key"] = over.get("ai_key", "")
    own_address = s["ai_base_url"].strip().rstrip("/") if p in ("custom", "ollama") else ""
    return {"provider": p, "label": label, "dialect": dialect, "key": s["ai_key"].strip(), "base": own_address or base,
            "model": s["ai_model"].strip() or (suggested[0] if suggested else "")}


def ready() -> bool:
    c = _cfg()
    return bool(c["model"]) and (bool(c["key"]) or c["provider"] in NO_KEY) and (bool(c["base"]) or c["dialect"] == "anthropic")


def _explain(status: int, body: str, c: dict) -> str:
    body = re.sub(r"\s+", " ", body or "")[:300]
    if status in (401, 403):
        return f"{c['label']} refused the key (HTTP {status}). Check it was copied whole and is still active. {body}"
    if status == 404:
        return f"{c['label']} does not know the model \"{c['model']}\" (HTTP 404). Press Find models and pick one. {body}"
    if status == 429:
        return f"{c['label']} says the limit is reached or there is no credit left (HTTP 429). {body}"
    return f"{c['label']} answered HTTP {status}. {body}"


def _parse(text: str):
    """JSON from a chat answer, even when the model wrapped it in ``` fences or added a sentence around it."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise AIError("The AI did not answer with JSON. Try another model.") from None
        try:
            return json.loads(m.group(0))
        except ValueError:
            raise AIError("The AI's JSON could not be read. Try another model.") from None


# ---------------------------------------------------------------- Claude, through the official SDK
def _anthropic(c, prompt, schema, max_tokens):
    import anthropic
    client = anthropic.Anthropic(api_key=c["key"] or None, timeout=120.0, max_retries=2)
    args = dict(model=c["model"], max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": schema}})
    try:
        if c["model"].startswith(_FALLBACK_MODELS):
            try:   # a declined request is re-run on another model by the service itself
                r = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **args)
            except anthropic.BadRequestError as e:
                if "fallback" not in str(e).lower():
                    raise
                r = client.messages.create(**args)
        else:
            r = client.messages.create(**args)
    except anthropic.AuthenticationError as e:
        raise AIError(_explain(401, e.message, c)) from e
    except anthropic.PermissionDeniedError as e:
        raise AIError(_explain(403, e.message, c)) from e
    except anthropic.NotFoundError as e:
        raise AIError(_explain(404, e.message, c)) from e
    except anthropic.RateLimitError as e:
        raise AIError(_explain(429, e.message, c)) from e
    except anthropic.APIStatusError as e:
        raise AIError(_explain(e.status_code, e.message, c)) from e
    except anthropic.APIConnectionError as e:
        raise AIError("Could not reach Claude. Check the internet connection.") from e
    if r.stop_reason == "refusal":
        raise AIError("Claude declined this request.")
    if r.stop_reason == "max_tokens":
        raise AIError("The AI's answer was cut off. Try again.")
    text = next((b.text for b in r.content if b.type == "text"), "")
    return _parse(text), r.usage.input_tokens + r.usage.output_tokens


# ---------------------------------------------------------------- Gemini
def _gemini_schema(s):
    """Gemini's schema dialect: upper-case type names and no additionalProperties."""
    if isinstance(s, dict):
        return {k: (v.upper() if k == "type" and isinstance(v, str) else _gemini_schema(v))
                for k, v in s.items() if k != "additionalProperties"}
    if isinstance(s, list):
        return [_gemini_schema(x) for x in s]
    return s


def _gemini(c, prompt, schema, max_tokens):
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": _gemini_schema(schema),
                                 "maxOutputTokens": max_tokens}}
    try:
        r = httpx.post(f"{c['base']}/models/{c['model']}:generateContent", headers={"x-goog-api-key": c["key"]},
                       json=body, timeout=TIMEOUT)
    except httpx.HTTPError as e:
        raise AIError(f"Could not reach Gemini: {type(e).__name__}") from e
    if r.status_code != 200:
        raise AIError(_explain(r.status_code, r.text, c))
    d = r.json()
    try:
        text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
    except (KeyError, IndexError):
        raise AIError("Gemini returned no answer (it may have blocked the request).") from None
    return _parse(text), int(d.get("usageMetadata", {}).get("totalTokenCount", 0))


# ---------------------------------------------------------------- everything that speaks the OpenAI chat format
def _openai(c, prompt, schema, max_tokens):
    msg = (prompt + "\n\nAnswer with ONE JSON object and nothing else. It must follow this JSON schema:\n"
           + json.dumps(schema))
    body = {"model": c["model"], "messages": [{"role": "user", "content": msg}],
            "response_format": {"type": "json_object"}}
    headers = {"Authorization": f"Bearer {c['key']}"} if c["key"] else {}
    try:
        r = httpx.post(f"{c['base']}/chat/completions", headers=headers, json=body, timeout=TIMEOUT)
        if r.status_code == 400 and "response_format" in r.text:      # an older server without JSON mode
            body.pop("response_format")
            r = httpx.post(f"{c['base']}/chat/completions", headers=headers, json=body, timeout=TIMEOUT)
    except httpx.HTTPError as e:
        raise AIError(f"Could not reach {c['label']} at {c['base']}: {type(e).__name__}") from e
    if r.status_code != 200:
        raise AIError(_explain(r.status_code, r.text, c))
    d = r.json()
    try:
        text = d["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise AIError(f"{c['label']} returned no answer.") from None
    return _parse(text), int((d.get("usage") or {}).get("total_tokens", 0))


_CALL = {"anthropic": _anthropic, "gemini": _gemini, "openai": _openai}


def ask(prompt: str, schema: dict, max_tokens: int = 8000, over=None):
    """Returns (the parsed JSON object, tokens used). Raises AIError with a plain-language reason."""
    c = _cfg(over)
    if not c["model"]:
        raise AIError("Choose an AI model first (Settings, AI).")
    if not c["key"] and c["provider"] not in NO_KEY:
        raise AIError("Add your AI key first (Settings, AI).")
    if not c["base"] and c["dialect"] != "anthropic":
        raise AIError("Add the address of your AI service (Settings, AI).")
    return _CALL[c["dialect"]](c, prompt, schema, max_tokens)


def test(over=None) -> dict:
    """A real request with the given (or saved) settings, so 'connected' means it truly works."""
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}, "word": {"type": "string"}},
              "required": ["ok", "word"], "additionalProperties": False}
    c = _cfg(over)
    try:
        out, tokens = ask('Reply with ok set to true and word set to "connected".', schema, 500, over)
    except AIError as e:
        return {"ok": False, "message": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"Something went wrong: {str(e)[:200]}"}
    if not isinstance(out, dict):
        return {"ok": False, "message": "The model answered, but not in the JSON shape ClipStorm needs. Try another model."}
    return {"ok": True, "message": f"Connected to {c['label']} with {c['model']}. ({tokens} tokens used for this test.)"}


def models(over=None) -> dict:
    """The service's own list of model ids, for the Find models button."""
    c = _cfg(over)
    try:
        if c["dialect"] == "anthropic":
            import anthropic
            ids = [m.id for m in anthropic.Anthropic(api_key=c["key"] or None, timeout=30.0).models.list(limit=100)]
        elif c["dialect"] == "gemini":
            r = httpx.get(f"{c['base']}/models?pageSize=200", headers={"x-goog-api-key": c["key"]}, timeout=30)
            if r.status_code != 200:
                return {"ok": False, "message": _explain(r.status_code, r.text, c), "models": []}
            ids = [m["name"].split("/")[-1] for m in r.json().get("models", [])
                   if "generateContent" in m.get("supportedGenerationMethods", [])]
        else:
            r = httpx.get(f"{c['base']}/models", headers={"Authorization": f"Bearer {c['key']}"} if c["key"] else {}, timeout=30)
            if r.status_code != 200:
                return {"ok": False, "message": _explain(r.status_code, r.text, c), "models": []}
            ids = sorted(m["id"] for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id"))
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": f"Could not get the list: {str(e)[:200]}", "models": []}
    return {"ok": True, "models": ids, "message": f"{len(ids)} models found."}


def catalog() -> list:
    return [{"id": k, "label": v[0], "key_url": v[3], "models": v[4], "needs_key": k not in NO_KEY,
             "needs_url": k == "custom"} for k, v in PROVIDERS.items()]
