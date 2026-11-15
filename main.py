#!/usr/bin/env python3
"""nurz aggregator — a free-model-aggregator-02 entry (rating-first routing).

One Python 3.13 server, standard library only, that serves the OpenAI Chat
Completions wire and the Anthropic Messages wire, answers every prompt with
a real completion from one of the fourteen free OpenRouter models dialled
through the grader's gateway, and exits 0 when the env says the work is
done.

Routing (the nurz strategy: rating-first with penalised failover):
  * the candidates that meet a prompt's requirements (tools / vision /
    JSON mode) are ordered by the brief's rating_milli, highest first;
  * a model that refuses (4xx/5xx) takes a penalty: -150 after one
    refusal, -650 after two refusals in a row, so it drops down the
    ranking; after three refusals in the trial it is treated as spent
    and is not dialled again. The penalised model is tried again on a
    later request, in its penalised place;
  * every refusal a retry or a failover recovers from is reported in the
    x-makefun-upstream-failures header, in the order the calls refused.

Every routing decision is a function of the brief, the requests and the
gateway's answers only: no wall clock, no unseeded randomness, so a
replay of the trial leaves the same answers.
"""

import json
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# ---------------------------------------------------------------- constants

PROTOCOL_VERSION = 1
POLL_EVERY_SEC = 0.1
MAX_CLIENT_BODY = 8 * 1024 * 1024
MAX_TRIES_PER_PROMPT = 8
GATEWAY_TIMEOUT_SEC = 25
REGISTER_RETRIES = 100
REGISTER_RETRY_WAIT_SEC = 0.2

# The routing penalties, in rating milli-points.
PENALTY_FIRST_REFUSAL = 150
PENALTY_REPEAT_REFUSAL = 650
# After this many refusals in one trial a model is spent.
REFUSAL_LIMIT = 3
# The gateway refuses an output cap above this.
MAX_OUTPUT_TOKENS = 8192

SCHEMA_PREFIX = "free-model-aggregator-02/"
PUBLIC_BASE_URL = "https://openrouter.ai/api/v1"
KEY_ENV_NAME = "OPENROUTER_API_KEY"
FIXED_CREATED = 1767225600
FIXED_CREATED_AT = "2026-01-01T00:00:00Z"
FAILURES_HEADER = "x-makefun-upstream-failures"
MAX_TRAIL_ROWS = 8

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")
GATEWAY_BASE = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
GATEWAY_KEY = os.environ.get("OPENAI_API_KEY", "")


# ---------------------------------------------------------------- small helpers


def is_obj(value):
    return isinstance(value, dict)


def as_int(value, default):
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return default


def as_str(value):
    return value if isinstance(value, str) else ""


def cap_tokens(value):
    """An output cap the gateway accepts: a whole number from 1 to 8192."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 1024
    n = int(value)
    if n < 1:
        return 1
    if n > MAX_OUTPUT_TOKENS:
        return MAX_OUTPUT_TOKENS
    return n


def log(message):
    sys.stderr.write(f"aggregator: {message}\n")
    sys.stderr.flush()


# ---------------------------------------------------------------- the brief


def json_objects_in(text):
    """Every top-level JSON object found in `text`, in the order they appear.

    The brief is prose with JSON documents embedded in it; each document is
    found by scanning for '{' and decoding the object that starts there.
    """
    decoder = json.JSONDecoder()
    at = 0
    while True:
        at = text.find("{", at)
        if at < 0:
            return
        try:
            value, end = decoder.raw_decode(text, at)
        except json.JSONDecodeError:
            at += 1
            continue
        if isinstance(value, dict):
            yield value
        at = end


def model_row(raw):
    """One model row of the brief, normalised to the shape the router uses."""
    if not is_obj(raw):
        return None
    raw_id = as_str(raw.get("id")) or as_str(raw.get("model"))
    if raw_id == "":
        return None
    model_id = raw_id if raw_id.startswith("openrouter/") else f"openrouter/{raw_id}"
    rating = raw.get("rating_milli")
    if not isinstance(rating, (int, float)) or isinstance(rating, bool):
        rating = 0
    context = raw.get("context_length")
    if not isinstance(context, (int, float)) or isinstance(context, bool):
        context = None
    else:
        context = int(context)
    return {
        "id": model_id,
        "provider": as_str(raw.get("provider")) or "openrouter",
        "display_name": as_str(raw.get("display_name")) or raw_id,
        "rating": int(rating),
        "context_length": context,
        "tools": raw.get("tools") if isinstance(raw.get("tools"), bool) else None,
        "vision": raw.get("vision") if isinstance(raw.get("vision"), bool) else None,
        "json": raw.get("json") if isinstance(raw.get("json"), bool) else None,
        "free": raw.get("free") is not False,
    }


def read_brief(path):
    """The brief's model list and prompt manifest, found by their schemas.

    Returns {"models": [...], "prompts": [...], "error": <str or None>}.
    """
    if path == "":
        return {"models": [], "prompts": [], "error": "MAKEFUN_PROMPT_FILE is not set"}
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        return {"models": [], "prompts": [], "error": f"cannot read the brief: {err}"}
    models = []
    prompts = []
    for doc in json_objects_in(text):
        schema = doc.get("schema")
        if not isinstance(schema, str) or not schema.startswith(SCHEMA_PREFIX):
            continue
        rows = doc.get("models")
        if rows is None:
            rows = doc.get("ratings")
        if isinstance(rows, list):
            models = [m for m in (model_row(r) for r in rows) if m is not None]
        rows = doc.get("prompts")
        if rows is None:
            rows = doc.get("manifest")
        if isinstance(rows, list):
            prompts = [p for p in rows if is_obj(p)]
    if not models:
        return {"models": [], "prompts": prompts, "error": "the brief holds no free-model list"}
    return {"models": models, "prompts": prompts, "error": None}


def build_catalog(models):
    """The catalog document: one provider, exactly the brief's free models."""
    seen = set()
    rows = []
    for model in models:
        if not model["free"] or model["id"] in seen:
            continue
        seen.add(model["id"])
        rows.append({"id": model["id"], "free": True, "context_length": model["context_length"]})
    if not rows:
        return {"providers": []}
    return {
        "providers": [
            {
                "id": "openrouter",
                "base_url": PUBLIC_BASE_URL,
                "key_env": KEY_ENV_NAME,
                "wire": "openai",
                "models": rows,
            }
        ]
    }


BRIEF = read_brief(PROMPT_FILE)
CATALOG = build_catalog(BRIEF["models"])
# The one canonical serialisation: --list prints it, register carries it.
CATALOG_TEXT = json.dumps(CATALOG, separators=(",", ":"), ensure_ascii=False)


def served_models():
    """The brief's free models, in the brief's order, as the lists name them."""
    return [m for m in BRIEF["models"] if m["free"]]


# ---------------------------------------------------------------- the router


def meets(model, need):
    """Whether a model's capabilities meet a prompt's requirements.

    An unknown capability never meets a requirement that names it.
    """
    if need.get("tools") is True and model["tools"] is not True:
        return False
    if need.get("vision") is True and model["vision"] is not True:
        return False
    if need.get("json") is True and model["json"] is not True:
        return False
    minimum = need.get("min_context")
    if minimum is not None and (model["context_length"] is None or model["context_length"] < minimum):
        return False
    return True


class Router:
    """Rating-first routing with a refusal ledger.

    The ledger is the trial's memory: each refusal lowers the model that
    made it, so the next prompt tries a better-ranked model first and the
    penalised one again only in its lowered place.
    """

    def __init__(self, models):
        self.models = list(models)
        self.ledger = {m["id"]: {"consec": 0, "total": 0} for m in models}
        self.lock = threading.Lock()

    def _score(self, model):
        entry = self.ledger[model["id"]]
        if entry["total"] >= REFUSAL_LIMIT:
            return None  # spent: three refusals in this trial
        penalty = 0
        if entry["consec"] == 1:
            penalty = PENALTY_FIRST_REFUSAL
        elif entry["consec"] >= 2:
            penalty = PENALTY_REPEAT_REFUSAL
        return model["rating"] - penalty

    def candidates(self, need):
        """The models that meet `need`, best effective rating first.

        Ties break on the model id, so the order is the same on a replay.
        """
        ranked = []
        for model in self.models:
            if not meets(model, need):
                continue
            score = self._score(model)
            if score is None:
                continue
            ranked.append((score, model["id"], model))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [model for _, _, model in ranked]

    def note_refusal(self, model_id):
        entry = self.ledger[model_id]
        entry["consec"] += 1
        entry["total"] += 1

    def note_success(self, model_id):
        self.ledger[model_id]["consec"] = 0


ROUTER = Router(served_models())


def needs_of_openai(body):
    """What a Chat Completions body requires of the model that serves it."""
    need = {}
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        need["tools"] = True
    fmt = body.get("response_format")
    if is_obj(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        need["json"] = True
    if _openai_has_image(body.get("messages")):
        need["vision"] = True
    return need


def needs_of_anthropic(body):
    """What a Messages body requires of the model that serves it."""
    need = {}
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        need["tools"] = True
    for message in body.get("messages") or []:
        if not is_obj(message) or not isinstance(message.get("content"), list):
            continue
        for block in message["content"]:
            if is_obj(block) and block.get("type") == "image":
                need["vision"] = True
    return need


def _openai_has_image(messages):
    if not isinstance(messages, list):
        return False
    for message in messages:
        if not is_obj(message) or not isinstance(message.get("content"), list):
            continue
        for part in message["content"]:
            if is_obj(part) and part.get("type") in ("image_url", "image"):
                return True
    return False


# ---------------------------------------------------------------- the gateway


class GatewayReply:
    """One gateway answer: its status and a body read whole or streamed."""

    def __init__(self, status, response):
        self.status = status
        self._response = response

    @property
    def ok(self):
        return 200 <= self.status < 300

    def iter_bytes(self):
        """The body's bytes as they arrive, never buffered whole (SSE streams)."""
        reader = getattr(self._response, "read1", None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def read_text(self):
        try:
            return self._response.read().decode("utf-8", "replace")
        finally:
            self.close()

    def close(self):
        try:
            self._response.close()
        except Exception:
            pass


def gateway_url():
    """The gateway's Chat Completions route, from the trial's base URL."""
    if GATEWAY_BASE == "":
        return ""
    if GATEWAY_BASE.endswith("/v1"):
        return f"{GATEWAY_BASE}/chat/completions"
    return f"{GATEWAY_BASE}/v1/chat/completions"


def dial(model_id, payload, stream):
    """POST one Chat Completions body to the gateway, naming `model_id`.

    A refusal comes back as a GatewayReply with its status, exactly as the
    provider sent it; only an unreachable gateway raises.
    """
    url = gateway_url()
    if url == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, sort_keys=True).encode("utf-8")
    headers = {"content-type": "application/json", "authorization": f"Bearer {GATEWAY_KEY}"}
    if stream:
        headers["accept"] = "text/event-stream"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=GATEWAY_TIMEOUT_SEC)
    except urllib.error.HTTPError as err:
        return GatewayReply(err.code, err)
    return GatewayReply(response.status, response)


def failure_code(text):
    """The error code of a refused gateway body, as the env reads it."""
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return ""
    if not is_obj(value) or not is_obj(value.get("error")):
        return ""
    error = value["error"]
    code = error.get("code")
    if not isinstance(code, str) or code == "":
        code = error.get("type")
    if not isinstance(code, str):
        return ""
    return code[:64]


# ---------------------------------------------------------------- request translation


def anthropic_text(content):
    """The text of a string or of a block array's text blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block["text"]
        for block in content
        if is_obj(block) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )


def tool_result_text(content):
    """A tool_result block's content as plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = anthropic_text(content)
        return text if text != "" else json.dumps(content, separators=(",", ":"), ensure_ascii=False)
    if content is None:
        return ""
    return json.dumps(content, separators=(",", ":"), ensure_ascii=False)


def anthropic_image_part(block):
    """One Anthropic image block as an OpenAI image_url part, else None."""
    source = block.get("source")
    if not is_obj(source):
        return None
    if source.get("type") == "base64" and isinstance(source.get("data"), str):
        media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
        return {"type": "image_url", "image_url": {"url": f"data:{media};base64,{source['data']}"}}
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def anthropic_to_openai(body, model_id, stream):
    """A Messages request as a Chat Completions request naming `model_id`.

    The system prompt becomes a system message, image blocks become
    image_url parts, tool_use blocks become tool_calls, tool_result blocks
    become tool messages, and tools, tool_choice, the output cap,
    temperature, top_p and stop sequences pass through.
    """
    messages = []
    system = anthropic_text(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") or []:
        if not is_obj(raw):
            continue
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if raw.get("role") == "assistant" else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if raw.get("role") == "assistant":
            calls = []
            for block in content:
                if not is_obj(block) or block.get("type") != "tool_use":
                    continue
                calls.append(
                    {
                        "id": block.get("id") if isinstance(block.get("id"), str) else "",
                        "type": "function",
                        "function": {
                            "name": block.get("name") if isinstance(block.get("name"), str) else "",
                            "arguments": json.dumps(
                                block.get("input") if is_obj(block.get("input")) else {},
                                separators=(",", ":"),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = anthropic_text(content)
            message = {"role": "assistant", "content": text if text != "" else None}
            if calls:
                message["tool_calls"] = calls
            messages.append(message)
            continue
        parts = []
        for block in content:
            if not is_obj(block):
                continue
            if block.get("type") == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                        "content": tool_result_text(block.get("content")),
                    }
                )
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"type": "text", "text": block["text"]})
                continue
            part = anthropic_image_part(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model_id, "messages": messages}
    out["max_tokens"] = cap_tokens(body.get("max_tokens"))
    for key in ("temperature", "top_p"):
        value = body.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        out["stop"] = stops
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        functions = []
        for raw in tools:
            if not is_obj(raw) or not isinstance(raw.get("name"), str):
                continue
            functions.append(
                {
                    "type": "function",
                    "function": {
                        "name": raw["name"],
                        "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                        "parameters": raw.get("input_schema")
                        if is_obj(raw.get("input_schema"))
                        else {"type": "object", "properties": {}},
                    },
                }
            )
        if functions:
            out["tools"] = functions
    choice = body.get("tool_choice")
    if is_obj(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if stream:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def openai_body(body, model_id, stream):
    """A Chat Completions body naming `model_id`, with a capped output."""
    out = {"model": model_id}
    for key in ("messages", "temperature", "top_p", "stop", "tools", "tool_choice", "response_format", "seed"):
        if key in body:
            out[key] = body[key]
    cap = body.get("max_tokens")
    if cap is None:
        cap = body.get("max_completion_tokens")
    out["max_tokens"] = cap_tokens(cap)
    if stream:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


# ---------------------------------------------------------------- response translation


def anthropic_stop(finish_reason):
    """An OpenAI finish_reason as an Anthropic stop_reason."""
    if finish_reason == "length":
        return "max_tokens"
    if finish_reason in ("tool_calls", "function_call"):
        return "tool_use"
    if finish_reason == "content_filter":
        return "refusal"
    return "end_turn"


def args_object(value):
    """A tool call's arguments as an object: the gateway's JSON string, or {}."""
    if is_obj(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        if is_obj(parsed):
            return parsed
    return {}


def fix_openai_answer(answer, model_id):
    """A Chat Completions answer with the dialled model named on it.

    The shape the wire checks read: an assistant message, a finish reason,
    and usage with an output count of at least 1 whenever text was made.
    """
    if not is_obj(answer):
        answer = {}
    choices = answer.get("choices") if isinstance(answer.get("choices"), list) else []
    if not choices or not is_obj(choices[0]):
        choices = [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]
        answer["choices"] = choices
    choice = choices[0]
    message = choice.get("message") if is_obj(choice.get("message")) else {}
    message["role"] = "assistant"
    choice["message"] = message
    if not isinstance(choice.get("finish_reason"), str):
        choice["finish_reason"] = "stop"
    usage = answer.get("usage") if is_obj(answer.get("usage")) else {}
    text = message.get("content")
    made_text = isinstance(text, str) and text != ""
    prompt_tokens = usage.get("prompt_tokens")
    prompt_tokens = prompt_tokens if isinstance(prompt_tokens, int) and prompt_tokens >= 0 else 0
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens < 1:
        completion_tokens = 1 if made_text else 0
    answer["usage"] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    answer.setdefault("object", "chat.completion")
    answer.setdefault("id", "chatcmpl-aggregate")
    answer["model"] = model_id
    return answer


def openai_to_anthropic(answer, model_id, declared_tools):
    """A Chat Completions answer as an Anthropic message naming `model_id`."""
    choices = answer.get("choices") if isinstance(answer.get("choices"), list) else []
    choice = choices[0] if choices and is_obj(choices[0]) else {}
    message = choice.get("message") if is_obj(choice.get("message")) else {}
    content = []
    text = message.get("content")
    if isinstance(text, str) and text != "":
        content.append({"type": "text", "text": text})
    elif isinstance(text, list):
        for part in text:
            if is_obj(part) and part.get("type") == "text" and isinstance(part.get("text"), str):
                content.append({"type": "text", "text": part["text"]})
    for call in message.get("tool_calls") or []:
        if not is_obj(call):
            continue
        fn = call.get("function") if is_obj(call.get("function")) else {}
        name = fn.get("name") if isinstance(fn.get("name"), str) else ""
        if declared_tools is not None and name not in declared_tools:
            continue  # never name a tool the client did not declare
        content.append(
            {
                "type": "tool_use",
                "id": call.get("id") if isinstance(call.get("id"), str) and call.get("id") != "" else "toolu_unknown",
                "name": name,
                "input": args_object(fn.get("arguments")),
            }
        )
    if not content:
        content.append({"type": "text", "text": ""})
    usage = answer.get("usage") if is_obj(answer.get("usage")) else {}
    prompt_tokens = usage.get("prompt_tokens")
    prompt_tokens = prompt_tokens if isinstance(prompt_tokens, int) and prompt_tokens >= 0 else 0
    completion_tokens = usage.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens < 1:
        completion_tokens = 1 if any(b.get("type") == "text" and b.get("text") != "" for b in content) else 0
    answer_id = answer.get("id") if isinstance(answer.get("id"), str) and answer.get("id") != "" else "aggregate"
    return {
        "id": f"msg_{answer_id}",
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content,
        "stop_reason": anthropic_stop(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
    }


def declared_tool_names(body):
    """The tool names a Messages body declares, or None when it declares none."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return None
    names = {t["name"] for t in tools if is_obj(t) and isinstance(t.get("name"), str)}
    return names if names else None


# ---------------------------------------------------------------- server-sent events


def sse_event(event, data):
    """One server-sent event: an `event:` line when named, then the data line."""
    prefix = "" if event is None else f"event: {event}\n"
    return f"{prefix}data: {json.dumps(data, separators=(',', ':'), ensure_ascii=False)}\n\n"


def split_frames(chunks):
    """SSE frames from an upstream body's byte chunks, split on blank lines.

    A frame cut across reads is held until its end arrives; CRLF is read
    as LF, as the SSE grammar reads it.
    """
    buffer = ""
    for chunk in chunks:
        buffer += chunk.decode("utf-8", "replace").replace("\r\n", "\n")
        at = buffer.find("\n\n")
        while at != -1:
            yield buffer[:at]
            buffer = buffer[at + 2 :]
            at = buffer.find("\n\n")
    if buffer.strip() != "":
        yield buffer


def openai_stream_frames(chunks, model_id):
    """A Chat Completions SSE stream as it came, the dialled id on each chunk."""
    for frame in split_frames(chunks):
        lines = []
        for line in frame.split("\n"):
            trimmed = line.strip()
            if not trimmed.startswith("data:"):
                lines.append(line)
                continue
            payload = trimmed[5:].strip()
            if payload == "" or payload == "[DONE]":
                lines.append(line)
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                lines.append(line)
                continue
            if is_obj(chunk):
                chunk = {**chunk, "model": model_id}
                lines.append(f"data: {json.dumps(chunk, separators=(',', ':'), ensure_ascii=False)}")
            else:
                lines.append(line)
        yield "\n".join(lines) + "\n\n"


def anthropic_stream_events(chunks, model_id, declared_tools):
    """A Chat Completions SSE stream as the Anthropic event sequence.

    message_start first, one content block per text run or tool call,
    message_delta with the stop reason and the output count, then
    message_stop — the order the Messages wire reads.
    """
    started = False
    finished = False
    text_block = None
    next_block = 0
    tool_blocks = {}
    suppressed_tools = set()
    pending_stop = None
    input_tokens = 0
    output_tokens = 0

    def start_events():
        nonlocal started
        if started:
            return
        started = True
        yield sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": model_id,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        )

    def close_events():
        nonlocal text_block
        if text_block is not None:
            yield sse_event("content_block_stop", {"type": "content_block_stop", "index": text_block})
            text_block = None
        for index in sorted(tool_blocks.values()):
            yield sse_event("content_block_stop", {"type": "content_block_stop", "index": index})
        tool_blocks.clear()

    def finish_events(stop_reason):
        nonlocal finished
        if finished:
            return
        finished = True
        yield from close_events()
        yield sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens},
            },
        )
        yield sse_event("message_stop", {"type": "message_stop"})

    for frame in split_frames(chunks):
        for line in frame.split("\n"):
            trimmed = line.strip()
            if not trimmed.startswith("data:"):
                continue
            payload = trimmed[5:].strip()
            if payload == "[DONE]":
                yield from start_events()
                yield from finish_events(pending_stop if pending_stop is not None else "end_turn")
                finished = True
                continue
            try:
                chunk = json.loads(payload)
            except (json.JSONDecodeError, ValueError):
                continue
            if not is_obj(chunk):
                continue
            usage = chunk.get("usage")
            if is_obj(usage):
                if isinstance(usage.get("prompt_tokens"), int):
                    input_tokens = usage["prompt_tokens"]
                if isinstance(usage.get("completion_tokens"), int):
                    output_tokens = usage["completion_tokens"]
            choices = chunk.get("choices") if isinstance(chunk.get("choices"), list) else []
            if not choices or not is_obj(choices[0]):
                continue
            choice = choices[0]
            delta = choice.get("delta") if is_obj(choice.get("delta")) else {}
            yield from start_events()
            text = delta.get("content")
            if isinstance(text, str) and text != "":
                if text_block is None:
                    text_block = next_block
                    next_block += 1
                    yield sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": text_block,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                yield sse_event(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": text_block, "delta": {"type": "text_delta", "text": text}},
                )
            for call in delta.get("tool_calls") or []:
                if not is_obj(call):
                    continue
                upstream_index = call.get("index")
                key = upstream_index if isinstance(upstream_index, int) else len(tool_blocks) + len(suppressed_tools)
                if key in suppressed_tools:
                    continue
                fn = call.get("function") if is_obj(call.get("function")) else {}
                name = fn.get("name") if isinstance(fn.get("name"), str) else ""
                local = tool_blocks.get(key)
                if local is None:
                    if declared_tools is not None and name != "" and name not in declared_tools:
                        suppressed_tools.add(key)
                        continue
                    if text_block is not None:
                        yield sse_event("content_block_stop", {"type": "content_block_stop", "index": text_block})
                        text_block = None
                    local = next_block
                    next_block += 1
                    tool_blocks[key] = local
                    yield sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": local,
                            "content_block": {
                                "type": "tool_use",
                                "id": call.get("id") if isinstance(call.get("id"), str) and call.get("id") != "" else f"toolu_{local}",
                                "name": name,
                                "input": {},
                            },
                        },
                    )
                arguments = fn.get("arguments")
                if isinstance(arguments, str) and arguments != "":
                    yield sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": local,
                            "delta": {"type": "input_json_delta", "partial_json": arguments},
                        },
                    )
            if isinstance(choice.get("finish_reason"), str):
                pending_stop = anthropic_stop(choice["finish_reason"])
    if not finished:
        if output_tokens < 1 and text_block is not None:
            output_tokens = 1
        yield from start_events()
        yield from finish_events(pending_stop if pending_stop is not None else "end_turn")


# ---------------------------------------------------------------- serving one prompt


def trail_header(failures):
    """The x-makefun-upstream-failures value for a prompt's refusals."""
    rows = [
        {"status": f["status"], "code": f["code"], "model": f["model"]}
        for f in failures[:MAX_TRAIL_ROWS]
    ]
    return json.dumps(rows, separators=(",", ":"), ensure_ascii=False)


def serve(wire, client_body, build_upstream, need):
    """Dial the prompt until a model answers or the candidates run out.

    Returns {"ok": True, "model": <id>, "reply": <GatewayReply>,
    "failures": [...]} on a served prompt, else {"ok": False,
    "failures": [...], "last_status": <int>, "last_text": <str>}.
    A model is tried at most once per prompt: a refusal penalises it and
    the failover moves on to the next candidate; the penalised model is
    tried again on a later prompt, in its penalised place.
    The failover loop is one lock: the env sends one prompt at a time, and
    the ledger must move in the order the calls were made.
    """
    failures = []
    tried_ids = set()
    with ROUTER.lock:
        tried = 0
        while tried < MAX_TRIES_PER_PROMPT:
            candidates = [m for m in ROUTER.candidates(need) if m["id"] not in tried_ids]
            if not candidates:
                break
            model = candidates[0]
            tried_ids.add(model["id"])
            try:
                reply = dial(model["id"], build_upstream(model["id"]), client_body.get("stream") is True)
            except Exception as err:
                failures.append({"status": 502, "code": "upstream_unreachable", "model": model["id"]})
                ROUTER.note_refusal(model["id"])
                tried += 1
                log(f"the gateway could not be reached for {model['id']}: {err}")
                continue
            tried += 1
            if reply.ok:
                ROUTER.note_success(model["id"])
                return {"ok": True, "model": model["id"], "reply": reply, "failures": failures}
            text = reply.read_text()
            code = failure_code(text)
            failures.append({"status": reply.status, "code": code, "model": model["id"]})
            ROUTER.note_refusal(model["id"])
            log(f"{model['id']} refused HTTP {reply.status} ({code or 'no code'})")
            if reply.status == 401:
                # No key was pushed: every model is refused the same way.
                return {"ok": False, "failures": failures, "last_status": reply.status, "last_text": text}
        last = failures[-1] if failures else {"status": 502, "code": "no_free_model", "model": None}
        return {
            "ok": False,
            "failures": failures,
            "last_status": last["status"],
            "last_text": "",
        }


# ---------------------------------------------------------------- the HTTP server


def openai_list(models):
    """GET /v1/models on an OpenAI-shaped request."""
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"]}
            for m in models
        ],
    }


def anthropic_list(models):
    """GET /v1/models on an Anthropic-shaped request (an x-api-key header)."""
    return {
        "data": [
            {"type": "model", "id": m["id"], "display_name": m["display_name"], "created_at": FIXED_CREATED_AT}
            for m in models
        ],
        "has_more": False,
    }


def openai_error(message, error_type):
    return {"error": {"message": message, "type": error_type, "code": None}}


def anthropic_error(error_type, message):
    return {"type": "error", "error": {"type": error_type, "message": message}}


class Handler(BaseHTTPRequestHandler):
    """The entry's one server: the models route, the two completion wires."""

    protocol_version = "HTTP/1.1"
    server_version = "nurz-aggregator/1"
    sys_version = ""

    def log_message(self, *args):  # the base class's name; stderr stays ours
        pass

    def do_GET(self):  # the base class's name
        self.dispatch()

    def do_POST(self):  # the base class's name
        self.dispatch()

    # -- plumbing -------------------------------------------------------------

    def dispatch(self):
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == "GET" and path == "/v1/models":
                self.models()
                return
            if self.command == "POST" and path == "/v1/chat/completions":
                self.chat()
                return
            if self.command == "POST" and path == "/v1/messages":
                self.messages()
                return
            wire = "anthropic" if path == "/v1/messages" else "openai"
            self.send_error(wire, 404, "not_found", f"no route {self.command} {path}")
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = "anthropic" if urlsplit(self.path).path.startswith("/v1/messages") else "openai"
                try:
                    self.send_error(wire, 500, "internal_error", "the server failed to answer")
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_json(self, status, body, extra_headers=None):
        raw = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(raw)))
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error(self, wire, status, error_type, message):
        if wire == "anthropic":
            self.send_json(status, anthropic_error(error_type, message))
        else:
            self.send_json(status, openai_error(message, error_type))

    def stream_headers(self, extra_headers=None):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text):
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def read_json(self):
        """One client request body as an object, or None when it is not one."""
        length = self.headers.get("content-length")
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_CLIENT_BODY:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b""
        if not raw:
            return None
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None
        return value if is_obj(value) else None

    def refused(self, wire, result):
        """A prompt no model served, as a 502 in the client's wire, with the trail."""
        status = result["last_status"]
        headers = {}
        if result["failures"]:
            headers[FAILURES_HEADER] = trail_header(result["failures"])
        detail = f"the gateway answered HTTP {status} and no free model served this prompt"
        if wire == "anthropic":
            self.send_json(502, anthropic_error("api_error", detail), headers)
        else:
            self.send_json(502, openai_error(detail, "upstream_error"), headers)

    # -- the routes -----------------------------------------------------------

    def models(self):
        listing = served_models()
        if self.headers.get("x-api-key") is not None:
            self.send_json(200, anthropic_list(listing))
        else:
            self.send_json(200, openai_list(listing))

    def chat(self):
        body = self.read_json()
        if body is None:
            self.send_error("openai", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        stream = body.get("stream") is True
        result = serve("openai", body, lambda model_id: openai_body(body, model_id, stream), needs_of_openai(body))
        if not result["ok"]:
            self.refused("openai", result)
            return
        reply = result["reply"]
        model_id = result["model"]
        headers = {}
        if result["failures"]:
            headers[FAILURES_HEADER] = trail_header(result["failures"])
        if stream:
            self.stream_headers(headers)
            try:
                for frame in openai_stream_frames(reply.iter_bytes(), model_id):
                    self.write_event(frame)
            finally:
                reply.close()
            return
        text = reply.read_text()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        self.send_json(200, fix_openai_answer(parsed if is_obj(parsed) else {}, model_id), headers)

    def messages(self):
        body = self.read_json()
        if body is None:
            self.send_error("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
            return
        stream = body.get("stream") is True
        declared = declared_tool_names(body)
        result = serve(
            "anthropic",
            body,
            lambda model_id: anthropic_to_openai(body, model_id, stream),
            needs_of_anthropic(body),
        )
        if not result["ok"]:
            self.refused("anthropic", result)
            return
        reply = result["reply"]
        model_id = result["model"]
        headers = {}
        if result["failures"]:
            headers[FAILURES_HEADER] = trail_header(result["failures"])
        if stream:
            self.stream_headers(headers)
            try:
                for event in anthropic_stream_events(reply.iter_bytes(), model_id, declared):
                    self.write_event(event)
            finally:
                reply.close()
            return
        text = reply.read_text()
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            parsed = None
        self.send_json(200, openai_to_anthropic(parsed if is_obj(parsed) else {}, model_id, declared), headers)


# ---------------------------------------------------------------- register and poll


def register(port):
    """POST /register with the port and the one catalog; retry until the env answers."""
    payload = json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": CATALOG},
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{ENV_URL}/register",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    last_error = None
    for _ in range(REGISTER_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                status = response.status
                text = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as err:
            raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {err.code}") from err
        except Exception as err:  # the env is not up yet: try again
            last_error = err
            time.sleep(REGISTER_RETRY_WAIT_SEC)
            continue
        if not 200 <= status < 300:
            raise RuntimeError(f"POST {ENV_URL}/register answered HTTP {status}: {text[:500]}")
        try:
            reply = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            reply = None
        if is_obj(reply) and isinstance(reply.get("requests"), int):
            log(f"registered: the trial holds {reply['requests']} prompts")
        return
    raise RuntimeError(f"POST {ENV_URL}/register never answered: {last_error}")


def poll():
    """Poll GET /work; exit 0 the moment it says done."""
    while True:
        time.sleep(POLL_EVERY_SEC)
        try:
            with urllib.request.urlopen(f"{ENV_URL}/work", timeout=2) as response:
                if response.status != 200:
                    continue
                work = json.loads(response.read().decode("utf-8", "replace"))
        except Exception:
            continue
        if is_obj(work) and work.get("done") is True:
            sys.stderr.flush()
            os._exit(0)


def stop(*_):
    """SIGTERM and SIGINT end the trial with exit 0, at once."""
    sys.stderr.flush()
    os._exit(0)


# ---------------------------------------------------------------- start


def start():
    if BRIEF["error"] is not None:
        log(BRIEF["error"])
    if "--list" in sys.argv[1:]:
        # The same catalog document the register body carries, byte for byte.
        sys.stdout.write(CATALOG_TEXT + "\n")
        sys.stdout.flush()
        return
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    if gateway_url() == "":
        raise RuntimeError("OPENAI_BASE_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
        log(f"listening on 0.0.0.0:{port} with {len(served_models())} free models")
        register(port)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name="work-poll", daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main():
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
