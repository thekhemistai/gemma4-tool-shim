"""
Gemma 4 Tool Shim

FastAPI proxy that translates raw Gemma-style tool call tokens from a local
OpenAI-compatible upstream into OpenAI `tool_calls[]` responses.
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

DEFAULT_UPSTREAM_URL = "http://localhost:8080"
DEFAULT_SHIM_HOST = "127.0.0.1"
DEFAULT_SHIM_PORT = 8083
MAX_REQUEST_BODY_BYTES = 4 * 1024 * 1024
MAX_UPSTREAM_ERROR_BYTES = 64 * 1024
MAX_SSE_BUFFER_BYTES = 8 * 1024 * 1024
MAX_TOOL_CONTENT_BYTES = 512 * 1024
MAX_TOOL_ARG_BYTES = 128 * 1024
MAX_TOOL_CALLS = 32
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=30.0, pool=30.0)

TOOL_BLOCK_RE = re.compile(r"<\|tool_call\>(.*?)<tool_call\|>", re.DOTALL)
CALL_NAME_RE = re.compile(r"call:([A-Za-z_][A-Za-z0-9_]*)\s*\{")
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def get_upstream_url() -> str:
    raw = os.getenv("UPSTREAM_URL", DEFAULT_UPSTREAM_URL).strip()
    parsed = urlparse(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError("UPSTREAM_URL must be an absolute http(s) URL")
    if parsed.path not in {"", "/"} or parsed.params or parsed.query or parsed.fragment:
        raise RuntimeError("UPSTREAM_URL must not include a path, query, or fragment")
    return raw.rstrip("/")


UPSTREAM_URL = get_upstream_url()
SHIM_HOST = os.getenv("SHIM_HOST", DEFAULT_SHIM_HOST)
SHIM_PORT = int(os.getenv("SHIM_PORT", str(DEFAULT_SHIM_PORT)))

app = FastAPI()
client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await client.aclose()


def sanitize_headers(headers: Any) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and "\r" not in value and "\n" not in value
    }


def limited_bytes_preview(data: bytes, max_bytes: int = MAX_UPSTREAM_ERROR_BYTES) -> bytes:
    return data[:max_bytes]


async def read_request_json(request: Request) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from exc

    body = await request.body()
    if len(body) > MAX_REQUEST_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    return payload


async def read_request_body(request: Request) -> bytes:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_REQUEST_BODY_BYTES:
                raise HTTPException(status_code=413, detail="Request body too large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid Content-Length header") from exc

    body = await request.body()
    if len(body) > MAX_REQUEST_BODY_BYTES:
        raise HTTPException(status_code=413, detail="Request body too large")
    return body


def parse_balanced_object(text: str, start: int) -> tuple[str, int] | None:
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(text)):
        char = text[idx]

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1], idx + 1

    return None


def repair_json_keys(value: str) -> str:
    return re.sub(r'(\{|,)\s*([a-zA-Z_]\w*)\s*:', r'\1"\2":', value)


def coerce_arguments(args_str: str) -> dict[str, Any]:
    args_str = args_str.strip()
    if not args_str:
        return {}
    if len(args_str.encode("utf-8")) > MAX_TOOL_ARG_BYTES:
        return {"raw": args_str[:MAX_TOOL_ARG_BYTES]}

    args_str = args_str.replace('<|"|>', '"').replace("<|'|>", "'")

    try:
        value = json.loads(args_str)
        return value if isinstance(value, dict) else {"raw": args_str}
    except json.JSONDecodeError:
        pass

    repaired_args_str = repair_json_keys(args_str)
    if repaired_args_str != args_str:
        try:
            value = json.loads(repaired_args_str)
            return value if isinstance(value, dict) else {"raw": args_str}
        except json.JSONDecodeError:
            pass

    try:
        value = ast.literal_eval(args_str)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return {"raw": args_str}

    return value if isinstance(value, dict) else {"raw": args_str}


def extract_tool_segments(content: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if len(content.encode("utf-8")) > MAX_TOOL_CONTENT_BYTES:
        return segments

    for match in TOOL_BLOCK_RE.finditer(content):
        inner = match.group(1)
        inner_match = CALL_NAME_RE.search(inner)
        if not inner_match:
            continue

        args = parse_balanced_object(inner, inner_match.end() - 1)
        if not args:
            continue

        args_str, _ = args
        segments.append(
            {
                "name": inner_match.group(1),
                "args_str": args_str,
                "span": match.span(),
            }
        )
        if len(segments) >= MAX_TOOL_CALLS:
            return segments

    if segments:
        return segments

    for match in CALL_NAME_RE.finditer(content):
        args = parse_balanced_object(content, match.end() - 1)
        if not args:
            continue

        args_str, end = args
        segments.append(
            {
                "name": match.group(1),
                "args_str": args_str,
                "span": (match.start(), end),
            }
        )
        if len(segments) >= MAX_TOOL_CALLS:
            break

    return segments


def parse_tool_calls(content: str) -> list[dict[str, Any]]:
    calls = []
    for i, segment in enumerate(extract_tool_segments(content)):
        arguments = coerce_arguments(segment["args_str"])
        calls.append(
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": segment["name"],
                    "arguments": json.dumps(arguments),
                },
            }
        )
    return calls


def strip_tool_tokens(content: str) -> str:
    segments = extract_tool_segments(content)
    if not segments:
        return content.strip()

    parts = []
    last = 0
    for segment in segments:
        start, end = segment["span"]
        if start > last:
            parts.append(content[last:start])
        last = end
    if last < len(content):
        parts.append(content[last:])
    return "".join(parts).strip()


def transform_response(data: dict[str, Any]) -> dict[str, Any]:
    for choice in data.get("choices", []):
        msg = choice.get("message", {})
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue

        if "<|tool_call" in content or "call:" in content:
            tool_calls = parse_tool_calls(content)
            if tool_calls:
                stripped_content = strip_tool_tokens(content)
                msg["tool_calls"] = tool_calls
                msg["content"] = stripped_content or None
                choice["finish_reason"] = "tool_calls"
            else:
                msg["content"] = strip_tool_tokens(content)
    return data


def build_sse_event(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")


def build_stream_chunks(data: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "id": data.get("id"),
        "object": "chat.completion.chunk",
        "created": data.get("created"),
        "model": data.get("model"),
    }
    if "system_fingerprint" in data:
        base["system_fingerprint"] = data["system_fingerprint"]

    chunks = []
    for choice in data.get("choices", []):
        index = choice.get("index", 0)
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")

        role_chunk = dict(base)
        role_chunk["choices"] = [
            {"index": index, "delta": {"role": message.get("role", "assistant")}, "finish_reason": None}
        ]
        chunks.append(role_chunk)

        if message.get("content"):
            content_chunk = dict(base)
            content_chunk["choices"] = [
                {"index": index, "delta": {"content": message["content"]}, "finish_reason": None}
            ]
            chunks.append(content_chunk)

        if message.get("tool_calls"):
            stream_tool_calls = []
            for tool_index, tool_call in enumerate(message["tool_calls"]):
                stream_tool_calls.append(
                    {
                        "index": tool_index,
                        "id": tool_call["id"],
                        "type": "function",
                        "function": tool_call["function"],
                    }
                )
            tool_chunk = dict(base)
            tool_chunk["choices"] = [
                {"index": index, "delta": {"tool_calls": stream_tool_calls}, "finish_reason": None}
            ]
            chunks.append(tool_chunk)

        final_chunk = dict(base)
        final_chunk["choices"] = [{"index": index, "delta": {}, "finish_reason": finish_reason or "stop"}]
        chunks.append(final_chunk)

    return chunks


async def collect_streamed_completion(resp: httpx.Response) -> dict[str, Any]:
    base: dict[str, Any] = {}
    choices: dict[int, dict[str, Any]] = {}
    buffered_bytes = 0

    async for line in resp.aiter_lines():
        if not line or not line.startswith("data:"):
            continue

        buffered_bytes += len(line.encode("utf-8"))
        if buffered_bytes > MAX_SSE_BUFFER_BYTES:
            raise HTTPException(status_code=502, detail="Upstream stream exceeded shim buffer limit")

        payload = line[5:].strip()
        if payload == "[DONE]":
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        for key in ("id", "model", "created", "system_fingerprint"):
            if key in chunk:
                base[key] = chunk[key]

        for choice in chunk.get("choices", []):
            index = choice.get("index", 0)
            entry = choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                },
            )

            delta = choice.get("delta", {})
            if delta.get("role"):
                entry["message"]["role"] = delta["role"]
            if delta.get("content"):
                entry["message"]["content"] += delta["content"]
            if choice.get("finish_reason") is not None:
                entry["finish_reason"] = choice["finish_reason"]

    return {
        **base,
        "object": "chat.completion",
        "choices": [choices[idx] for idx in sorted(choices)],
    }


def upstream_url_for_path(path: str) -> str:
    normalized = path.lstrip("/")
    return f"{UPSTREAM_URL}/{normalized}" if normalized else UPSTREAM_URL


async def proxy_error_response(resp: httpx.Response) -> Response:
    return Response(
        content=limited_bytes_preview(await resp.aread()),
        status_code=resp.status_code,
        headers=sanitize_headers(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


@app.exception_handler(httpx.HTTPError)
async def handle_upstream_http_error(_: Request, __: httpx.HTTPError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "Upstream request failed"})


@app.post("/v1/chat/completions")
async def proxy_completions(request: Request) -> Response:
    body = await read_request_json(request)
    headers = sanitize_headers(request.headers)

    if body.get("stream"):
        async with client.stream(
            "POST",
            upstream_url_for_path("/v1/chat/completions"),
            json=body,
            headers=headers,
            params=request.query_params,
        ) as resp:
            if resp.status_code != 200:
                return await proxy_error_response(resp)

            if not body.get("tools"):
                return StreamingResponse(
                    resp.aiter_raw(),
                    status_code=resp.status_code,
                    headers=sanitize_headers(resp.headers),
                    media_type=resp.headers.get("content-type"),
                )

            buffered = await collect_streamed_completion(resp)
            transformed = transform_response(buffered)
            chunks = build_stream_chunks(transformed)

            async def event_stream():
                for chunk in chunks:
                    yield build_sse_event(chunk)
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                status_code=resp.status_code,
                media_type="text/event-stream",
            )

    resp = await client.post(
        upstream_url_for_path("/v1/chat/completions"),
        json=body,
        headers=headers,
        params=request.query_params,
    )

    if resp.status_code != 200:
        return Response(
            content=limited_bytes_preview(resp.content),
            status_code=resp.status_code,
            headers=sanitize_headers(resp.headers),
            media_type=resp.headers.get("content-type"),
        )

    try:
        data = resp.json()
    except json.JSONDecodeError:
        return Response(
            content=limited_bytes_preview(resp.content),
            status_code=resp.status_code,
            headers=sanitize_headers(resp.headers),
            media_type=resp.headers.get("content-type"),
        )

    if body.get("tools"):
        data = transform_response(data)

    return JSONResponse(content=data)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(path: str, request: Request) -> Response:
    body = await read_request_body(request)
    resp = await client.request(
        method=request.method,
        url=upstream_url_for_path(path),
        content=body,
        headers=sanitize_headers(request.headers),
        params=request.query_params,
    )

    return Response(
        content=limited_bytes_preview(resp.content, max_bytes=MAX_REQUEST_BODY_BYTES),
        status_code=resp.status_code,
        headers=sanitize_headers(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


if __name__ == "__main__":
    print(f"Gemma 4 Tool Shim listening on {SHIM_HOST}:{SHIM_PORT} -> {UPSTREAM_URL}")
    uvicorn.run(app, host=SHIM_HOST, port=SHIM_PORT)
