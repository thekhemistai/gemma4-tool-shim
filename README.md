# Gemma 4 Tool Shim

`gemma4-tool-shim.py` is a small FastAPI proxy that sits in front of a local
OpenAI-compatible inference server and rewrites Gemma-style raw tool call text
into proper OpenAI `tool_calls[]` responses.

It is intended for local model stacks that emit content such as:

```text
<|tool_call>call:get_weather {"city":"Phoenix"}<tool_call|>
```

The shim forwards requests to your upstream server, preserves normal responses,
and only transforms chat completions when tool parsing is needed.

## Features

- Parses raw `call:<tool_name> {...}` tool blocks
- Repairs a limited class of malformed JSON keys
- Falls back to Python-literal parsing for dict-like argument payloads
- Converts buffered streamed completions into valid OpenAI-style SSE chunks
- Provides a generic passthrough proxy for other upstream endpoints
- Uses environment variables instead of hardcoded host and port values

## Installation

```bash
pip install fastapi uvicorn httpx
```

Optional:

```bash
cp .env.example .env
```

## Usage

Run directly with Python:

```bash
python gemma4-tool-shim.py
```

Run with Uvicorn:

```bash
uvicorn gemma4-tool-shim:app --host 127.0.0.1 --port 8083
```

Point it at a different upstream:

```bash
UPSTREAM_URL=http://127.0.0.1:8080 SHIM_HOST=127.0.0.1 SHIM_PORT=8083 python gemma4-tool-shim.py
```

Example request:

```bash
curl http://127.0.0.1:8083/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemma",
    "messages": [{"role": "user", "content": "Call the weather tool"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "parameters": {"type": "object","properties":{"city":{"type":"string"}}}
      }
    }]
  }'
```

## Environment Variables

- `UPSTREAM_URL`
  Default: `http://localhost:8080`
  Base URL for the upstream OpenAI-compatible server. Use only a base URL, not a path.

- `SHIM_HOST`
  Default: `127.0.0.1`
  Host interface for the shim listener.

- `SHIM_PORT`
  Default: `8083`
  Port for the shim listener.

## Compatibility Notes

- Designed for local or otherwise trusted upstream OpenAI-compatible servers.
- Stream transformation buffers the full completion before emitting tool-call SSE chunks.
- The parser handles quoted JSON objects, a limited malformed-key repair path, and Python dict-like literals.
- If your upstream already returns proper OpenAI `tool_calls`, the shim should be unnecessary.
- This project keeps the original balanced-brace parser and tool token stripping behavior.

## Security Notes

- The public version adds request and stream size limits.
- The upstream URL is validated as an absolute `http(s)` base URL.
- Hop-by-hop headers are stripped before forwarding.
- There is no built-in authentication or rate limiting; keep it behind localhost or add a reverse proxy if exposing it.
