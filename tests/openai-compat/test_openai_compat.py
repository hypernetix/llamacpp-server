"""
OpenAI-Compatible API integration test.

Tests the /v1/ endpoints using the official OpenAI Python SDK,
proving drop-in compatibility with existing OpenAI client libraries.

Usage (standalone):
    SERVER_URL=http://localhost:8080 MODEL_PATH=/models/model.gguf python test_openai_compat.py

Usage (docker-compose):
    docker compose -f docker/docker-compose.openai-test.yml up --build --abort-on-container-exit
"""

import os
import sys
import time

import requests
from openai import OpenAI

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8080")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/model.gguf")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "50"))


def wait_for_server(url: str, timeout: int = 120):
    """Wait for the server health endpoint to respond."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                print(f"[OK] Server is healthy at {url}")
                return
        except requests.ConnectionError:
            pass
        time.sleep(2)
    print(f"[FAIL] Server not reachable at {url} after {timeout}s")
    sys.exit(1)


def load_model(url: str, model_path: str):
    """Load a model via the custom /models/load SSE endpoint."""
    print(f"\n--- Loading model: {model_path} ---")
    r = requests.post(
        f"{url}/models/load",
        json={"path": model_path},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=300,
    )
    r.raise_for_status()
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                print("[OK] Model loaded successfully")
                return
            print(f"  progress: {payload}")
    print("[FAIL] Model load stream ended without [DONE]")
    sys.exit(1)


def test_list_models(client: OpenAI, model_id: str):
    """GET /v1/models — verify the loaded model appears in the list."""
    print("\n--- Test: GET /v1/models ---")
    models = client.models.list()
    ids = [m.id for m in models.data]
    print(f"  Models: {ids}")
    assert model_id in ids, f"Expected {model_id} in model list, got {ids}"
    assert models.data[0].owned_by == "local"
    print("[OK] /v1/models")


def test_completions(client: OpenAI, model_id: str):
    """POST /v1/completions — non-streaming text completion."""
    print("\n--- Test: POST /v1/completions (non-streaming) ---")
    resp = client.completions.create(
        model=model_id,
        prompt="The capital of France is",
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    text = resp.choices[0].text
    print(f"  ID:            {resp.id}")
    print(f"  Model:         {resp.model}")
    print(f"  Text:          {text!r}")
    print(f"  Finish reason: {resp.choices[0].finish_reason}")
    assert resp.id.startswith("cmpl-"), f"Bad ID prefix: {resp.id}"
    assert resp.object == "text_completion"
    assert resp.model == model_id
    assert len(text) > 0, "Empty completion text"
    assert resp.choices[0].finish_reason in ("stop", "length")
    print("[OK] /v1/completions (non-streaming)")


def test_completions_streaming(client: OpenAI, model_id: str):
    """POST /v1/completions — streaming text completion."""
    print("\n--- Test: POST /v1/completions (streaming) ---")
    stream = client.completions.create(
        model=model_id,
        prompt="The capital of France is",
        max_tokens=MAX_TOKENS,
        temperature=0,
        stream=True,
    )
    chunks = list(stream)
    texts = [c.choices[0].text for c in chunks if c.choices[0].text]
    full_text = "".join(texts)
    print(f"  Chunks:    {len(chunks)}")
    print(f"  Full text: {full_text!r}")
    assert len(chunks) >= 2, f"Expected at least 2 chunks (content + finish), got {len(chunks)}"
    assert chunks[0].id.startswith("cmpl-")
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert len(full_text) > 0, "Empty streaming text"
    print("[OK] /v1/completions (streaming)")


def test_chat_completions(client: OpenAI, model_id: str):
    """POST /v1/chat/completions — non-streaming chat completion."""
    print("\n--- Test: POST /v1/chat/completions (non-streaming) ---")
    resp = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        max_tokens=MAX_TOKENS,
        temperature=0,
    )
    content = resp.choices[0].message.content
    print(f"  ID:            {resp.id}")
    print(f"  Model:         {resp.model}")
    print(f"  Role:          {resp.choices[0].message.role}")
    print(f"  Content:       {content!r}")
    print(f"  Finish reason: {resp.choices[0].finish_reason}")
    assert resp.id.startswith("chatcmpl-"), f"Bad ID prefix: {resp.id}"
    assert resp.object == "chat.completion"
    assert resp.model == model_id
    assert resp.choices[0].message.role == "assistant"
    assert len(content) > 0, "Empty chat content"
    assert resp.choices[0].finish_reason in ("stop", "length")
    print("[OK] /v1/chat/completions (non-streaming)")


def test_chat_completions_streaming(client: OpenAI, model_id: str):
    """POST /v1/chat/completions — streaming chat completion."""
    print("\n--- Test: POST /v1/chat/completions (streaming) ---")
    stream = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        max_tokens=MAX_TOKENS,
        temperature=0,
        stream=True,
    )
    chunks = list(stream)
    print(f"  Chunks: {len(chunks)}")

    # First chunk should announce the assistant role
    first = chunks[0]
    assert first.choices[0].delta.role == "assistant", \
        f"First chunk should have role='assistant', got {first.choices[0].delta}"
    print(f"  First chunk role: {first.choices[0].delta.role}")

    # Middle chunks should have content
    content_parts = []
    for c in chunks[1:]:
        delta = c.choices[0].delta
        if delta and delta.content:
            content_parts.append(delta.content)

    full_content = "".join(content_parts)
    print(f"  Full content: {full_content!r}")
    assert len(full_content) > 0, "Empty streaming content"

    # Last chunk should have finish_reason
    last = chunks[-1]
    assert last.choices[0].finish_reason == "stop", \
        f"Last chunk should have finish_reason='stop', got {last.choices[0].finish_reason}"
    print(f"  Finish reason: {last.choices[0].finish_reason}")

    # All chunks share the same ID
    ids = set(c.id for c in chunks)
    assert len(ids) == 1, f"All chunks should share one ID, got {ids}"
    assert chunks[0].id.startswith("chatcmpl-")
    print("[OK] /v1/chat/completions (streaming)")


def test_error_handling(client: OpenAI):
    """Verify that invalid requests return proper OpenAI-format errors."""
    print("\n--- Test: Error handling ---")
    try:
        client.chat.completions.create(
            model="nonexistent-model",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=10,
        )
        print("[FAIL] Expected an error for nonexistent model")
        sys.exit(1)
    except Exception as e:
        print(f"  Got expected error: {type(e).__name__}: {e}")
        print("[OK] Error handling")


def main():
    print("=" * 60)
    print("  OpenAI-Compatible API Integration Test")
    print("=" * 60)
    print(f"  Server:     {SERVER_URL}")
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Max tokens: {MAX_TOKENS}")

    wait_for_server(SERVER_URL)
    load_model(SERVER_URL, MODEL_PATH)

    client = OpenAI(
        base_url=f"{SERVER_URL}/v1",
        api_key="not-needed",
    )

    passed = 0
    failed = 0
    tests = [
        ("list_models",                lambda: test_list_models(client, MODEL_PATH)),
        ("completions",                lambda: test_completions(client, MODEL_PATH)),
        ("completions_streaming",      lambda: test_completions_streaming(client, MODEL_PATH)),
        ("chat_completions",           lambda: test_chat_completions(client, MODEL_PATH)),
        ("chat_completions_streaming", lambda: test_chat_completions_streaming(client, MODEL_PATH)),
        ("error_handling",             lambda: test_error_handling(client)),
    ]

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
