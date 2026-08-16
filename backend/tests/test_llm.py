"""LLM layer tests — all offline: transports and API clients are injected."""
import unittest
import urllib.error
from types import SimpleNamespace

from app.core.llm import (
    ClaudeLLM,
    FallbackLLM,
    LLMRefusedError,
    LLMReply,
    LLMUnavailableError,
    LocalLLM,
)


class FakeBackend:
    def __init__(self, source, reply=None, error=None):
        self.source = source
        self._reply = reply
        self._error = error
        self.calls = 0

    def complete(self, system, prompt, max_tokens=4096):
        self.calls += 1
        if self._error:
            raise self._error
        return self._reply


class LocalLLMTests(unittest.TestCase):
    def test_prefers_native_ollama_with_num_ctx(self):
        def post(url, payload, timeout):
            self.assertTrue(url.endswith("/api/chat"))
            self.assertEqual(payload["options"]["num_ctx"], LocalLLM.NUM_CTX)
            self.assertEqual(payload["format"], "json")
            self.assertEqual(payload["messages"][0]["role"], "system")
            return {"model": "test-model:latest",
                    "message": {"content": "hello"}}

        llm = LocalLLM("http://localhost:11434/v1", "test-model", http_post=post)
        reply = llm.complete("sys", "hi")
        self.assertEqual(reply.text, "hello")
        self.assertEqual(reply.model, "test-model:latest")
        self.assertEqual(reply.source, "local")

    def test_parses_openai_shape_when_native_missing(self):
        import io as _io

        def post(url, payload, timeout):
            if url.endswith("/api/chat"):  # not an Ollama server
                raise urllib.error.HTTPError(
                    url, 404, "not found", {},  # type: ignore[arg-type]
                    _io.BytesIO(b"404 page not found"))
            self.assertTrue(url.endswith("/chat/completions"))
            self.assertEqual(payload["model"], "test-model")
            self.assertEqual(payload["messages"][0]["role"], "system")
            return {"model": "test-model:latest",
                    "choices": [{"message": {"content": "hello"}}]}

        llm = LocalLLM("http://localhost:11434/v1", "test-model", http_post=post)
        reply = llm.complete("sys", "hi")
        self.assertEqual(reply.text, "hello")
        self.assertEqual(reply.model, "test-model:latest")
        self.assertEqual(reply.source, "local")

    def test_native_missing_model_hints_ollama_pull(self):
        import io as _io

        def post(url, payload, timeout):
            raise urllib.error.HTTPError(
                url, 404, "not found", {},  # type: ignore[arg-type]
                _io.BytesIO(b'{"error":"model \'qwen2.5:7b\' not found"}'))

        llm = LocalLLM("http://localhost:11434/v1", "qwen2.5:7b", http_post=post)
        with self.assertRaises(LLMUnavailableError) as ctx:
            llm.complete("sys", "hi")
        self.assertIn("ollama pull qwen2.5:7b", str(ctx.exception))

    def test_unreachable_maps_to_unavailable(self):
        def post(url, payload, timeout):
            raise urllib.error.URLError("refused")

        llm = LocalLLM("http://localhost:11434/v1", "m", http_post=post)
        with self.assertRaises(LLMUnavailableError) as ctx:
            llm.complete("sys", "hi")
        self.assertIn("unreachable", str(ctx.exception))

    def test_missing_model_self_heals_or_hints_ollama_pull(self):
        """A missing model now auto-starts `ollama pull` in the background;
        the error either says the download started (Ollama installed) or
        still carries the manual hint (no Ollama on this box). Both name
        the model — the user always learns what is missing."""
        def post(url, payload, timeout):
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)  # type: ignore[arg-type]

        llm = LocalLLM("http://localhost:11434/v1", "qwen2.5:7b", http_post=post)
        with self.assertRaises(LLMUnavailableError) as ctx:
            llm.complete("sys", "hi")
        msg = str(ctx.exception)
        self.assertIn("qwen2.5:7b", msg)
        self.assertTrue("downloading it in the background" in msg
                        or "ollama pull qwen2.5:7b" in msg, msg)


class ClaudeLLMTests(unittest.TestCase):
    def _client(self, response):
        captured = {}

        class Messages:
            def create(self, **kw):
                captured.update(kw)
                return response

        client = SimpleNamespace(beta=SimpleNamespace(messages=Messages()))
        return client, captured

    def test_request_shape_and_provenance(self):
        response = SimpleNamespace(
            stop_reason="end_turn", model="claude-opus-4-8",  # fallback served it
            content=[SimpleNamespace(type="text", text="{}")])
        client, captured = self._client(response)
        llm = ClaudeLLM(client_factory=lambda: client)
        reply = llm.complete("sys", "make a workflow")

        self.assertEqual(captured["model"], "claude-fable-5")
        self.assertNotIn("thinking", captured)  # Fable 5: must be omitted
        self.assertEqual(captured["betas"], ["server-side-fallback-2026-06-01"])
        self.assertEqual(captured["fallbacks"], [{"model": "claude-opus-4-8"}])
        # provenance reports the model that actually served the reply
        self.assertEqual(reply.model, "claude-opus-4-8")
        self.assertEqual(reply.source, "api")

    def test_refusal_raises(self):
        response = SimpleNamespace(stop_reason="refusal", model="claude-fable-5",
                                   content=[])
        client, _ = self._client(response)
        llm = ClaudeLLM(client_factory=lambda: client)
        with self.assertRaises(LLMRefusedError):
            llm.complete("sys", "x")

    def test_api_error_maps_to_unavailable(self):
        class Messages:
            def create(self, **kw):
                raise ConnectionError("boom")

        client = SimpleNamespace(beta=SimpleNamespace(messages=Messages()))
        llm = ClaudeLLM(client_factory=lambda: client)
        with self.assertRaises(LLMUnavailableError):
            llm.complete("sys", "x")


class FallbackLLMTests(unittest.TestCase):
    def test_primary_wins_when_healthy(self):
        primary = FakeBackend("local", reply=LLMReply("a", "m1", "local"))
        fallback = FakeBackend("api", reply=LLMReply("b", "m2", "api"))
        reply = FallbackLLM(primary, fallback).complete("s", "p")
        self.assertEqual(reply.source, "local")
        self.assertEqual(fallback.calls, 0)

    def test_falls_back_when_local_unavailable(self):
        primary = FakeBackend("local", error=LLMUnavailableError("down"))
        fallback = FakeBackend("api", reply=LLMReply("b", "m2", "api"))
        reply = FallbackLLM(primary, fallback).complete("s", "p")
        self.assertEqual(reply.source, "api")

    def test_reports_all_failures(self):
        primary = FakeBackend("local", error=LLMUnavailableError("ollama down"))
        fallback = FakeBackend("api", error=LLMUnavailableError("no api key"))
        with self.assertRaises(LLMUnavailableError) as ctx:
            FallbackLLM(primary, fallback).complete("s", "p")
        self.assertIn("ollama down", str(ctx.exception))
        self.assertIn("no api key", str(ctx.exception))

    def test_refusal_is_not_retried_elsewhere(self):
        primary = FakeBackend("local", error=LLMRefusedError("nope"))
        fallback = FakeBackend("api", reply=LLMReply("b", "m2", "api"))
        with self.assertRaises(LLMRefusedError):
            FallbackLLM(primary, fallback).complete("s", "p")
        self.assertEqual(fallback.calls, 0)

    def test_no_backends_configured(self):
        with self.assertRaises(LLMUnavailableError):
            FallbackLLM(None, None).complete("s", "p")


if __name__ == "__main__":
    unittest.main()
