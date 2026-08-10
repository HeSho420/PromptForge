import hashlib
import io
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from app.core.db import Database
from app.core.registry import DownloadError, ModelDownloader, ModelInfo, ModelRegistry


class _FakeResp:
    """Minimal stand-in for an http response context manager."""

    def __init__(self, data: bytes, status: int = 200,
                 headers: dict | None = None):
        self._buf = io.BytesIO(data)
        self.status = status
        self.headers = headers or {"Content-Length": str(len(data))}

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db = Database(root / "test.sqlite3")
        self.registry = ModelRegistry(self.db, root / "models")
        self.downloader = ModelDownloader(self.registry)
        # a local "remote" file served over file:// so download logic runs for real
        self.source = root / "weights.bin"
        self.payload = b"pretend-model-weights" * 1000
        self.source.write_bytes(self.payload)
        self.sha = hashlib.sha256(self.payload).hexdigest()

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _register(self, sha: str | None):
        self.registry.register(ModelInfo(
            name="test-model", purpose="unit test", license="MIT",
            url=self.source.as_uri(), sha256=sha, vram_gb=1.0))

    def test_register_and_list(self):
        self._register(self.sha)
        models = self.registry.list()
        self.assertEqual(len(models), 1)
        m = models[0]
        self.assertEqual((m.name, m.license, m.status, m.vram_gb),
                         ("test-model", "MIT", "not_downloaded", 1.0))

    def test_download_with_valid_checksum(self):
        self._register(self.sha)
        progress: list[int] = []
        model = self.downloader.download("test-model", lambda done, total: progress.append(done))
        self.assertEqual(model.status, "ready")
        self.assertTrue(Path(model.path).exists())
        self.assertEqual(Path(model.path).read_bytes(), self.payload)
        self.assertTrue(progress and progress[-1] == len(self.payload))
        self.assertTrue(self.registry.is_ready("test-model"))

    def test_meta_file_names_the_saved_file(self):
        """Civitai URLs end in a bare version id — the file must be saved
        under meta.file's real name or ComfyUI's extension-filtered loaders
        never list it (live bug: checkpoints saved as '456538')."""
        # A civitai-style source: URL basename has no extension.
        versioned = self.source.parent / "456538"
        versioned.write_bytes(self.payload)
        self.registry.register(ModelInfo(
            name="civitai-model", purpose="unit test", license="x",
            url=versioned.as_uri(), sha256=self.sha,
            meta={"folder": "checkpoints",
                  "file": "properName_v10.safetensors"}))
        model = self.downloader.download("civitai-model")
        self.assertEqual(Path(model.path).name, "properName_v10.safetensors")
        self.assertTrue(Path(model.path).exists())

    def test_checksum_mismatch_discards_file(self):
        self._register("0" * 64)
        with self.assertRaises(DownloadError) as ctx:
            self.downloader.download("test-model")
        self.assertIn("Checksum mismatch", str(ctx.exception))
        self.assertEqual(self.registry.get("test-model").status, "checksum_failed")
        # no stray files left behind (models live in typed subfolders)
        leftovers = [p for p in self.registry.models_dir.rglob("*") if p.is_file()]
        self.assertEqual(leftovers, [])
        self.assertFalse(self.registry.is_ready("test-model"))

    def test_download_without_url_fails_cleanly(self):
        self.registry.register(ModelInfo(name="no-url", purpose="x"))
        with self.assertRaises(DownloadError) as ctx:
            self.downloader.download("no-url")
        self.assertIn("no download URL", str(ctx.exception))

    def test_unknown_model_fails_cleanly(self):
        with self.assertRaises(DownloadError):
            self.downloader.download("ghost")

    def test_unreachable_source_marks_failed(self):
        self.registry.register(ModelInfo(
            name="gone", purpose="x", url=(Path(self.tmp.name) / "missing.bin").as_uri()))
        with self.assertRaises(DownloadError):
            self.downloader.download("gone")
        self.assertEqual(self.registry.get("gone").status, "failed")

    def test_redownload_of_ready_model_is_noop(self):
        self._register(self.sha)
        first = self.downloader.download("test-model")
        second = self.downloader.download("test-model")
        self.assertEqual(first.path, second.path)

    def test_reset_stale_reenables_stuck_download(self):
        self._register(self.sha)
        self.registry.set_status("test-model", "downloading")
        reset = self.registry.reset_stale()
        self.assertEqual(reset, ["test-model"])
        self.assertEqual(self.registry.get("test-model").status, "not_downloaded")


class ResilientDownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.db = Database(root / "t.sqlite3")
        self.registry = ModelRegistry(self.db, root / "models")
        self.payload = b"abcdefgh" * 4096
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.registry.register(ModelInfo(
            name="m", purpose="x", license="MIT",
            url="https://huggingface.co/x/resolve/main/w.safetensors", sha256=self.sha))
        self.slept: list[float] = []
        self.downloader = ModelDownloader(self.registry, sleep=self.slept.append)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_transient_failure_is_retried_then_succeeds(self):
        calls = {"n": 0}

        def fake(req, timeout):
            calls["n"] += 1
            if calls["n"] == 1:
                raise urllib.error.URLError("connection reset")
            return _FakeResp(self.payload)

        with mock.patch("app.core.registry.urlopen_verified", fake):
            m = self.downloader.download("m")
        self.assertEqual(m.status, "ready")
        self.assertEqual(Path(m.path).read_bytes(), self.payload)
        self.assertEqual(calls["n"], 2)
        self.assertTrue(self.slept)  # backed off before the retry

    def test_partial_download_resumes_via_range(self):
        half = len(self.payload) // 2
        part = self.registry.models_dir / "checkpoints" / "w.safetensors.part"
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_bytes(self.payload[:half])  # leftover from an interrupted run
        seen = {}

        def fake(req, timeout):
            seen["range"] = req.headers.get("Range")
            start = int(seen["range"].split("=")[1].split("-")[0])
            rest = self.payload[start:]
            return _FakeResp(rest, status=206, headers={
                "Content-Length": str(len(rest)),
                "Content-Range": f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}",
            })

        with mock.patch("app.core.registry.urlopen_verified", fake):
            m = self.downloader.download("m")
        self.assertEqual(seen["range"], f"bytes={half}-")
        self.assertEqual(Path(m.path).read_bytes(), self.payload)
        self.assertEqual(m.status, "ready")

    def test_permanent_http_error_not_retried(self):
        def fake(req, timeout):
            raise urllib.error.HTTPError(
                "https://x/w.bin", 404, "Not Found", {}, None)

        with mock.patch("app.core.registry.urlopen_verified", fake):
            with self.assertRaises(DownloadError):
                self.downloader.download("m")
        self.assertEqual(self.registry.get("m").status, "failed")
        self.assertEqual(self.slept, [])  # 404 is terminal — no backoff


if __name__ == "__main__":
    unittest.main()
