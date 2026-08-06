"""Model search tests — offline, Hugging Face API responses are injected."""
import tempfile
import unittest
from pathlib import Path

from app.core.db import Database
from app.core.model_search import ModelSearch, ModelSearchError
from app.core.registry import ModelRegistry

SEARCH_RESPONSE = [
    {"id": "author/great-model", "downloads": 5000, "likes": 42,
     "pipeline_tag": "text-to-image", "gated": False},
    {"id": "author/other", "downloads": 10, "likes": 0,
     "pipeline_tag": None, "gated": True},
    {"unexpected": "shape"},  # tolerated, skipped
]

TREE_RESPONSE = [
    {"type": "file", "path": "model.safetensors", "size": 4_000_000_000,
     "lfs": {"oid": "a" * 64, "size": 4_000_000_000}},
    {"type": "file", "path": "small.pt", "size": 1000},  # non-LFS: no sha256
    {"type": "file", "path": "README.md", "size": 500},  # not a weight file
    {"type": "directory", "path": "assets"},
]


class FakeHttp:
    def __init__(self, routes):
        self.routes = routes
        self.urls = []

    def __call__(self, url, timeout):
        self.urls.append(url)
        for fragment, data in self.routes.items():
            if fragment in url:
                if isinstance(data, Exception):
                    raise data
                return data
        raise AssertionError(f"unexpected url: {url}")


class ModelSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        db = Database(Path(self.tmp.name) / "t.sqlite3")
        self.registry = ModelRegistry(db, Path(self.tmp.name) / "models")

    def tearDown(self):
        self.tmp.cleanup()

    def _search(self, routes):
        return ModelSearch(self.registry, http_get=FakeHttp(routes))

    def test_search_parses_candidates(self):
        ms = self._search({"/api/models?": SEARCH_RESPONSE})
        hits = ms.search("great model")
        self.assertEqual(len(hits), 2)  # malformed entry skipped
        self.assertEqual(hits[0].repo_id, "author/great-model")
        self.assertEqual(hits[0].downloads, 5000)
        self.assertTrue(hits[1].gated)

    def test_weight_files_extract_lfs_sha256(self):
        ms = self._search({"/tree/main": TREE_RESPONSE})
        files = ms.list_weight_files("author/great-model")
        self.assertEqual([f.filename for f in files],
                         ["model.safetensors", "small.pt"])  # sorted by size
        self.assertEqual(files[0].sha256, "a" * 64)
        self.assertIsNone(files[1].sha256)  # non-LFS: no published checksum

    def test_propose_registers_with_checksum_and_constructed_url(self):
        ms = self._search({"/tree/main": TREE_RESPONSE})
        model = ms.propose("author/great-model", "model.safetensors",
                           name="great-model", purpose="testing", vram_gb=6.0)
        self.assertEqual(model.sha256, "a" * 64)
        self.assertEqual(
            model.url,
            "https://huggingface.co/author/great-model/resolve/main/model.safetensors")
        # really registered, downloadable via the existing verified path
        stored = self.registry.get("great-model")
        self.assertIsNotNone(stored)
        self.assertEqual(stored.status, "not_downloaded")
        self.assertEqual(stored.sha256, "a" * 64)

    def test_propose_without_checksum_flags_the_gap(self):
        ms = self._search({"/tree/main": TREE_RESPONSE})
        model = ms.propose("author/great-model", "small.pt",
                           name="small", purpose="testing")
        self.assertIsNone(model.sha256)
        self.assertIn("cannot be verified", model.license)
        self.assertIn("Pickle-based", model.license)

    def test_propose_duplicate_name_rejected(self):
        ms = self._search({"/tree/main": TREE_RESPONSE})
        ms.propose("author/great-model", "model.safetensors",
                   name="dup", purpose="testing")
        with self.assertRaises(ModelSearchError):
            ms.propose("author/great-model", "model.safetensors",
                       name="dup", purpose="testing")

    def test_propose_unknown_file_rejected(self):
        ms = self._search({"/tree/main": TREE_RESPONSE})
        with self.assertRaises(ModelSearchError):
            ms.propose("author/great-model", "README.md",
                       name="readme", purpose="not a model")

    def test_network_failure_maps_to_search_error(self):
        ms = self._search({"/api/models?": ModelSearchError("HF unreachable")})
        with self.assertRaises(ModelSearchError):
            ms.search("anything")


if __name__ == "__main__":
    unittest.main()
