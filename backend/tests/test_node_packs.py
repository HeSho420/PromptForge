"""Node-pack subsystem: curated list shape, probed status honesty, and the
new templates/models that ride on it — plus the LLM teaching surfaces
(every allowed node documented, every registry model with a usage line)."""
import tempfile
import unittest
from pathlib import Path

from app.adapters.comfyui import ALLOWED_NODE_TYPES
from app.core import node_packs
from app.core.services import DEFAULT_MODELS, MODEL_USAGE
from app.core.workflow_ai import NODE_GUIDE, NODE_OUTPUTS, SYSTEM_PROMPT


class LLMTeachingTests(unittest.TestCase):
    def test_every_allowed_node_is_documented_for_the_llm(self):
        """The generator LLM must never see a node it has no semantics for:
        every allowed node needs a NODE_GUIDE line AND a NODE_OUTPUTS entry,
        and both must appear in the actual system prompt."""
        self.assertEqual(set(NODE_GUIDE), ALLOWED_NODE_TYPES)
        self.assertEqual(set(NODE_OUTPUTS), ALLOWED_NODE_TYPES)
        for node, desc in NODE_GUIDE.items():
            self.assertTrue(desc.strip(), node)
            self.assertIn(node, SYSTEM_PROMPT)
        # Spot-check that semantics (not just names) reach the prompt.
        self.assertIn("LEGACY inpaint encode", SYSTEM_PROMPT)
        self.assertIn("regional", SYSTEM_PROMPT)

    def test_every_registry_model_has_a_usage_line(self):
        """Model choice must never be a guess: every registered model needs
        a when-to-use line, and pairing rules must be explicit."""
        for m in DEFAULT_MODELS:
            self.assertIn(m.name, MODEL_USAGE, m.name)
        self.assertIn("SDXL checkpoints ONLY", MODEL_USAGE["dmd2-sdxl-lora"])
        self.assertIn("SD15 checkpoints ONLY", MODEL_USAGE["lcm-lora-sd15"])
        self.assertIn("readable text", MODEL_USAGE["zimage-turbo"])


class KnownPacksTests(unittest.TestCase):
    def test_curated_packs_are_complete_and_pinned(self):
        expected = {"impact-pack", "controlnet-aux", "frame-interpolation",
                    "rmbg", "ic-light", "instantid", "gguf"}
        self.assertEqual(set(node_packs.KNOWN_PACKS), expected)
        for pack in node_packs.KNOWN_PACKS.values():
            self.assertRegex(pack.repo, r"^[\w.-]+/[\w.-]+$")
            self.assertTrue(pack.verify_node)
            self.assertTrue(pack.unlocks)
            for url in node_packs.zip_urls(pack.repo):
                self.assertTrue(url.startswith("https://github.com/"))

    def test_status_is_probed_never_assumed(self):
        pack = node_packs.KNOWN_PACKS["gguf"]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # absent: nothing on disk
            self.assertEqual(
                node_packs.pack_status(pack, base, None)["status"], "absent")
            (base / "custom_nodes" / pack.dir_name).mkdir(parents=True)
            # on disk, ComfyUI down -> installed (we can't verify more)
            self.assertEqual(
                node_packs.pack_status(pack, base, None)["status"], "installed")
            # on disk + node live -> active
            self.assertEqual(
                node_packs.pack_status(
                    pack, base, {"UnetLoaderGGUF"})["status"], "active")
            # on disk but ComfyUI does NOT expose it -> broken, not "works"
            self.assertEqual(
                node_packs.pack_status(pack, base, {"KSampler"})["status"],
                "broken")
        self.assertEqual(
            node_packs.pack_status(pack, None, None)["status"], "absent")


class NewModelsTests(unittest.TestCase):
    def test_new_registry_entries_have_checksums_and_folders(self):
        by_name = {m.name: m for m in DEFAULT_MODELS}
        for name in ("dmd2-sdxl-lora", "lcm-lora-sd15",
                     "controlnet-union-sdxl", "iclight-sd15-fc",
                     "zimage-turbo", "zimage-text-encoder", "flux-ae",
                     "flux-kontext-q4", "flux-t5-fp8", "flux-clip-l",
                     "qwen-image-edit-2511"):
            m = by_name[name]
            self.assertRegex(m.sha256 or "", r"^[0-9a-f]{64}$", name)
            self.assertIn("folder", m.meta or {}, name)

    def test_big_models_declare_min_vram(self):
        by_name = {m.name: m for m in DEFAULT_MODELS}
        qwen = by_name["qwen-image-edit-2511"]
        self.assertGreaterEqual((qwen.meta or {}).get("min_vram_gb", 0), 16)

    def test_pack_gated_models_name_a_curated_pack(self):
        for m in DEFAULT_MODELS:
            pack = (m.meta or {}).get("requires_pack")
            if pack:
                self.assertIn(pack, node_packs.KNOWN_PACKS, m.name)


class ComfyPythonResolutionTests(unittest.TestCase):
    """The pip-into-the-wrong-venv regression: pack requirements must target
    ComfyUI's OWN interpreter, never the backend venv."""

    def _services(self):
        """A Services + its data dir; cleanup closes the DB (services.stop)
        BEFORE removing the folder — LIFO order avoids a WinError 32 on the
        still-open SQLite file under Windows."""
        from app.config import Settings
        from app.core.services import Services
        tmp = tempfile.TemporaryDirectory()
        s = Services(Settings(
            data_dir=Path(tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(tmp.cleanup)   # runs LAST
        self.addCleanup(s.stop)        # runs FIRST — closes the DB
        return s, Path(tmp.name)

    def test_prefers_comfy_venv_and_never_backend(self):
        s, data = self._services()
        base = data / "ComfyUI"
        (base / ".venv" / "Scripts").mkdir(parents=True)
        venv_py = base / ".venv" / "Scripts" / "python.exe"
        venv_py.write_text("")
        self.assertEqual(s._comfy_python(base), str(venv_py))

    def test_portable_python_embeded_layout(self):
        s, data = self._services()
        base = data / "ComfyUI"
        base.mkdir()
        emb = base.parent / "python_embeded"
        emb.mkdir()
        (emb / "python.exe").write_text("")
        self.assertEqual(s._comfy_python(base), str(emb / "python.exe"))

    def test_no_env_fails_honestly_never_backend_fallback(self):
        s, data = self._services()
        base = data / "ComfyUI"
        base.mkdir()
        with self.assertRaises(Exception) as ctx:
            s._comfy_python(base)
        self.assertIn("ComfyUI's Python", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
