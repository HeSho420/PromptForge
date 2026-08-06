"""Image → 3D, and picking the best rung this machine can actually run.

Two promises here. First, that the mesh pipeline uses ComfyUI's built-in
Hunyuan3D nodes — no node pack, nothing compiled, nothing that can break the
working install. Second, that the tier chosen is the best the HARDWARE
supports, so the same code produces a better avatar on a bigger GPU without
any configuration, and says plainly what it could not do here.
"""
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from app.adapters.comfyui import ALLOWED_NODE_TYPES, ALLOWED_TASKS, WorkflowLibrary, build_workflow
from app.config import Settings
from app.core import quality
from app.core.jobs import Job
from app.core.services import DEFAULT_MODELS, Services

WORKFLOWS = Path(__file__).parent.parent / "app" / "workflows"


class OneJsonPlan:
    """An LLM that always returns the same plan, so the deterministic
    coercions can be tested without a model."""

    source = "local"

    def __init__(self, payload):
        self.payload = payload

    def complete(self, system, prompt, max_tokens=4096):
        import json

        from app.core.llm import LLMReply
        return LLMReply(json.dumps(self.payload), "fake", "local")


class TierTests(unittest.TestCase):
    """The ladder, and the honesty about where it stops.

    What a rung buys is SURFACE DETAIL. It used to buy more VIEWS, on the
    assumption that four beat one; measured, conditioning on views another
    model invented made the figure 40% too deep (depth/width 0.73 against
    0.53 from the single photograph, where a standing human is about 0.5).
    These tests encode the measurement, not the assumption."""

    def test_this_machine_gets_the_middle_rung(self):
        """RTX 4060 Laptop: 8 GB VRAM, 16 GB RAM."""
        tier = quality.choose_reconstruction(8, 16)
        self.assertEqual(tier.level, 2)
        self.assertEqual(tier.octree, 256)

    def test_hardware_buys_octree_resolution_not_view_count(self):
        octrees = [quality.choose_reconstruction(v, r).octree
                   for v, r in ((6, 12), (8, 16), (12, 24), (16, 32))]
        self.assertEqual(octrees, [192, 256, 384, 512])

    def test_no_rung_claims_a_model_painted_texture(self):
        """ComfyUI ships no Hunyuan3D paint stage and the official one needs a
        CUDA rasterizer with no wheel for this Python. Colour still happens —
        it is projected from the photographs — but no MODEL paints it."""
        for tier in quality.RECON_TIERS:
            self.assertFalse(tier.paint_model, tier.name)

    def test_a_small_gpu_still_gets_a_mesh(self):
        tier = quality.choose_reconstruction(6, 12)
        self.assertEqual(tier.level, 1)
        self.assertEqual(tier.models, ("hunyuan3d-v2",))

    def test_no_gpu_means_no_mesh_and_says_so(self):
        tier = quality.choose_reconstruction(0, 8)
        self.assertEqual(tier.level, 0)
        self.assertEqual(tier.models, ())
        self.assertIn("no mesh", tier.what)

    def test_ram_gates_as_hard_as_vram(self):
        """RAM, not VRAM, is what OS-killed the heavy loads on this box —
        octree 512 died asking the CPU allocator for 4.3 GB."""
        self.assertEqual(quality.choose_reconstruction(8, 8).level, 0)

    def test_the_ladder_is_ordered_best_first(self):
        levels = [t.level for t in quality.RECON_TIERS]
        self.assertEqual(levels, sorted(levels, reverse=True))

    def test_multi_view_is_off_for_people_however_many_photos(self):
        """Hunyuan3D's multi-view model wants four renders of one RIGID
        object. Two photographs of a person differ in pose, framing, clothing
        and light; measured on a real 9-photo dataset, two real angles gave
        depth/width 1.19 where the single best photo gave 0.68."""
        for angles in (0, 1, 2, 4, 8):
            self.assertFalse(quality.use_multiview(angles))
        self.assertTrue(quality.use_multiview(2, rigid_subject=True))
        self.assertFalse(quality.use_multiview(1, rigid_subject=True))

    def test_the_turbo_checkpoint_is_driven_at_turbo_settings(self):
        """It is a distillation. At the undistilled 20 steps / cfg 4.0 it was
        both slower and worse (0.73 against 0.63)."""
        self.assertLessEqual(quality.MULTIVIEW_SAMPLER["steps"], 8)
        self.assertEqual(quality.MULTIVIEW_SAMPLER["cfg"], 1.0)

    def test_the_note_says_where_the_colour_comes_from(self):
        note = quality.reconstruction_note(
            quality.choose_reconstruction(8, 16), 8)
        self.assertIn("tier 2 of 4", note)
        self.assertIn("octree 256", note)
        self.assertIn("no model-generated paint stage", note)

    def test_the_note_does_not_claim_a_texture_that_was_switched_off(self):
        """Saying "textured" while texturing is off is the exact kind of note
        this file exists to avoid."""
        note = quality.reconstruction_note(
            quality.choose_reconstruction(8, 16), 8, textured=False)
        self.assertIn("texturing switched off", note)
        self.assertNotIn("textured from your photographs", note)
        self.assertNotIn("paint stage", note)

    def test_the_note_separates_shape_from_colour(self):
        """More photographs must be reported as helping COLOUR, because that
        is the only thing they are allowed to change."""
        note = quality.reconstruction_note(
            quality.choose_reconstruction(8, 16), 8, real_angles=1,
            colour_views=4)
        self.assertIn("single sharpest photograph", note)
        self.assertIn("coloured from 4 views", note)
        self.assertIn("without ever distorting the shape", note)


class RegistryTests(unittest.TestCase):
    def test_every_tier_model_is_in_the_registry(self):
        known = {m.name for m in DEFAULT_MODELS}
        for tier in quality.RECON_TIERS:
            for name in tier.models:
                with self.subTest(tier=tier.name, model=name):
                    self.assertIn(name, known)

    def test_the_mesh_models_declare_their_hardware_floor(self):
        by_name = {m.name: m for m in DEFAULT_MODELS}
        for name in ("hunyuan3d-v2", "hunyuan3d-v2-mv", "hunyuan3d-v21"):
            meta = by_name[name].meta or {}
            self.assertIn("min_vram_gb", meta)
            self.assertIn("min_ram_gb", meta)

    def test_the_licence_states_the_untextured_limit(self):
        """A user reading the model list should learn this before a 4.6 GB
        download, not after a grey mesh comes out."""
        entry = next(m for m in DEFAULT_MODELS if m.name == "hunyuan3d-v2")
        self.assertIn("untextured", entry.license.lower())


class TemplateTests(unittest.TestCase):
    def setUp(self):
        self.template = WorkflowLibrary(WORKFLOWS).load("reconstruct")
        self.graph = build_workflow(self.template,
                                    {"image": "src.png", "seed": 3})

    def test_the_task_and_nodes_are_allowed(self):
        self.assertEqual(self.template["task"], "reconstruct")
        self.assertIn("reconstruct", ALLOWED_TASKS)
        for node in self.graph.values():
            self.assertIn(node["class_type"], ALLOWED_NODE_TYPES)

    def test_it_needs_no_custom_node_pack(self):
        """Every node here ships in ComfyUI core. TripoSR/InstantMesh/TRELLIS
        were rejected precisely because their packs pin an older
        torch/transformers than this machine runs, and installing them would
        break the working InstantID and controlnet_aux."""
        core = {"ImageOnlyCheckpointLoader", "LoadImage", "CLIPVisionEncode",
                "Hunyuan3Dv2Conditioning", "EmptyLatentHunyuan3Dv2",
                "KSampler", "VAEDecodeHunyuan3D", "VoxelToMesh", "SaveGLB",
                # Not optional, and not obvious: Hunyuan3D's DiT reuses the
                # Flux blocks and sets pe=None, so without the AuraFlow
                # sampling patch KSampler dies inside apply_rope with
                # "'NoneType' object has no attribute 'dtype'". Both come
                # straight from ComfyUI's own shipped template.
                "ModelSamplingAuraFlow", "FluxGuidance"}
        self.assertEqual({n["class_type"] for n in self.graph.values()}, core)

    def test_the_sampling_patch_is_between_checkpoint_and_sampler(self):
        """The regression guard for the pe=None crash: the sampler must take
        its model from ModelSamplingAuraFlow, never straight from the
        checkpoint loader."""
        ks = next(d for d in self.graph.values()
                  if d["class_type"] == "KSampler")
        patch = next(n for n, d in self.graph.items()
                     if d["class_type"] == "ModelSamplingAuraFlow")
        self.assertEqual(ks["inputs"]["model"], [patch, 0])
        guide = next(n for n, d in self.graph.items()
                     if d["class_type"] == "FluxGuidance")
        self.assertEqual(ks["inputs"]["positive"], [guide, 0])

    def test_it_ends_in_a_glb(self):
        save = next(d for d in self.graph.values()
                    if d["class_type"] == "SaveGLB")
        mesh = next(n for n, d in self.graph.items()
                    if d["class_type"] == "VoxelToMesh")
        self.assertEqual(save["inputs"]["mesh"], [mesh, 0])

    def test_the_checkpoint_supplies_its_own_clip_vision(self):
        """CLIPVisionLoader has no standalone model on this machine, so the
        graph must take CLIP_VISION from the checkpoint loader's slot 1."""
        enc = next(d for d in self.graph.values()
                   if d["class_type"] == "CLIPVisionEncode")
        loader = next(n for n, d in self.graph.items()
                      if d["class_type"] == "ImageOnlyCheckpointLoader")
        self.assertEqual(enc["inputs"]["clip_vision"], [loader, 1])


class MeshViewTests(unittest.TestCase):
    """Which frames are allowed to describe the SHAPE.

    Each of these guards a defect that reached a real render: a mesh 1.81 x
    1.97 x 0.03 (a flat sheet), and figures 40-50% too deep."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)

    def _frame(self, azimuth, real, aspect=2.0):
        im = Image.new("RGB", (128, 128), (128, 128, 128))
        h = int(100 * min(1.0, aspect / 2.5))
        w = max(2, int(h / aspect))
        ImageDraw.Draw(im).rectangle(
            [64 - w // 2, 64 - h // 2, 64 + w // 2, 64 + h // 2],
            fill=(200, 60, 40))
        buf = __import__("io").BytesIO()
        im.save(buf, format="PNG")
        return self.s.store.save_upload(
            f"angle_{azimuth}.png", buf.getvalue(),
            meta={"azimuth": azimuth, "synthetic": not real}).id

    def test_a_far_off_frame_is_not_pressed_into_a_slot(self):
        """min() always returns something. An orbit covering only the front
        used to hand a near-front frame to the BACK input, telling the model
        both sides of the subject look the same."""
        ids = [self._frame(a, True) for a in (0, 20, 40)]
        picked = self.s._views_for_mesh(ids)
        self.assertIn("front", picked)
        self.assertNotIn("back", picked)
        self.assertNotIn("left", picked)

    def test_real_only_excludes_synthesised_views(self):
        real = self._frame(0, True)
        self._frame(180, False)
        self.assertEqual(self.s._views_for_mesh(
            [a for a in (real,)] + [], real_only=True), {"front": real})

    def test_shape_uses_photographs_and_colour_uses_everything(self):
        ids = [self._frame(0, True), self._frame(180, False)]
        self.assertEqual(set(self.s._views_for_mesh(ids, real_only=True)),
                         {"front"})
        self.assertEqual(set(self.s._views_for_mesh(ids)), {"front", "back"})

    def test_a_differently_framed_photo_is_dropped_from_the_shape(self):
        """Measured on a real dataset: the photographs handed to the mesh had
        silhouette aspects of 1.8 and 1.46 while rendered views of the same
        person had 4.45 — three different objects, one model."""
        job = Job(id="j", type="avatar", payload={})
        views = {"front": self._frame(0, True, aspect=2.0),
                 "back": self._frame(180, True, aspect=2.1),
                 "left": self._frame(270, True, aspect=6.0)}
        keep = self.s._consistent_real_views(job, views)
        self.assertEqual(set(keep), {"front", "back"})

    @staticmethod
    def _mask(w, h, box):
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).rectangle(box, fill=255)
        return m

    def test_a_subject_running_along_an_edge_is_penalised(self):
        """Measured on a real nine-photo set: two photographs had the
        silhouette running along 0.68 and 0.60 of the bottom edge with squat
        aspects (1.26, 1.21). Both are fragments of a person."""
        clean = self._mask(200, 400, (60, 20, 140, 360))     # nothing touching
        cropped = self._mask(200, 400, (20, 120, 180, 399))  # wide, edge-bound
        self.assertEqual(Services._framing_penalty(clean), 0.0)
        self.assertGreater(Services._framing_penalty(cropped), 0.5)

    def test_feet_at_the_bottom_are_not_treated_as_a_crop(self):
        """A full-length photo touches the bottom edge at the feet. Punishing
        that would reject exactly the photographs that work best."""
        standing = self._mask(200, 400, (85, 10, 115, 399))
        self.assertLess(Services._framing_penalty(standing), 0.2)

    def test_framing_outranks_sharpness_for_the_orbit_source(self):
        coverage = {b: [] for b in Services.VIEW_BINS}
        coverage["front"] = ["fragment", "whole"]
        picked = self.s._best_orbit_source(
            coverage, ["fragment", "whole"],
            {"fragment": 9000.0, "whole": 10.0},
            {"fragment": 0.9, "whole": 0.0})
        self.assertEqual(picked, "whole")

    def test_an_unrepeated_view_answer_is_not_acted_on(self):
        """The classifier is non-deterministic — the same photo came back
        'left' then 'front', and one run put all nine photos of a set in
        'front'. A bin is only trusted when a second, independent ask agrees,
        because an unconfirmed 'back' splices a front view into the orbit at
        180 degrees."""
        answers = iter(["left", "front"])
        self.s._classify_view = lambda _im: next(answers)
        view, sure = self.s._confident_view(Image.new("RGB", (8, 8)))
        self.assertEqual((view, sure), ("front", False))

    def test_a_repeated_view_answer_is_trusted(self):
        self.s._classify_view = lambda _im: "left"
        self.assertEqual(self.s._confident_view(Image.new("RGB", (8, 8))),
                         ("left", True))

    def test_unused_view_slots_are_removed_not_duplicated(self):
        template = WorkflowLibrary(WORKFLOWS).load_named("reconstruct_mv")
        graph = build_workflow(template, {"front": "f.png", "back": "b.png",
                                          "seed": 1})
        graph = Services._prune_mesh_views(graph, {"front", "back"})
        cond = graph["10"]["inputs"]
        self.assertEqual(set(cond), {"front", "back"})
        for node in ("3", "7", "5", "9"):       # left and right loaders
            self.assertNotIn(node, graph)
        # and what survives still references only nodes that exist
        for spec in graph.values():
            for value in spec["inputs"].values():
                if isinstance(value, list) and len(value) == 2:
                    self.assertIn(str(value[0]), graph)


class PhotoChoiceTests(unittest.TestCase):
    """Which of the uploaded photos actually gets used."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)

    @staticmethod
    def _sharp():
        im = Image.new("RGB", (256, 256), "black")
        d = ImageDraw.Draw(im)
        for i in range(0, 256, 8):
            d.line([(i, 0), (i, 256)], fill="white", width=3)
        return im

    def test_sharp_scores_above_blurry(self):
        sharp = self._sharp()
        blurry = sharp.filter(__import__("PIL.ImageFilter", fromlist=["x"])
                              .GaussianBlur(6))
        self.assertGreater(self.s._focus_score(sharp),
                           self.s._focus_score(blurry))

    def test_the_orbit_starts_from_the_most_frontal_bin(self):
        """SV3D orbits FROM a front view; started from behind it invents the
        face, which is the one thing an avatar cannot get wrong."""
        coverage = {b: [] for b in Services.VIEW_BINS}
        coverage["back"] = ["b1"]
        coverage["front-right"] = ["fr1"]
        self.assertEqual(
            self.s._best_orbit_source(coverage, ["b1", "fr1"],
                                      {"b1": 999.0, "fr1": 1.0}),
            "fr1", "a sharp back view must not beat a frontal one")

    def test_within_a_bin_the_sharpest_photo_wins(self):
        coverage = {b: [] for b in Services.VIEW_BINS}
        coverage["front"] = ["dull", "crisp"]
        self.assertEqual(
            self.s._best_orbit_source(coverage, ["dull", "crisp"],
                                      {"dull": 10.0, "crisp": 900.0}),
            "crisp")

    def test_with_no_classified_views_the_sharpest_photo_is_used(self):
        """Not asset_ids[0] — the old code's choice, which meant every photo
        after the first changed nothing at all."""
        coverage = {b: [] for b in Services.VIEW_BINS}
        self.assertEqual(
            self.s._best_orbit_source(coverage, ["first", "better"],
                                      {"first": 5.0, "better": 500.0}),
            "better")

    def test_the_face_reference_is_a_frontal_photo(self):
        coverage = {b: [] for b in Services.VIEW_BINS}
        coverage["left"] = ["side"]
        coverage["front"] = ["face"]
        self.assertEqual(
            self.s._best_face_photo(coverage, ["side", "face"],
                                    {"side": 900.0, "face": 10.0}),
            "face")


if __name__ == "__main__":
    unittest.main()


class MeshOptionTests(unittest.TestCase):
    """Texturing and body-completion are choices, not fixed behaviour."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.s = Services(Settings(
            data_dir=Path(self.tmp.name), inpaint_backend="mock",
            segment_backend="mock", critic_model="", first_run_setup=False,
            comfyui_dir=""))
        self.addCleanup(self.s.stop)

    def _stub_a_mesh_build(self):
        """Everything _build_mesh needs, replaced with the cheapest thing that
        still runs the real decision code."""
        painted = []
        self.s.hardware = type("HW", (), {"vram_gb": 8.0, "ram_gb": 16.0})()
        self.s._require_comfy = lambda job: None
        self.s._ensure_model = lambda name, job: None
        self.s._views_for_mesh = lambda *a, **k: {}
        self.s._prepare_heavy_render = lambda job, need_gb=0: None
        self.s._free_vram = lambda job: None
        self.s.open_asset_image = lambda aid: Image.new("RGB", (32, 32))
        self.s.comfy.upload_image = lambda image, prefix: "x.png"
        self.s.comfy.submit = lambda graph: "pid"
        self.s.comfy.wait_for_mesh = lambda pid: (b"GLB", "m.glb")
        self.s.comfy.installed_checkpoints = lambda: []

        def fake_texture(job, data, photos):
            painted.append(len(photos))
            return b"PAINTED", {"seen_pct": 90.0, "orientation_iou": 0.9}

        self.s._texture_mesh = fake_texture
        return painted

    def test_texturing_on_paints_the_mesh(self):
        painted = self._stub_a_mesh_build()
        job = Job(id="j", type="avatar", payload={})
        out = self.s._build_mesh(job, [], "src", texture=True)
        self.assertIsNotNone(out)
        self.assertTrue(painted, "the colouring pass never ran")
        self.assertTrue(out["textured"])

    def test_texturing_off_skips_the_colouring_pass(self):
        """Bare geometry is a legitimate deliverable — it is what you want if
        you are going to paint or sculpt the mesh yourself."""
        painted = self._stub_a_mesh_build()
        job = Job(id="j", type="avatar", payload={})
        out = self.s._build_mesh(job, [], "src", texture=False)
        self.assertIsNotNone(out)
        self.assertFalse(painted, "colouring ran despite being switched off")
        self.assertFalse(out["textured"])
        self.assertTrue(any("switched off" in e["msg"] for e in job.logs),
                        [e["msg"] for e in job.logs])

    def test_the_texture_flag_reaches_the_builder(self):
        import inspect
        sig = inspect.signature(Services._build_mesh)
        self.assertIn("texture", sig.parameters)
        self.assertIs(sig.parameters["texture"].default, True)

    def test_a_cut_off_subject_is_detected(self):
        """The trigger for completing a body: the silhouette runs along a
        frame edge. Measured on real photos: 0.14 and 0.28 of the bottom
        edge, against a 0.12 threshold."""
        matte = Image.new("L", (200, 400), 0)
        ImageDraw.Draw(matte).rectangle([60, 150, 140, 399], fill=255)
        edges = self.s._subject_edges(matte)
        self.assertGreaterEqual(edges["bottom"], self.s._CUTOFF_FRACTION)
        self.assertEqual(edges["top"], 0.0)

    def test_a_whole_subject_is_left_alone(self):
        matte = Image.new("L", (200, 400), 0)
        ImageDraw.Draw(matte).rectangle([60, 20, 140, 360], fill=255)
        edges = self.s._subject_edges(matte)
        self.assertLess(edges["bottom"], self.s._CUTOFF_FRACTION)

    def test_completion_returns_the_input_when_nothing_is_cut(self):
        self.s._cut_edges = lambda im: {}
        photo = Image.new("RGB", (64, 64))
        job = Job(id="j", type="avatar", payload={})
        self.assertIs(self.s._complete_subject(job, photo), photo)


class Scene3DTests(unittest.TestCase):
    """A photograph turned into somewhere you can move around.

    Different problem from an avatar: that is an OBJECT you orbit, this is a
    SCENE you stand inside — and it was unreachable from a prompt, which is
    the worst kind of missing feature because everything about it worked."""

    def test_it_asks_for_a_place_not_a_look(self):
        for text in ("make this 3d", "turn this photo into a 3d scene",
                     "rebuild this into 3d", "make a 3d environment from this",
                     "I want to walk around in this photo",
                     "let me move around this room", "explore this photo"):
            self.assertTrue(quality.scene3d_intent(text), text)

    def test_a_3d_LOOK_is_an_ordinary_image_request(self):
        """"3d render" and "make it look 3d" ask for a style. Routing those
        to a geometry model would hand back a mesh nobody asked for."""
        for text in ("give it a 3d render style", "make it look 3d",
                     "add 3d text", "a 3d cartoon of a cat",
                     "make this a 3d printed model",
                     "change the background to a forest", "upscale this"):
            self.assertFalse(quality.scene3d_intent(text), text)

    def test_the_route_exists_end_to_end(self):
        self.assertEqual(quality.OPERATION_TASK["SCENE_3D"], "scene3d")
        self.assertIn("scene3d", quality.EDIT_TASKS)
        self.assertIn("scene3d", ALLOWED_TASKS)
        template = WorkflowLibrary(WORKFLOWS).load("scene3d")
        self.assertEqual(template["task"], "scene3d")
        for node in template["graph"].values():
            self.assertIn(node["class_type"], ALLOWED_NODE_TYPES)

    def test_building_a_scene_happens_after_every_edit(self):
        """It consumes the FINISHED picture — running it first would mesh the
        photo before the edit that changed what the photo shows."""
        ordered = quality.order_steps([
            {"task": "scene3d"}, {"task": "img2img"}, {"task": "outpaint"}])
        self.assertEqual(ordered[-1]["task"], "scene3d")

    def test_a_request_for_a_scene_is_added_to_the_plan(self):
        llm = OneJsonPlan({"steps": [
            {"operation": "CHANGE_STYLE", "target": "", "instruction": "x"}]})
        steps = quality.plan_edit(llm, "make this 3d", has_mask=False)
        self.assertIn("scene3d", [s["task"] for s in steps])


class LayerAlignmentTests(unittest.TestCase):
    """Two reconstructions of one photograph must end up the same size.

    A monocular geometry model recovers shape up to an overall scale and
    picks a different one per image: measured on two layers of the same
    photo, 1.53 x 2.32 x 3.19 against 1.42 x 1.96 x 2.07. Merged untouched,
    the second layer sits inside the first instead of behind it."""

    @staticmethod
    def _load():
        import importlib.util
        path = (Path(__file__).parent.parent / "app" / "tools"
                / "merge_meshes.py")
        spec = importlib.util.spec_from_file_location("merge_meshes", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_a_known_scale_is_recovered(self):
        try:
            import numpy as np
        except ImportError:                     # pragma: no cover
            self.skipTest("numpy unavailable")
        merge = self._load()
        rng = np.random.default_rng(7)
        # A slab of scene in front of a camera at the origin looking down -Z.
        points = np.column_stack([
            rng.uniform(-1, 1, 4000), rng.uniform(-1, 1, 4000),
            rng.uniform(-4, -2, 4000)])
        for truth in (0.5, 0.857, 1.0, 1.9):
            found = merge.align_scale(points, points / truth)
            self.assertAlmostEqual(found, truth, places=2, msg=str(truth))

    def test_too_little_overlap_leaves_the_layer_alone(self):
        try:
            import numpy as np
        except ImportError:                     # pragma: no cover
            self.skipTest("numpy unavailable")
        merge = self._load()
        a = np.array([[0.0, 0.0, -2.0]] * 5)
        self.assertEqual(merge.align_scale(a, a * 3.0), 1.0)
