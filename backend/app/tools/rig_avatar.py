"""Give an avatar mesh a posable skeleton, headlessly, with Blender.

    blender --background --factory-startup --python rig_avatar.py -- \
            <in.glb> <out.glb> [--pose-test <posed.glb>]

The mesh arrives as a static scan-like figure (Hunyuan3D output, textured).
This builds a humanoid armature whose joints are placed by MEASURING the
mesh — band widths locate the neck and shoulders, symmetry locates left and
right — then binds it with Blender's heat solver and exports a GLB that any
DCC or engine can pose.

Three lessons from how this class of mesh actually behaves, built in:

  clean first     the heat solver ("Automatic Weights") fails outright on
                  doubled vertices and non-manifold shells, which is what
                  AI meshers emit. Merge-by-distance runs before anything.

  never refuse    when bone heat fails anyway, envelope weights bind
                  instead. Cruder deformation beats an unrigged file, and
                  the report says which method bound.

  prove it        a --pose-test export bends the head and an arm and saves
                  a second GLB. If the pose file looks wrong, the rig IS
                  wrong, whatever the report claims.

Bust or full figure is detected from proportions: a head-and-torso mesh
gets spine/neck/head/shoulders/arms only; legs are added when the lower
half actually contains two separable columns.
"""
import json
import math
import sys

import bpy
from mathutils import Vector


def parse_args() -> dict:
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) < 2:
        print(json.dumps({"error": "usage: -- in.glb out.glb "
                                   "[--pose-test posed.glb]"}))
        sys.exit(2)
    out = {"src": argv[0], "dst": argv[1], "pose_test": None}
    if "--pose-test" in argv:
        out["pose_test"] = argv[argv.index("--pose-test") + 1]
    return out


def import_mesh(path: str) -> bpy.types.Object:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        print(json.dumps({"error": "no mesh in file"}))
        sys.exit(1)
    bpy.ops.object.select_all(action="DESELECT")
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def clean(obj: bpy.types.Object) -> dict:
    """Merge doubles — the single most frequent cause of a failed bind."""
    bpy.context.view_layer.objects.active = obj
    before = len(obj.data.vertices)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-4)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    # Smooth shading: with flat per-face normals the glTF exporter splits
    # every face corner into its own vertex — measured 300k vertices
    # exploding to 1.3M and a 75 MB file. Shared smooth normals let the
    # exporter re-weld everything the UV seams don't genuinely split.
    bpy.ops.object.shade_smooth()
    return {"vertices_before": before, "vertices": len(obj.data.vertices)}


def measure(obj: bpy.types.Object) -> dict:
    """Joint landmarks from the mesh itself (Blender Z-up after glTF import).

    Band widths do the anatomy: the neck is the narrowest band in the upper
    quarter, the shoulders the widest band just below it. A full figure is
    recognised by the lower third splitting into two columns (legs); anything
    else is treated as a bust and gets no leg bones to flail."""
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    xs = [v.x for v in verts]
    ys = [v.y for v in verts]
    zs = [v.z for v in verts]
    lo, hi = min(zs), max(zs)
    height = hi - lo or 1.0
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2

    def band(f0: float, f1: float) -> list[Vector]:
        z0, z1 = lo + f0 * height, lo + f1 * height
        return [v for v in verts if z0 <= v.z <= z1]

    def width(f0: float, f1: float) -> float:
        b = band(f0, f1)
        return (max(v.x for v in b) - min(v.x for v in b)) if b else 0.0

    # Neck: narrowest 4%-band between 70% and 96% of the height.
    neck_f, neck_w = 0.85, float("inf")
    f = 0.70
    while f < 0.96:
        w = width(f, f + 0.04)
        if 0 < w < neck_w:
            neck_w, neck_f = w, f + 0.02
        f += 0.02
    # Shoulders: widest band in the 15% below the neck.
    sh_f, sh_w = max(0.55, neck_f - 0.10), 0.0
    f = max(0.50, neck_f - 0.15)
    while f < neck_f:
        w = width(f, f + 0.04)
        if w > sh_w:
            sh_w, sh_f = w, f + 0.02
        f += 0.02

    # Legs: does the 20%-30% band split into two x-columns around centre?
    legs = False
    low = band(0.18, 0.32)
    if len(low) > 200:
        left = [v for v in low if v.x < cx - 0.02 * height]
        right = [v for v in low if v.x > cx + 0.02 * height]
        gap = [v for v in low
               if abs(v.x - cx) <= 0.015 * height]
        aspect = height / max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
        legs = (len(left) > 100 and len(right) > 100
                and len(gap) < 0.1 * len(low)
                and aspect > 1.9)   # a bust is squat; a standing figure tall
    return {"lo": lo, "height": height, "cx": cx, "cy": cy,
            "neck_f": neck_f, "shoulder_f": sh_f,
            "shoulder_half": sh_w / 2, "full_body": legs}


def build_armature(m: dict) -> bpy.types.Object:
    H, lo, cx, cy = m["height"], m["lo"], m["cx"], m["cy"]

    def z(f: float) -> float:
        return lo + f * H

    bpy.ops.object.armature_add(enter_editmode=True,
                                location=(0, 0, 0))
    arm = bpy.context.view_layer.objects.active
    arm.name = "Avatar_Rig"
    eb = arm.data.edit_bones
    for b in list(eb):
        eb.remove(b)

    def bone(name: str, head, tail, parent=None, connect=False):
        b = eb.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        if parent is not None:
            b.parent = parent
            b.use_connect = connect
        return b

    sh_f = m["shoulder_f"]
    neck_f = m["neck_f"]
    sh_half = max(m["shoulder_half"] * 0.75, 0.05 * H)
    full = m["full_body"]

    hips_f = 0.50 if full else 0.02
    hips = bone("hips", (cx, cy, z(hips_f)), (cx, cy, z(hips_f + 0.08)))
    spine = bone("spine", hips.tail, (cx, cy, z(sh_f - 0.06)), hips, True)
    chest = bone("chest", spine.tail, (cx, cy, z(neck_f)), spine, True)
    neck = bone("neck", chest.tail, (cx, cy, z(min(0.99, neck_f + 0.05))),
                chest, True)
    bone("head", neck.tail, (cx, cy, z(1.0) + 0.02 * H), neck, True)

    arm_len = 0.16 * H if full else 0.22 * H
    for side, sx in (("L", 1), ("R", -1)):
        sh = bone(f"shoulder.{side}",
                  (cx + sx * 0.02 * H, cy, z(neck_f - 0.02)),
                  (cx + sx * sh_half, cy, z(sh_f + 0.02)), chest)
        up = bone(f"upper_arm.{side}", sh.tail,
                  (sh.tail.x + sx * 0.02 * H, cy,
                   sh.tail.z - arm_len), sh)
        fo = bone(f"forearm.{side}", up.tail,
                  (up.tail.x, cy, up.tail.z - arm_len * 0.9), up, True)
        bone(f"hand.{side}", fo.tail,
             (fo.tail.x, cy, fo.tail.z - 0.06 * H), fo, True)

    if full:
        hip_half = 0.09 * H
        for side, sx in (("L", 1), ("R", -1)):
            th = bone(f"thigh.{side}",
                      (cx + sx * hip_half, cy, z(0.50)),
                      (cx + sx * hip_half, cy, z(0.28)), hips)
            sn = bone(f"shin.{side}", th.tail,
                      (th.tail.x, cy, z(0.06)), th, True)
            bone(f"foot.{side}", sn.tail,
                 (sn.tail.x, cy - 0.08 * H, z(0.02)), sn, True)

    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


def bind(obj: bpy.types.Object, arm: bpy.types.Object) -> str:
    """Heat weights, falling back to envelope weights rather than failing."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        groups = [g for g in obj.vertex_groups]
        weighted = sum(1 for v in obj.data.vertices if v.groups)
        if groups and weighted > 0.5 * len(obj.data.vertices):
            return "bone_heat"
    except Exception:  # noqa: BLE001 — the fallback is the answer
        pass
    for g in list(obj.vertex_groups):
        obj.vertex_groups.remove(g)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
    return "envelope"


def export(path: str) -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB",
                              export_yup=True, export_skins=True,
                              export_apply=False)


def pose_test(obj: bpy.types.Object, arm: bpy.types.Object,
              path: str) -> None:
    """Bend the head and raise an arm, BAKE it, export, restore.

    Baking (applying the armature modifier on a duplicate) is the point:
    the glTF exporter writes skins in REST position by default and most
    loaders never evaluate skinning, so an exported live pose renders
    identical to rest and proves nothing — measured on the first attempt.
    A baked static mesh shows the deformation in any viewer."""
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    rotations = {"head": ("X", 20), "upper_arm.L": ("Z", 45),
                 "forearm.L": ("X", -30)}
    for name, (axis, deg) in rotations.items():
        pb = arm.pose.bones.get(name)
        if pb is None:
            continue
        pb.rotation_mode = "XYZ"
        setattr(pb.rotation_euler, axis.lower(), math.radians(deg))
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate()
    dup = bpy.context.view_layer.objects.active
    for mod in list(dup.modifiers):
        if mod.type == "ARMATURE":
            bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.ops.object.select_all(action="DESELECT")
    dup.select_set(True)
    bpy.context.view_layer.objects.active = dup
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB",
                              use_selection=True, export_yup=True)
    bpy.ops.object.delete()

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    for name in rotations:
        pb = arm.pose.bones.get(name)
        if pb is not None:
            pb.rotation_euler = (0, 0, 0)
    bpy.ops.object.mode_set(mode="OBJECT")


def main() -> None:
    args = parse_args()
    obj = import_mesh(args["src"])
    stats = clean(obj)
    m = measure(obj)
    arm = build_armature(m)
    method = bind(obj, arm)
    export(args["dst"])
    if args["pose_test"]:
        pose_test(obj, arm, args["pose_test"])
    print(json.dumps({
        "bones": len(arm.data.bones),
        "weights": method,
        "full_body": m["full_body"],
        "vertices": stats["vertices"],
        "merged": stats["vertices_before"] - stats["vertices"],
        "neck_at": round(m["neck_f"], 3),
        "shoulders_at": round(m["shoulder_f"], 3),
    }))


if __name__ == "__main__":
    main()
