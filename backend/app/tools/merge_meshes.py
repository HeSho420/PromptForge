"""Combine several GLB layers into one file, keeping each one's texture.

Run with ComfyUI's interpreter, where trimesh already lives.

    python merge_meshes.py <out.glb> <layer.glb> [<layer.glb> ...]

Used for a layered scene: one photograph is one camera frustum, so everything
behind the foreground is missing and moving sideways opens black wedges. The
second layer is the same scene with the foreground painted out and meshed
again — geometry for what was probably behind. Merging keeps them as separate
primitives so each keeps its own texture image; welding them into one mesh
would force a shared material and lose both.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# trimesh lives in ComfyUI's interpreter, not PromptForge's, so it is imported
# where it is used. The alignment maths below is plain numpy and is unit-tested
# from the backend's own environment.


def _bin_radii(verts: np.ndarray, bins: int = 96) -> dict[tuple[int, int], float]:
    """Median distance from the camera, per direction.

    The geometry model puts the camera at the origin looking down -Z, so
    x/-z and y/-z are image coordinates: binning on those groups vertices
    that came from the same part of the SAME photograph."""
    z = -verts[:, 2]
    ok = z > 1e-3
    if not ok.any():
        return {}
    v = verts[ok]
    z = z[ok]
    u = np.clip(((v[:, 0] / z) * 0.5 + 0.5) * bins, 0, bins - 1).astype(int)
    w = np.clip(((v[:, 1] / z) * 0.5 + 0.5) * bins, 0, bins - 1).astype(int)
    radius = np.linalg.norm(v, axis=1)
    out: dict[tuple[int, int], list[float]] = {}
    for key, r in zip(zip(u.tolist(), w.tolist(), strict=False), radius.tolist(), strict=False):
        out.setdefault(key, []).append(r)
    return {k: float(np.median(vs)) for k, vs in out.items() if len(vs) >= 3}


def align_scale(reference: np.ndarray, other: np.ndarray) -> float:
    """The single number that puts `other` in `reference`'s metric frame.

    A monocular geometry model recovers shape up to an overall scale, and it
    picks a different one for every image it is given: measured on two layers
    of the SAME photograph, one came out 1.53 x 2.32 x 3.19 and the other
    1.42 x 1.96 x 2.07. Merged untouched, the second layer sits inside the
    first instead of behind it, which looks like a doubled, broken scene.

    The two layers differ only where the foreground was painted out, so most
    directions see the same surface in both. The MEDIAN ratio over shared
    directions therefore recovers the scale and ignores the region that
    genuinely changed."""
    a, b = _bin_radii(reference), _bin_radii(other)
    shared = set(a) & set(b)
    if len(shared) < 32:
        return 1.0
    ratios = np.array([a[k] / b[k] for k in shared if b[k] > 1e-6])
    return float(np.median(ratios)) if len(ratios) else 1.0


def main() -> int:
    import trimesh

    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    out = Path(sys.argv[1])
    scene = trimesh.Scene()
    report = []
    anchor: np.ndarray | None = None
    for i, path in enumerate(sys.argv[2:]):
        loaded = trimesh.load(path, force="scene")
        geometries = (loaded.geometry.values()
                      if isinstance(loaded, trimesh.Scene) else [loaded])
        for j, geom in enumerate(geometries):
            if not len(getattr(geom, "faces", [])):
                continue
            scale = 1.0
            if anchor is None:
                anchor = np.asarray(geom.vertices).copy()
            else:
                scale = align_scale(anchor, np.asarray(geom.vertices))
                if abs(scale - 1.0) > 1e-3:
                    geom.vertices = np.asarray(geom.vertices) * scale
            scene.add_geometry(geom, node_name=f"layer{i}_{j}",
                               geom_name=f"layer{i}_{j}")
            report.append({"layer": i, "vertices": int(len(geom.vertices)),
                           "faces": int(len(geom.faces)),
                           "scale": round(scale, 4),
                           "textured": type(geom.visual).__name__})
    if not report:
        print("nothing to merge", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(scene.export(file_type="glb"))
    total = sum(r["faces"] for r in report)
    print(json.dumps({"layers": len(report), "faces": total,
                      "bytes": out.stat().st_size, "parts": report}))
    print(f"Merged {len(report)} layers into one scene "
          f"({total:,} triangles) — the second layer is a reconstruction of "
          f"what stood behind the foreground, not a photograph of it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
