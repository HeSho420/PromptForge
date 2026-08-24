"""ArcFace identity similarity between two images.

Runs in ComfyUI's OWN interpreter (the InstantID pack installs
insightface there, and the antelopev2 models already live under
ComfyUI's models dir) — the backend venv never needs onnxruntime.

argv: <reference.png> <candidate.png> <insightface_root>
stdout: one JSON line:
  {"similarity": 0.78, "ref_faces": 1, "cand_faces": 1}
  {"error": "<why>"} when either image has no detectable face.

Calibrated on this machine (2026-08-24): the SAME person measures
0.88-0.98 across pixel-preserving edits and 0.6-0.8 across full
re-renders; a DIFFERENT person measures ~0.0-0.2 (PhotoMaker's generic
look-alike scored 0.213). Bands used by the pipeline: >=0.50 same
person, 0.35-0.50 recognizable, <0.35 not the person.
"""
import json
import sys


def main() -> int:
    ref_path, cand_path, root = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
    except Exception as exc:  # noqa: BLE001 — reported, not raised
        print(json.dumps({"error": f"insightface unavailable: {exc}"}))
        return 1
    try:
        app = FaceAnalysis(name="antelopev2", root=root,
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"error": f"antelopev2 failed to load: {exc}"}))
        return 1

    def largest_embedding(path: str):
        img = cv2.imread(path)
        if img is None:
            return None, 0
        faces = app.get(img)
        if not faces:
            return None, 0
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                   * (f.bbox[3] - f.bbox[1]))
        return face.normed_embedding, len(faces)

    def largest_face(path: str):
        img = cv2.imread(path)
        if img is None:
            return None, 0
        faces = app.get(img)
        if not faces:
            return None, 0
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0])
                   * (f.bbox[3] - f.bbox[1])), len(faces)

    ref_face, n_ref = largest_face(ref_path)
    cand_face, n_cand = largest_face(cand_path)
    if ref_face is None or cand_face is None:
        which = "reference" if ref_face is None else "candidate"
        print(json.dumps({"error": f"no face found in the {which} image",
                          "ref_faces": n_ref, "cand_faces": n_cand}))
        return 1
    ref, cand = ref_face.normed_embedding, cand_face.normed_embedding
    # cand_box: the candidate's face bbox [x1, y1, x2, y2] — the crop
    # anchor for an identity-conditioned face restore.
    print(json.dumps({"similarity": round(float(np.dot(ref, cand)), 4),
                      "ref_faces": n_ref, "cand_faces": n_cand,
                      "cand_box": [round(float(v), 1)
                                   for v in cand_face.bbox]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
