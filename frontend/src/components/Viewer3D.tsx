import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

/**
 * A real 3D viewer for generated meshes.
 *
 * three.js rather than <model-viewer> deliberately: the same component has to
 * grow into walking through a reconstructed scene, and model-viewer only ever
 * orbits an object. Everything is bundled by Vite — the app is fully offline
 * and must never reach for a CDN.
 *
 * `mode="orbit"` circles a subject (an avatar). `mode="walk"` puts the camera
 * inside the scene and drives it with WASD, for scene reconstructions.
 */
export function Viewer3D({
  url,
  mode = "orbit",
  height = 460,
}: {
  url: string;
  mode?: "orbit" | "walk";
  height?: number;
}) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [detail, setDetail] = useState("");

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    // A new model starts loading again — otherwise the previous model's
    // "ready" and its triangle count sit under a blank canvas.
    setStatus("loading");
    setDetail("");

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101013);

    const camera = new THREE.PerspectiveCamera(
      mode === "walk" ? 70 : 40,
      host.clientWidth / height,
      0.01,
      500,
    );

    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    } catch {
      setStatus("error");
      setDetail("This browser has no WebGL.");
      return;
    }
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    // The panel is display:none when this mounts (every mode stays mounted),
    // so clientWidth is 0 and a 0-wide drawing buffer draws nothing ever
    // again. Start at a sane width and let the ResizeObserver below correct
    // it the moment the panel is actually shown.
    renderer.setSize(Math.max(host.clientWidth, 640), height);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    // Neutral studio light: the meshes are untextured, so shape has to read
    // from shading alone.
    scene.add(new THREE.HemisphereLight(0xffffff, 0x202028, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(3, 5, 4);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x88aaff, 1.0);
    rim.position.set(-4, 2, -3);
    scene.add(rim);

    // Orbit circles a subject. Walking needs the opposite: the camera turns
    // in place and moves independently, so OrbitControls is wrong for it —
    // dragging would swing you around a point instead of turning your head.
    const controls =
      mode === "orbit" ? new OrbitControls(camera, renderer.domElement) : null;
    if (controls) {
      controls.enableDamping = true;
      controls.dampingFactor = 0.08;
    }

    const keys = new Set<string>();
    const onKeyDown = (e: KeyboardEvent) => keys.add(e.key.toLowerCase());
    const onKeyUp = (e: KeyboardEvent) => keys.delete(e.key.toLowerCase());
    // Mouse-look by dragging: yaw and pitch the camera itself. No pointer
    // lock — it hijacks the cursor, which is hostile in a panel inside a page.
    let yaw = 0;
    let pitch = 0;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    const onDown = (e: PointerEvent) => {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      renderer.domElement.setPointerCapture(e.pointerId);
    };
    const onMove = (e: PointerEvent) => {
      if (!dragging) return;
      yaw -= (e.clientX - lastX) * 0.005;
      // Stop just short of straight up/down so the view cannot flip over.
      pitch = Math.max(
        -Math.PI / 2 + 0.05,
        Math.min(Math.PI / 2 - 0.05, pitch - (e.clientY - lastY) * 0.005),
      );
      lastX = e.clientX;
      lastY = e.clientY;
      camera.rotation.set(pitch, yaw, 0, "YXZ");
    };
    const onUp = () => {
      dragging = false;
    };
    // Alt-tabbing with a key held never delivers the keyup, which leaves the
    // camera driving itself when you come back.
    const onBlur = () => {
      keys.clear();
      dragging = false;
    };
    if (mode === "walk") {
      window.addEventListener("keydown", onKeyDown);
      window.addEventListener("keyup", onKeyUp);
      window.addEventListener("blur", onBlur);
      renderer.domElement.addEventListener("pointerdown", onDown);
      renderer.domElement.addEventListener("pointermove", onMove);
      renderer.domElement.addEventListener("pointerup", onUp);
      renderer.domElement.addEventListener("pointercancel", onUp);
    }

    let disposed = false;
    let root: THREE.Object3D | null = null;

    new GLTFLoader().load(
      url,
      (gltf) => {
        if (disposed) return;
        root = gltf.scene;
        // Centre on the origin and normalise the size — generated meshes come
        // out at arbitrary scale and offset.
        const box = new THREE.Box3().setFromObject(root);
        const size = box.getSize(new THREE.Vector3());
        const centre = box.getCenter(new THREE.Vector3());
        if (mode === "orbit") {
          // Centre and normalise so any subject frames the same way.
          root.position.sub(centre);
          const span = Math.max(size.x, size.y, size.z) || 1;
          root.scale.setScalar(2 / span);
        }
        // In walk mode the geometry keeps its own scale: a reconstructed
        // room is metric, and rescaling it to a 2-unit box would make the
        // walking speed below mean nothing.
        scene.add(root);

        let tris = 0;
        root.traverse((o) => {
          const m = o as THREE.Mesh;
          if (m.isMesh) {
            const g = m.geometry as THREE.BufferGeometry;
            tris += (g.index ? g.index.count : g.attributes.position.count) / 3;
            // Clay ONLY when the mesh carries no colour of its own. The
            // backend colours meshes with VERTEX colours (a glTF COLOR_0
            // accessor), not a texture map — testing for `.map` alone threw
            // away every photo-projected colour and showed grey instead.
            const mats = Array.isArray(m.material) ? m.material : [m.material];
            const hasColour = mats.some((mat) => {
              const std = mat as THREE.MeshStandardMaterial;
              return std && (std.map != null || std.vertexColors === true);
            });
            const hasVertexColour =
              (g.attributes as Record<string, unknown>).color != null;
            if (!hasColour && !hasVertexColour) {
              mats.forEach((mat) => mat?.dispose());
              m.material = new THREE.MeshStandardMaterial({
                color: 0xb9bcc4,
                roughness: 0.62,
                metalness: 0.0,
              });
            } else if (hasVertexColour) {
              // glTF gives COLOR_0 but the loader does not always switch the
              // material on to reading it.
              mats.forEach((mat) => {
                const std = mat as THREE.MeshStandardMaterial;
                if (std) { std.vertexColors = true; std.needsUpdate = true; }
              });
            }
          }
        });
        setDetail(`${Math.round(tris).toLocaleString()} triangles`);

        if (mode === "walk") {
          // Start where the original camera stood, at eye height.
          camera.position.set(centre.x, centre.y, box.max.z + 0.1);
          camera.rotation.set(0, 0, 0, "YXZ");
        } else {
          camera.position.set(0, 0.15, 3.4);
          controls?.target.set(0, 0, 0);
        }
        controls?.update();
        setStatus("ready");
      },
      undefined,
      (err) => {
        if (disposed) return;
        setStatus("error");
        setDetail(String((err as Error)?.message || err));
      },
    );

    const clock = new THREE.Clock();
    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      // Skip work while the panel is hidden (display:none -> zero width).
      if (host.clientWidth === 0) return;
      // A hidden panel skips the body, so getDelta() returns the whole
      // hidden duration on the first visible frame — a single-frame teleport
      // of speed * minutes. Clamp to one slow frame.
      const dt = Math.min(clock.getDelta(), 0.1);
      if (mode === "walk" && keys.size) {
        const speed = (keys.has("shift") ? 4 : 1.6) * dt;
        const dir = new THREE.Vector3();
        camera.getWorldDirection(dir);
        const right = new THREE.Vector3()
          .crossVectors(dir, camera.up)
          .normalize();
        const move = new THREE.Vector3();
        if (keys.has("w")) move.add(dir);
        if (keys.has("s")) move.sub(dir);
        if (keys.has("d")) move.add(right);
        if (keys.has("a")) move.sub(right);
        if (keys.has("q")) move.y -= 1;
        if (keys.has("e")) move.y += 1;
        if (move.lengthSq() > 0) {
          camera.position.add(move.normalize().multiplyScalar(speed));
        }
      }
      controls?.update();
      renderer.render(scene, camera);
    };
    tick();

    const onResize = () => {
      const w = host.clientWidth;
      if (!w) return;                       // still hidden; try again later
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    };
    window.addEventListener("resize", onResize);
    // A mode switch changes display:none -> block, which fires NO resize
    // event. Without this the viewer stays at its start size forever.
    const ro = new ResizeObserver(onResize);
    ro.observe(host);

    return () => {
      disposed = true;
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", onResize);
      ro.disconnect();
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", onBlur);
      controls?.dispose();
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointermove", onMove);
      renderer.domElement.removeEventListener("pointerup", onUp);
      renderer.domElement.removeEventListener("pointercancel", onUp);
      // Free the GPU buffers explicitly — several of these can be mounted at
      // once and WebGL contexts are a scarce resource.
      const killMaterial = (mat: THREE.Material | THREE.Material[] | null) => {
        for (const m of Array.isArray(mat) ? mat : [mat]) {
          if (!m) continue;
          // material.dispose() does NOT free its textures in three.js.
          for (const v of Object.values(m as unknown as Record<string, unknown>)) {
            const tex = v as THREE.Texture;
            if (tex && (tex as THREE.Texture).isTexture) tex.dispose();
          }
          m.dispose();
        }
      };
      root?.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.isMesh) {
          m.geometry.dispose();
          killMaterial(m.material);
        }
      });
      renderer.dispose();
      // dispose() tears down caches but leaves the WebGL context alive until
      // the canvas is collected. Browsers cap contexts at ~16, and every
      // mount and every model change makes one — so release it explicitly.
      renderer.forceContextLoss();
      renderer.domElement.remove();
    };
  }, [url, mode, height]);

  return (
    <div className="viewer3d">
      <div ref={hostRef} className="viewer3d-canvas" style={{ height }} />
      {status === "loading" && (
        <span className="viewer3d-note">Loading the model…</span>
      )}
      {status === "error" && (
        <span className="viewer3d-note err">Could not display it: {detail}</span>
      )}
      {status === "ready" && (
        <span className="viewer3d-note">
          {mode === "walk"
            ? "drag to look · W A S D to move · Q E for height · shift to hurry"
            : "drag to orbit · scroll to zoom"}
          {detail && ` · ${detail}`}
        </span>
      )}
    </div>
  );
}
