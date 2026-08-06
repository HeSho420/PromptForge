import { api } from "../api";
import { revealOnLoad } from "./parts";
import { Viewer3D } from "./Viewer3D";
import type { Asset } from "../types";

/**
 * One component that can display anything the pipeline produces.
 *
 * Results used to be rendered ad-hoc per page: Forge put everything in an
 * <img>, Avatar rendered even its VIDEO result through an <img>, and a mesh
 * had nowhere to go at all. Kind is on the asset — so switch on it once,
 * here, and every surface gets it right.
 */
export function ResultView({
  asset,
  kind,
  url,
  height = 460,
  mode = "orbit",
}: {
  /** Either pass the asset… */
  asset?: Asset | null;
  /** …or the kind + url directly, when only an id is in hand. */
  kind?: string;
  url?: string;
  height?: number;
  mode?: "orbit" | "walk";
}) {
  const resolved = kind ?? asset?.kind ?? "image";
  const src = url ?? (asset ? api.assetFileUrl(asset.id) : "");
  if (!src) return null;

  if (resolved === "model") {
    return (
      <div className="stack" style={{ gap: 8 }}>
        <Viewer3D url={src} mode={mode} height={height} />
        <div className="row" style={{ gap: 8 }}>
          <a className="btn ghost small" href={src} download>
            Download GLB
          </a>
          <span className="dim" style={{ fontSize: 11.5, alignSelf: "center" }}>
            opens in Blender, Unity, Unreal, or any 3D tool
          </span>
        </div>
      </div>
    );
  }

  if (resolved === "video") {
    return (
      <video
        className="result-video"
        src={src}
        controls
        loop
        autoPlay
        muted
        playsInline
      />
    );
  }

  return <img ref={revealOnLoad} src={src} alt="result" />;
}
