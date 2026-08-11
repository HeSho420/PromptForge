import { useRef, useState, type ReactNode } from "react";
import { api } from "../api";
import { QueueDock } from "../components/QueueDock";
import { Avatar } from "./Avatar";
import { Forge } from "./Forge";
import { Motion } from "./Motion";
import { Studio } from "./Studio";
import type { Asset } from "../types";

/**
 * One Studio.
 *
 * Editing, forging, motion and avatars were four pages doing the same thing —
 * attach something, describe what you want, watch a job run. They differed
 * only in WHAT you attach, so that is what picks the mode now: drop a photo
 * and you are editing, drop a video and you are transferring motion, drop
 * several photos of a person and you are building an avatar. The chips let
 * you override whenever the guess is wrong.
 *
 * Every mode stays MOUNTED and is hidden with `display`, exactly as App.tsx
 * does for pages. That is not laziness: each mode owns a running job poll,
 * and nothing in the app can re-attach to a job by id, so unmounting a mode
 * with work in flight would silently lose it.
 */

export type Mode = "edit" | "forge" | "motion" | "avatar";

const MODES: { key: Mode; label: string; hint: string; icon: string }[] = [
  {
    key: "edit",
    label: "Edit a photo",
    hint: "Change anything in a picture you already have",
    icon: "M12 20h9 M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z",
  },
  {
    key: "forge",
    label: "Make something new",
    hint: "Generate from a description, upscale, or animate",
    icon: "M15 12l-8.5 8.5a2.1 2.1 0 0 1-3-3L12 9 M17.6 11.6 22 7l-3-3-4.6 4.4 M9 6 6.5 3.5 3 4l2.5 2.5",
  },
  {
    key: "motion",
    label: "Copy a motion",
    hint: "Put a person from a photo into a video's movement",
    icon: "M4 5h11v14H4z M18 8l3-2v12l-3-2 M8 9l3 3-3 3",
  },
  {
    key: "avatar",
    label: "Build an avatar",
    hint: "Several photos of one person, from every side",
    icon: "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8 M4 21c0-4 3.6-6.5 8-6.5s8 2.5 8 6.5",
  },
];

const icon = (d: string) => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
       strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d={d} />
  </svg>
);

export function Workspace() {
  const [mode, setMode] = useState<Mode>("edit");
  // Set when the shell accepts a file on a mode's behalf; the mode adopts it
  // and calls onConsumed so a later re-render cannot re-adopt the same asset.
  const [handoff, setHandoff] = useState<{ mode: Mode; assets: Asset[] } | null>(null);
  const [dropping, setDropping] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Modes that have started at least one job — used only to keep the chip
  // labelled while work runs in a mode you are not looking at.
  const [active, setActive] = useState<Partial<Record<Mode, boolean>>>({});
  const inputRef = useRef<HTMLInputElement | null>(null);

  const accept = async (files: FileList | null) => {
    const list = Array.from(files ?? []);
    if (!list.length) return;
    setError(null);
    setBusy(true);
    try {
      // Upload everything that was dropped, in order, then decide from the
      // SET what it means. One import point, any number of files.
      const assets: Asset[] = [];
      const failed: string[] = [];
      for (const file of list) {
        try {
          assets.push(await api.uploadAsset(file));
        } catch (e) {
          // One rejected file must not discard the ones that already
          // uploaded — they are on disk either way.
          failed.push(`${file.name}: ${(e as Error).message}`);
        }
      }
      if (failed.length) setError(failed.join("; "));
      if (!assets.length) return;
      const videos = assets.filter((a) => a.kind === "video");
      const images = assets.filter((a) => a.kind === "image");

      let target: Mode;
      if (videos.length && images.length < 3) {
        // A video means motion — unless the drop is mostly a photo set with
        // one clip in it, in which case the photos are the point and routing
        // to motion would quietly ignore all but one of them.
        target = "motion";
      } else if (images.length >= 3) {
        // Three or more photos is a person's dataset, not an edit.
        target = "avatar";
        if (videos.length) {
          setError(
            `Ignoring ${videos.length} video file(s) — a photo set builds an ` +
            "avatar. Drop the clip on its own to copy a motion.");
        }
      } else {
        // One photo edits it; two makes the second a reference to take a
        // subject or a face FROM — the backend decides which from the words.
        target = "edit";
      }
      setMode(target);
      setHandoff({ mode: target, assets });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const current = MODES.find((m) => m.key === mode) ?? MODES[0];

  return (
    <div className="workspace">
      <header className="page-head">
        <h1>Studio</h1>
        <p className="sub">
          Attach something and say what you want. {current.hint}.
        </p>
      </header>

      {/* Avatar builds from a SET and has its own multi-file drop that shows
          the thumbnails and the running count — a second, dumber drop zone
          above it just duplicates the job. */}
      <div
        className={`ws-drop${dropping ? " over" : ""}${
          mode === "avatar" ? " is-hidden" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDropping(true);
        }}
        onDragLeave={() => setDropping(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDropping(false);
          void accept(e.dataTransfer.files);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*,video/mp4,video/quicktime,video/webm,.mkv"
          multiple
          hidden
          onChange={(e) => {
            void accept(e.target.files);
            e.target.value = "";
          }}
        />
        <button
          type="button"
          className="ws-drop-btn"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
        >
          {busy ? "Uploading…" : "Drop images or a video here, or click to choose"}
        </button>
        <span className="ws-drop-hint">
          1 photo edits it · 2 combines them · 3+ of one person builds an avatar · a video becomes a motion to copy
        </span>
      </div>

      {/* A mode switcher, styled as a segmented control and operated as a
          toggle-button group: aria-pressed states which mode is active, and
          every mode stays Tab-reachable. A full ARIA tab widget was the
          other option, but its roving tabindex takes the non-active modes
          out of the Tab order — the wrong trade for primary navigation, and
          the panels were never wired up as tabpanels to begin with. */}
      {/* The queue, always in reach: what runs, what waits, and a one-line
          way to cue up the next task without leaving this page. */}
      <QueueDock />

      <nav className="ws-modes" role="group" aria-label="What to do">
        {MODES.map((m) => (
          <button
            key={m.key}
            type="button"
            aria-pressed={mode === m.key}
            className={`ws-mode${mode === m.key ? " active" : ""}`}
            onClick={() => setMode(m.key)}
            title={m.hint}
          >
            {icon(m.icon)}
            <span>{m.label}</span>
            {active[m.key] && mode !== m.key && (
              <span className="ws-dot" title="a job is running here" />
            )}
          </button>
        ))}
      </nav>

      {error && <div className="notice">{error}</div>}

      <ModePanel show={mode === "edit"}>
        <Studio
          incoming={handoff?.mode === "edit" ? handoff.assets : null}
          onConsumed={() => setHandoff(null)}
          onBusy={(b) => setActive((s) => ({ ...s, edit: b }))}
        />
      </ModePanel>
      <ModePanel show={mode === "forge"}>
        <Forge onBusy={(b) => setActive((s) => ({ ...s, forge: b }))} />
      </ModePanel>
      <ModePanel show={mode === "motion"}>
        <Motion
          incoming={handoff?.mode === "motion" ? handoff.assets : null}
          onConsumed={() => setHandoff(null)}
          onBusy={(b) => setActive((s) => ({ ...s, motion: b }))}
        />
      </ModePanel>
      <ModePanel show={mode === "avatar"}>
        <Avatar
          incoming={handoff?.mode === "avatar" ? handoff.assets : null}
          onConsumed={() => setHandoff(null)}
          onBusy={(b) => setActive((s) => ({ ...s, avatar: b }))}
        />
      </ModePanel>
    </div>
  );
}

/** Hidden, never unmounted — see the note at the top of this file. */
function ModePanel({ show, children }: { show: boolean; children: ReactNode }) {
  return <div className={`ws-panel${show ? "" : " is-hidden"}`}>{children}</div>;
}
