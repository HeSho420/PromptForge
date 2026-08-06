"""Workflow memory — the program learns from its own renders.

Every workflow run is recorded: the graph that ran, whether it succeeded, the
realism score the critic gave it, how many repairs it needed, and every error
ComfyUI reported along the way. Before planning a new workflow, the planner
is handed *lessons*:

  * the best-scoring past graph for the same task (as a proven example,
    weighted towards prompts that share words with the new one), and
  * recent distinct error messages for that task (known pitfalls to avoid).

This is deliberately plain SQL + keyword overlap — transparent, inspectable
(`workflow_memory` table), and cheap. An embedding index can replace the
ranking later without changing the interface.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .db import Database

_WORD = re.compile(r"[a-z]{3,}")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


class ExperienceStore:
    def __init__(self, db: Database):
        self._db = db
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS workflow_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now')),
                task TEXT NOT NULL,
                prompt TEXT NOT NULL,
                graph TEXT NOT NULL,
                success INTEGER NOT NULL,
                realism REAL,
                repairs INTEGER NOT NULL DEFAULT 0,
                errors TEXT NOT NULL DEFAULT '[]'
            )""")
        # Long-term repair knowledge: every time the LLM successfully fixes a
        # ComfyUI error, the error→fix pair is kept and replayed as a hint
        # whenever a similar error (or task) comes up — the planner gets
        # smarter with every mistake it survives.
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS repair_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now')),
                task TEXT NOT NULL,
                error TEXT NOT NULL,
                fix TEXT NOT NULL,
                uses INTEGER NOT NULL DEFAULT 0
            )""")

    # -- writing ------------------------------------------------------------------
    def record(self, task: str, prompt: str, graph: dict[str, Any] | None,
               success: bool, realism: float | None = None,
               repairs: int = 0, errors: list[str] | None = None) -> None:
        self._db.execute(
            "INSERT INTO workflow_memory (task, prompt, graph, success, "
            "realism, repairs, errors) VALUES (?,?,?,?,?,?,?)",
            (task, prompt[:2000], json.dumps(graph or {}), int(success),
             realism, repairs, json.dumps([e[:500] for e in (errors or [])][:10])))

    def scrub_prompts(self) -> int:
        """Blank the verbatim prompt text stored with every workflow memory
        row. The learned graphs, scores and error lessons stay usable (recall
        just loses its word-overlap ranking); the user's exact words are
        gone. Part of the user-facing 'delete prompt history' action."""
        rows = self._db.query(
            "SELECT COUNT(*) AS n FROM workflow_memory WHERE prompt != ''")
        n = int(rows[0]["n"]) if rows else 0
        self._db.execute("UPDATE workflow_memory SET prompt=''")
        return n

    def record_repair(self, task: str, error: str,
                      before: dict[str, Any], after: dict[str, Any]) -> None:
        """Distil a successful LLM repair into a reusable fix note."""
        fix = _describe_fix(before, after)
        if not fix:
            return
        # Same error prefix + same fix already known → bump its use count.
        rows = self._db.query(
            "SELECT id, fix FROM repair_knowledge WHERE task=? AND "
            "substr(error,1,80)=? LIMIT 5", (task, error[:80]))
        for r in rows:
            if r["fix"] == fix:
                self._db.execute(
                    "UPDATE repair_knowledge SET uses=uses+1 WHERE id=?",
                    (r["id"],))
                return
        self._db.execute(
            "INSERT INTO repair_knowledge (task, error, fix) VALUES (?,?,?)",
            (task, error[:500], fix[:500]))

    def repair_hints(self, task: str, limit: int = 3) -> list[str]:
        """Most-proven fixes for this task, best first."""
        rows = self._db.query(
            "SELECT error, fix, uses FROM repair_knowledge WHERE task=? "
            "ORDER BY uses DESC, id DESC LIMIT ?", (task, limit))
        return [f"When ComfyUI says \"{r['error'][:100]}\": {r['fix']}"
                for r in rows]

    # -- reading ------------------------------------------------------------------
    def best_example(self, task: str, prompt: str) -> dict[str, Any] | None:
        """Highest-quality past graph for this task, biased to similar prompts."""
        rows = self._db.query(
            "SELECT prompt, graph, realism, repairs FROM workflow_memory "
            "WHERE task=? AND success=1 ORDER BY id DESC LIMIT 50", (task,))
        if not rows:
            return None
        want = _words(prompt)

        def score(row: Any) -> float:
            overlap = len(want & _words(row["prompt"])) if want else 0
            realism = row["realism"] if row["realism"] is not None else 5.0
            return realism + overlap * 0.5 - row["repairs"] * 0.25

        best = max(rows, key=score)
        try:
            graph = json.loads(best["graph"])
        except json.JSONDecodeError:
            return None
        return graph if isinstance(graph, dict) and graph else None

    def known_pitfalls(self, task: str, limit: int = 4) -> list[str]:
        """Recent distinct error messages seen for this task."""
        rows = self._db.query(
            "SELECT errors FROM workflow_memory WHERE task=? "
            "ORDER BY id DESC LIMIT 30", (task,))
        seen: list[str] = []
        for row in rows:
            try:
                for err in json.loads(row["errors"]):
                    key = err[:120]
                    if key and all(key[:60] != s[:60] for s in seen):
                        seen.append(key)
                        if len(seen) >= limit:
                            return seen
            except json.JSONDecodeError:
                continue
        return seen

    def lessons(self, task: str, prompt: str) -> str | None:
        """Context block for the planner; None when there is nothing useful."""
        parts: list[str] = []
        example = self.best_example(task, prompt)
        if example:
            parts.append("A past workflow that produced a good result for a "
                         "similar request (adapt it, do not copy blindly):\n"
                         + json.dumps(example))
        pitfalls = self.known_pitfalls(task)
        if pitfalls:
            parts.append("Errors seen before on this task — avoid causing "
                         "these again:\n- " + "\n- ".join(pitfalls))
        hints = self.repair_hints(task)
        if hints:
            parts.append("Proven fixes from past repairs:\n- "
                         + "\n- ".join(hints))
        return "\n".join(parts) if parts else None


def _describe_fix(before: dict[str, Any], after: dict[str, Any]) -> str | None:
    """Human/LLM-readable one-liner of what changed between two graphs.

    Nodes are compared BY ID (and only when the class_type matches) — pairing
    by type would diff the positive prompt node against the negative one and
    poison the knowledge base with fabricated "fixes"."""
    changes: list[str] = []
    b_types = sorted(n.get("class_type", "?") for n in before.values())
    a_types = sorted(n.get("class_type", "?") for n in after.values())
    if b_types != a_types:
        added = [t for t in a_types if t not in b_types]
        removed = [t for t in b_types if t not in a_types]
        if added:
            changes.append("added " + ", ".join(dict.fromkeys(added)))
        if removed:
            changes.append("removed " + ", ".join(dict.fromkeys(removed)))
    for nid, n in after.items():
        old = before.get(nid)
        if not old or old.get("class_type") != n.get("class_type"):
            continue
        for key, val in n.get("inputs", {}).items():
            ov = old.get("inputs", {}).get(key)
            if ov is None or ov == val:
                continue
            if isinstance(val, list) or isinstance(ov, list):
                changes.append(f"{n['class_type']}.{key} re-linked")
            else:
                changes.append(f"{n['class_type']}.{key}: {ov!r} → {val!r}")
    if not changes:
        return None
    return "; ".join(changes[:6])
