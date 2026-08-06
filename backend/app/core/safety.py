"""Safety policy — the single home for ALL content-safety rules.

Per project decision, the general adult-content filter and the "adult mode"
toggle have been REMOVED: ordinary generation (including nudity/NSFW) passes
straight through. A fuller, configurable content policy will be reintroduced
later in the project.

What remains hardcoded and non-negotiable — because this app edits real
photos of real people and can place a real face into any scene — are the
three categories that separate an image editor from a tool for abuse:

  - sexual/appearance content involving minors,
  - undressing/exposure edits of an existing photo of a person (NCII),
  - deepfake-style identity manipulation / impersonation of a real person.

Every filter still lives in THIS module (SafetyFilter.check plus the
consent_verdict and model_source_blocked helpers); other modules call these
rather than embedding their own rules, so the whole policy is auditable here.

Matching runs on a normalized string (lowercased, punctuation stripped,
separators between letters removed) to defeat trivial obfuscation like
"n.u.d.e", and rules are word-boundary regexes to avoid false positives such
as "grass" matching "ass".
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

from .db import Database


@dataclass(frozen=True)
class SafetyVerdict:
    allowed: bool
    category: str | None = None
    reason: str | None = None
    matched: str | None = None  # kept out of user-facing UI; useful in logs


def _normalize(text: str) -> str:
    text = text.lower()
    # remove separators bad actors use to split words: n.u.d.e / n-u-d-e
    text = re.sub(r"(?<=\w)[.\-_*]+(?=\w)", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _rx(words: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(words) + r")\b")


# Sexual/adult wording — NO LONGER a block on its own. Retained only so the
# panini + sexual combination can be detected below.
_ADULT = _rx([
    "nude", "nudes", "naked", "nsfw", "porn", "pornographic", "topless",
    "lingerie", "explicit", "erotic", "sexual(?:ized|ise|ized)?", "sexy",
    "bikini body", "cleavage", "genitals?", "breasts?", "nipples?",
])

# Undressing an existing photo of a real person = NCII. Always blocked on edits.
_EXPOSURE = _rx([
    "undress(?:ed|ing)?", "strip(?:ped|ping)? (?:her|him|them|off)",
    "see through clothes", "x ray clothes",
    "remove (?:her|his|their|the) (?:clothes|clothing|top|shirt|dress|bra)",
])

# Words that mean the request involves a minor. "minor" alone is matched only
# as a noun ("a minor", "minors") — the adjective ("minor adjustments") is
# ordinary editing language.
_MINORS = _rx([
    "child", "children", "childs", "kid", "kids", "minors", "a minor",
    "teen", "teens", "teenager", "teenagers", "underage", "preteen",
    "preteens", "tween", "tweens", "toddler", "toddlers", "baby", "babies",
    "infant", "infants", "schoolgirl", "schoolgirls", "schoolboy",
    "schoolboys",
])

# Identity manipulation of a real person: face swaps, deepfakes,
# impersonation. This app edits photographs of real, identifiable people, so
# these are blocked outright rather than gated.
_DEEPFAKE = _rx([
    "deep ?fakes?", "deepfaked?", "deepfaking",
    "face ?swap(?:s|ped|ping)?", "swap (?:her|his|their|my|our|the) faces?",
    "swap faces?", "put (?:my|her|his|their|someones?) face on(?:to)?",
    "replace (?:her|his|their|my|the) face",
    "impersonat(?:e|es|ed|ing|ion)", "pretend to be (?:her|him|them)",
    "pass (?:off )?as (?:her|him|them)",
])

_NONCONSENSUAL = _rx([
    "without (?:her|his|their) consent", "non consensual", "nonconsensual",
    "revenge", "secretly (?:photograph|record|edit)ed?",
    "hidden camera", "spy (?:cam|camera|photo)",
])


# The built-in, non-removable protections, in the order check() enforces them.
# Shown (read-only) in the Settings UI so users can see what can't be deleted.
BUILTIN_RULES: list[tuple[str, str]] = [
    ("minors", "Sexual or appearance content involving minors."),
    ("exposure", "Undressing / exposure edits of an existing photo of a "
                 "person (non-consensual intimate imagery)."),
    ("deepfake", "Deepfake-style identity manipulation or impersonation of "
                 "a real person."),
    ("nonconsensual", "Editing people without their consent."),
]

# Categories a user rule may never reuse (would confuse the locked built-ins).
_RESERVED_CATEGORIES = {"minors", "exposure", "deepfake", "nonconsensual",
                        "empty", "consent", "sexual"}


class SafetyRuleError(ValueError):
    """A user-supplied safety rule was rejected (bad pattern / reserved name)."""


def _compile_rule(pattern: str) -> re.Pattern[str]:
    """Compile a user rule. A plain keyword/phrase is auto-wrapped in word
    boundaries (so 'gun' doesn't match 'begun'); anything with regex
    metacharacters is used verbatim. Matching is case-insensitive and runs on
    the same normalized text the built-ins use."""
    src = pattern.strip()
    if re.fullmatch(r"[\w ]+", src):
        src = rf"\b(?:{re.escape(src)})\b"
    return re.compile(src, re.IGNORECASE)


class SafetyRuleStore:
    """The single source of truth for USER-DEFINED block rules — a DB table.

    Design guarantees the caller asked for:
      * Built-in protections are NOT stored here (they live in code), so the
        add/delete surface can never touch them.
      * compiled() re-reads the table on every call and the filter calls it on
        every check — the rules are never held in an in-memory cache, so an
        add or delete takes effect on the very next prompt.
    """

    MAX_PATTERN = 200

    def __init__(self, db: Database):
        self._db = db
        db.execute("""
            CREATE TABLE IF NOT EXISTS safety_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now')),
                category TEXT NOT NULL,
                pattern TEXT NOT NULL,
                reason TEXT NOT NULL
            )""")

    def list(self) -> list[dict]:
        return [{"id": r["id"], "category": r["category"],
                 "pattern": r["pattern"], "reason": r["reason"],
                 "created_at": r["created_at"]}
                for r in self._db.query(
                    "SELECT id, category, pattern, reason, created_at "
                    "FROM safety_rules ORDER BY id")]

    def add(self, pattern: str, reason: str = "",
            category: str = "custom") -> dict:
        pattern = (pattern or "").strip()
        reason = (reason or "").strip()
        category = ((category or "custom").strip().lower() or "custom")
        if not pattern:
            raise SafetyRuleError("Rule pattern must not be empty.")
        if len(pattern) > self.MAX_PATTERN:
            raise SafetyRuleError(
                f"Pattern is too long (max {self.MAX_PATTERN} characters).")
        if category in _RESERVED_CATEGORIES:
            raise SafetyRuleError(
                f"'{category}' is a built-in category and cannot be reused — "
                "pick another label (e.g. 'custom').")
        try:
            _compile_rule(pattern)
        except re.error as exc:
            raise SafetyRuleError(f"Not a valid pattern: {exc}") from exc
        if not reason:
            reason = f"Blocked by a custom rule ({category})."
        self._db.execute(
            "INSERT INTO safety_rules (category, pattern, reason) VALUES (?,?,?)",
            (category, pattern, reason))
        r = self._db.query(
            "SELECT id, category, pattern, reason, created_at FROM "
            "safety_rules ORDER BY id DESC LIMIT 1")[0]
        return {"id": r["id"], "category": r["category"], "pattern": r["pattern"],
                "reason": r["reason"], "created_at": r["created_at"]}

    def delete(self, rule_id: int) -> bool:
        if not self._db.query("SELECT id FROM safety_rules WHERE id=?", (rule_id,)):
            return False
        self._db.execute("DELETE FROM safety_rules WHERE id=?", (rule_id,))
        return True

    def compiled(self) -> list[tuple[str, re.Pattern[str], str]]:
        """Live-compiled rules for the filter. Never memoized."""
        out: list[tuple[str, re.Pattern[str], str]] = []
        for r in self._db.query(
                "SELECT category, pattern, reason FROM safety_rules ORDER BY id"):
            try:
                out.append((r["category"], _compile_rule(r["pattern"]),
                            r["reason"]))
            except re.error:
                continue  # a corrupt row must never disable the filter
        return out


@dataclass
class SafetyFilter:
    """Screens prompts. The general adult filter and adult mode are gone; the
    minors / photo-exposure / deepfake / non-consensual rules stay enforced.

    extra_rules  — in-process additions (category, compiled pattern, reason).
    custom_provider — a callable returning the user's DB-backed rules; called
        LIVE on every check so edits are never served from a stale cache."""

    extra_rules: list[tuple[str, re.Pattern[str], str]] = field(default_factory=list)
    custom_provider: Callable[[], list[tuple[str, re.Pattern[str], str]]] | None = None

    @staticmethod
    def builtin_summary() -> list[dict]:
        return [{"category": c, "description": d, "locked": True}
                for c, d in BUILTIN_RULES]

    def check(self, prompt: str, editing: bool = True) -> SafetyVerdict:
        """Screen a prompt.

        editing — True when a source photo is being modified (enables the
            exposure/undressing block, which only makes sense against a real
            existing image).

        General adult content is allowed. Blocked in every case: minors,
        non-consensual imagery, undressing/exposure edits of a photo, and
        deepfake-style impersonation of a real person.
        """
        text = _normalize(prompt or "")
        if not text:
            return SafetyVerdict(False, "empty", "Enter a prompt describing the edit.")

        # Sexual/appearance content involving minors — always blocked.
        if (_ADULT.search(text) or _EXPOSURE.search(text)) and _MINORS.search(text):
            return SafetyVerdict(False, "minors", "This request is not allowed.")

        rules: list[tuple[str, re.Pattern[str], str]] = [
            ("deepfake", _DEEPFAKE,
             "Identity manipulation (face swaps, impersonation of a real "
             "person) is not supported."),
            ("nonconsensual", _NONCONSENSUAL,
             "Edits involving people without their consent are not supported."),
            *self.extra_rules,
        ]
        if editing:
            rules.insert(0, ("exposure", _EXPOSURE,
                             "Exposure/undressing edits of an existing photo "
                             "are not supported."))
        # User-defined rules, read LIVE from the store (never cached). A broken
        # provider must never take the built-in protections offline.
        if self.custom_provider is not None:
            try:
                rules.extend(self.custom_provider())
            except Exception:  # noqa: BLE001 — defensive; built-ins still apply
                pass
        for category, pattern, reason in rules:
            m = pattern.search(text)
            if m:
                return SafetyVerdict(False, category, reason, m.group(0))

        # A minor + anything that reads as body/appearance editing gets blocked
        if _MINORS.search(text) and re.search(
                r"\b(body|face|skin|clothe?s?|dress|outfit)\b", text):
            return SafetyVerdict(
                False, "minors",
                "Appearance edits of minors are not supported.")

        return SafetyVerdict(True)


def consent_verdict(consent: bool) -> SafetyVerdict:
    """Avatar datasets require an explicit consent attestation — building a
    digital human of someone without their consent is never supported."""
    if consent:
        return SafetyVerdict(True)
    return SafetyVerdict(
        False, "consent",
        "Avatar datasets require explicit consent of the person depicted. "
        "Tick the consent attestation and try again.")


def model_source_blocked(nsfw_flagged: bool) -> bool:
    """Model-download content policy: models their own source flags as NSFW
    are not auto-installed. (Download-time policy, not a prompt filter.)"""
    return bool(nsfw_flagged)
