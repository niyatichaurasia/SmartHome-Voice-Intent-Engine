"""
utils/intents.py
Intent label construction and the full label list for Fluent Speech Commands.

Each intent = action × object × location, e.g. "activate_lights_bedroom".
"""


def build_intent_label(action: str, obj: str, location: str) -> str:
    """Combine FSC fields into a single intent string."""
    parts = [action.strip().lower().replace(" ", "_")]
    if obj and obj.lower() not in ("", "none"):
        parts.append(obj.strip().lower().replace(" ", "_"))
    if location and location.lower() not in ("", "none"):
        parts.append(location.strip().lower().replace(" ", "_"))
    return "__".join(parts)


# ── Full list of 31 FSC intent classes ──────────────────────────────────────
# Generated from the unique (action, object, location) combinations in FSC.
INTENT_LIST = [
    "activate__lights__none",
    "activate__lights__bedroom",
    "activate__lights__kitchen",
    "activate__lights__washroom",
    "activate__music__none",
    "activate__lamp__none",
    "deactivate__lights__none",
    "deactivate__lights__bedroom",
    "deactivate__lights__kitchen",
    "deactivate__lights__washroom",
    "deactivate__music__none",
    "deactivate__lamp__none",
    "increase__volume__none",
    "increase__heat__none",
    "increase__heat__bedroom",
    "increase__heat__kitchen",
    "decrease__volume__none",
    "decrease__heat__none",
    "decrease__heat__bedroom",
    "decrease__heat__kitchen",
    "bring__newspaper__none",
    "bring__shoes__none",
    "bring__juice__none",
    "bring__socks__none",
    "change language__Chinese__none",
    "change language__English__none",
    "change language__German__none",
    "change language__Korean__none",
    "change language__none__none",
    "get__weather__none",
    "get__time__none",
]

# Friendly display names (shown in the UI)
INTENT_DISPLAY = {i: i.replace("__", " → ").replace("_", " ").title() for i in INTENT_LIST}