"""Shared colour and label mapping for TST state overlays.

The tuner preview and the validation video draw the same states, so they
share one mapping instead of keeping two copies in sync by hand.
"""

from typing import Optional, Tuple

IMMOBILE_STATES = frozenset({"immobile_static", "passive_swing"})
KNOWN_STATES = IMMOBILE_STATES | {"active_escape"}

# BGR colours (OpenCV order) and overlay labels.
UNCERTAIN_STYLE = ((0, 165, 255), "UNCERTAIN")
PASSIVE_SWING_STYLE = ((180, 0, 180), "PASSIVE SWING")
IMMOBILE_STATIC_STYLE = ((0, 0, 255), "IMMOBILE STATIC")
SHORT_PAUSE_STYLE = ((0, 215, 255), "SHORT PAUSE")
ACTIVE_ESCAPE_STYLE = ((0, 255, 0), "ACTIVE ESCAPE")


def tst_display_style(
    behavior_state: str,
    is_immobile: Optional[bool],
) -> Tuple[Tuple[int, int, int], str]:
    """Return the (BGR colour, label) for one frame.

    ``is_immobile`` is the temporally filtered immobility flag: ``True`` for
    a scored immobile bout, ``False`` for mobile, ``None`` when unscorable. A
    frame whose raw state is immobile but that did not survive the minimum
    bout duration is drawn as a short pause rather than as an escape.
    """
    if behavior_state not in KNOWN_STATES or is_immobile is None:
        return UNCERTAIN_STYLE
    if is_immobile:
        if behavior_state == "passive_swing":
            return PASSIVE_SWING_STYLE
        return IMMOBILE_STATIC_STYLE
    if behavior_state in IMMOBILE_STATES:
        return SHORT_PAUSE_STYLE
    return ACTIVE_ESCAPE_STYLE
