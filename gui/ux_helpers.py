"""
gui/ux_helpers.py
Small shared pieces of dialog behaviour: a "discard your work?" confirmation,
a button-per-choice question box, and the pure decision helpers behind them.

The Qt-free helpers hold the decisions (which options to offer, what a choice
means) so they can be tested without opening any dialog.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from PyQt6.QtWidgets import QMessageBox, QWidget


# ----------------------------------------------------------------------
# Question boxes
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Choice:
    key: str
    label: str
    role: QMessageBox.ButtonRole = QMessageBox.ButtonRole.AcceptRole


def ask_choice(
    parent: Optional[QWidget],
    title: str,
    text: str,
    choices: Sequence[Choice],
    default_key: Optional[str] = None,
    informative: str = "",
    icon: QMessageBox.Icon = QMessageBox.Icon.Question,
) -> Optional[str]:
    """Show a box with one real button per choice.

    Returns the chosen key, or the key of the RejectRole choice when the box
    is dismissed with Esc / the window close button, or None if there is no
    reject choice.
    """
    box = QMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    box.setText(text)
    if informative:
        box.setInformativeText(informative)
    buttons = {}
    for choice in choices:
        buttons[choice.key] = box.addButton(choice.label, choice.role)
    if default_key in buttons:
        box.setDefaultButton(buttons[default_key])
    reject_keys = [
        choice.key for choice in choices
        if choice.role == QMessageBox.ButtonRole.RejectRole
    ]
    if reject_keys:
        box.setEscapeButton(buttons[reject_keys[0]])
    box.exec()
    clicked = box.clickedButton()
    for key, button in buttons.items():
        if clicked is button:
            return key
    return reject_keys[0] if reject_keys else None


def confirm_discard(
    parent: Optional[QWidget],
    what: str,
    consequence: str = "",
) -> bool:
    """Ask before a dialog throws away the user's work; True means discard."""
    answer = ask_choice(
        parent,
        "Discard changes?",
        f"Discard {what}?",
        [
            Choice("discard", "Discard", QMessageBox.ButtonRole.DestructiveRole),
            Choice("keep", "Keep editing", QMessageBox.ButtonRole.RejectRole),
        ],
        default_key="keep",
        informative=consequence,
        icon=QMessageBox.Icon.Warning,
    )
    return answer == "discard"


# ----------------------------------------------------------------------
# Pure decisions
# ----------------------------------------------------------------------
def should_confirm_discard(initial_state, current_state) -> bool:
    """A dialog only needs a confirmation if its content changed."""
    return initial_state != current_state


def has_drawn_work(roi_manager, pending_points: int) -> bool:
    """True if the ROI selector holds any finished polygon or points in progress."""
    drawn = any(roi_manager.rois.get(category) for category in roi_manager.rois)
    return bool(drawn or pending_points > 0)


VERIFY_CONTINUE = "continue"
VERIFY_RUN = "run"
VERIFY_SAVE = "save"
VERIFY_DISCARD = "discard"


def verification_cancel_choices(accepted: int) -> List[Choice]:
    """Options offered when per-video verification is cancelled.

    With nothing accepted yet there is nothing to keep, so no options.
    """
    if accepted <= 0:
        return []
    noun = "video" if accepted == 1 else "videos"
    return [
        Choice(VERIFY_RUN, f"Run the {accepted} verified {noun}",
               QMessageBox.ButtonRole.AcceptRole),
        Choice(VERIFY_SAVE, "Save partial batch plan…",
               QMessageBox.ButtonRole.ActionRole),
        Choice(VERIFY_DISCARD, "Discard",
               QMessageBox.ButtonRole.DestructiveRole),
        Choice(VERIFY_CONTINUE, "Keep verifying",
               QMessageBox.ButtonRole.RejectRole),
    ]


@dataclass
class PartialVerificationOutcome:
    """What to do with the videos accepted before verification was cancelled."""
    run_plan: Dict  # plan to run now (empty: do not run)
    save_plan: bool  # ask where to save the partial plan
    keep_verifying: bool


def partial_verification_outcome(choice: Optional[str], accepted_plan: Dict
                                 ) -> PartialVerificationOutcome:
    if choice == VERIFY_RUN:
        return PartialVerificationOutcome(dict(accepted_plan), False, False)
    if choice == VERIFY_SAVE:
        return PartialVerificationOutcome({}, True, False)
    if choice == VERIFY_CONTINUE:
        return PartialVerificationOutcome({}, False, True)
    return PartialVerificationOutcome({}, False, False)
