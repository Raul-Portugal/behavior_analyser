"""Synthetic TST clips shared by the engine, geometry and tuner tests.

A rigid ellipse "mouse" hangs from a pivot above it.  The clip contains a
static period, a rigid pendulum swing and a non-rigid limb wiggle, and can be
rendered at any frame rate or resolution from the same physical description.
"""

import cv2
import numpy as np

# (name, start_s, end_s) of each physical phase of the default clip.
PHASES = (
    ("static", 0.0, 1.5),
    ("swing", 1.5, 4.5),
    ("hold", 4.5, 5.5),
    ("wiggle", 5.5, 7.0),
    ("rest", 7.0, 8.0),
)
DURATION_S = 8.0
BASE_SIZE = (400, 300)  # width, height of the reference rendering


SHIFT = 4  # sub-pixel drawing so slow motion is not quantized to pixels
SUBPIXEL = 1 << SHIFT


def _fixed(point):
    return (
        int(round(float(point[0]) * SUBPIXEL)),
        int(round(float(point[1]) * SUBPIXEL)),
    )


def _background(size, seed=7):
    width, height = size
    rng = np.random.default_rng(seed)
    noise = rng.integers(-5, 6, (height, width, 1))
    gray = np.clip(215 + noise, 0, 255).astype(np.uint8)
    return np.repeat(gray, 3, axis=2)


def swing_angle_deg(t: float) -> float:
    """Pendulum angle: 4.5 degrees at 1 Hz between 1.5 s and 4.5 s."""
    if 1.5 <= t < 4.5:
        return 4.5 * np.sin(2.0 * np.pi * (t - 1.5))
    return 0.0


def limb_angle_deg(t: float) -> float:
    """Non-rigid limb angle: 2 Hz wiggle between 5.5 s and 7 s."""
    if 5.5 <= t < 7.0:
        return 45.0 * np.sin(2.0 * np.pi * 2.0 * (t - 5.5))
    return 0.0


def render_frame(t: float, background=None) -> np.ndarray:
    """Render the clip at time ``t`` seconds at the base resolution."""
    base = background if background is not None else _background(BASE_SIZE)
    image = base.copy()
    cx, cy = 200.0, 150.0
    pivot = np.array([cx, cy - 55.0])

    def place(point, angle_deg):
        radians = np.deg2rad(angle_deg)
        rotation = np.array([
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ])
        return pivot + rotation @ (np.asarray(point, dtype=float) - pivot)

    angle = swing_angle_deg(t)
    centre = place((cx, cy), angle)
    cv2.ellipse(
        image,
        _fixed(centre),
        (24 * SUBPIXEL, 42 * SUBPIXEL), angle, 0, 360, (35, 35, 35), -1,
        cv2.LINE_AA, SHIFT,
    )
    for dx, dy in [(-8, -30), (9, -22), (-12, -12), (10, -4), (-5, 6), (12, 14),
               (-9, 22), (6, 30), (0, 0), (-14, 2)]:
        spot = place((cx + dx, cy + dy), angle)
        cv2.circle(
            image, _fixed(spot), 3 * SUBPIXEL, (130, 130, 130), -1,
            cv2.LINE_AA, SHIFT,
        )
    limb = limb_angle_deg(t)
    if limb:
        radians = np.deg2rad(limb)
        start = np.array([cx + 20.0, cy])
        end = start + 38.0 * np.array([np.cos(radians), np.sin(radians)])
        cv2.line(
            image, _fixed(start), _fixed(end), (35, 35, 35), 8,
            cv2.LINE_AA, SHIFT,
        )
    return image


def clip_roi_points() -> np.ndarray:
    return np.array(
        [[100, 20], [300, 20], [300, 280], [100, 280]], dtype=np.int32
    )


def render_clip(fps: float, duration=DURATION_S):
    """Return a list of frames sampled at ``fps`` (frame i is at i / fps)."""
    background = _background(BASE_SIZE)
    count = int(round(duration * fps))
    return [render_frame(i / fps, background) for i in range(count)]
