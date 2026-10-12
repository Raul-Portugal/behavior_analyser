"""Synthetic arena images and known camera moves for alignment tests."""
from typing import Tuple

import cv2
import numpy as np

TEMPLATE_DIMS = (640, 480)


def arena_image(dims: Tuple[int, int] = TEMPLATE_DIMS, seed: int = 7) -> np.ndarray:
    """A textured top-down 'arena': a plus-shaped maze on a speckled floor.

    The texture is random but seeded so features are reproducible and rich
    enough for ORB, while every shape stays inside the frame.
    """
    rng = np.random.default_rng(seed)
    width, height = dims
    image = np.full((height, width, 3), 150, dtype=np.uint8)
    noise = rng.normal(0, 18, (height, width, 1))
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (0, 0), 1.2)
    for _ in range(140):
        centre = (int(rng.integers(40, width - 40)), int(rng.integers(40, height - 40)))
        size = int(rng.integers(5, 22))
        colour = tuple(int(c) for c in rng.integers(30, 230, 3))
        if rng.random() < 0.5:
            cv2.circle(image, centre, size, colour, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(
                image, (centre[0] - size, centre[1] - size // 2),
                (centre[0] + size, centre[1] + size // 2), colour, -1, cv2.LINE_AA,
            )
    cx, cy = width // 2, height // 2
    cv2.rectangle(image, (cx - 40, 60), (cx + 40, height - 60), (40, 40, 40), 4, cv2.LINE_AA)
    cv2.rectangle(image, (80, cy - 40), (width - 80, cy + 40), (40, 40, 40), 4, cv2.LINE_AA)
    return image


def camera_move(
    shift=(0.0, 0.0), rotation_deg=0.0, scale=1.0, dims=TEMPLATE_DIMS,
) -> np.ndarray:
    """3x3 matrix mapping template px to target px for a similarity move
    about the template centre, followed by a pixel shift."""
    centre = (dims[0] / 2.0, dims[1] / 2.0)
    rot = cv2.getRotationMatrix2D(centre, rotation_deg, scale)
    matrix = np.vstack([rot, [0.0, 0.0, 1.0]])
    matrix[0, 2] += shift[0]
    matrix[1, 2] += shift[1]
    return matrix


def rescale(matrix: np.ndarray, factor: float) -> np.ndarray:
    """Compose ``matrix`` with a uniform target-resolution change."""
    return np.diag([factor, factor, 1.0]) @ matrix


def warp(image: np.ndarray, matrix: np.ndarray, dims: Tuple[int, int]) -> np.ndarray:
    """The image as a camera with the given template->target map would see it."""
    return cv2.warpAffine(
        image, matrix[:2], dims, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def max_grid_error(estimated: np.ndarray, truth: np.ndarray,
                   dims: Tuple[int, int] = TEMPLATE_DIMS) -> float:
    """Largest distance between two transforms over a grid inside the frame."""
    xs = np.linspace(40, dims[0] - 40, 6)
    ys = np.linspace(40, dims[1] - 40, 5)
    grid = np.array([[x, y] for x in xs for y in ys], dtype=np.float64)

    def apply(m):
        homogeneous = np.hstack([grid, np.ones((len(grid), 1))]) @ m.T
        return homogeneous[:, :2] / homogeneous[:, 2:3]

    return float(np.max(np.linalg.norm(apply(estimated) - apply(truth), axis=1)))
