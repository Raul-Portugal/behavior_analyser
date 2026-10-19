"""
Central registry for all available maze configurations.
This is the single source of truth for the application.
"""
from mazes.y_maze import YMaze
from mazes.epm import EPM
from mazes.freestyle import Freestyle
from mazes.tst import TST
from mazes.base_maze import Maze
from typing import Callable

# The application will use this dictionary to populate the maze selection dialog.
# To add a new maze, simply create its class inheriting from Maze and add an instance here.
AVAILABLE_MAZES: dict[str, Callable[[], Maze]] = {
    "Y-Maze": YMaze,
    "Elevated Plus Maze": EPM,
    "Freestyle / Open Field": Freestyle,
    "Tail Suspension Test": TST,
}


def create_maze(name: str) -> Maze:
    """Create an isolated maze instance for one configuration or run."""
    try:
        return AVAILABLE_MAZES[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown maze type: {name}") from exc


def maze_class_for_settings(settings) -> Callable[[], Maze]:
    """The maze class that saved ``BatchSettings`` belong to.

    A ``paradigm_id`` naming a known paradigm decides. Otherwise (files saved
    before the id existed, or an unknown id) the legacy ROI-name heuristics
    are tried in registry order; Freestyle, which matches anything, is last.
    """
    paradigm_id = getattr(settings, 'paradigm_id', None)
    if paradigm_id in AVAILABLE_MAZES:
        return AVAILABLE_MAZES[paradigm_id]
    names = list(settings.roi_manager.rois.keys())
    for maze_class in AVAILABLE_MAZES.values():
        if maze_class.matches_roi_names(names):
            return maze_class
    raise ValueError("Could not infer the paradigm of these settings")


def infer_maze_from_settings(settings) -> Maze:
    """Create and configure the maze that saved settings were made for."""
    maze = maze_class_for_settings(settings)()
    maze.infer_from_settings(settings)
    return maze


def paradigm_analysis_kind(paradigm_name) -> str:
    """``analysis_kind`` of a paradigm by name ("tracking" if unknown)."""
    maze_class = AVAILABLE_MAZES.get(paradigm_name)
    return maze_class().analysis_kind if maze_class else "tracking"
