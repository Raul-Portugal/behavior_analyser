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
