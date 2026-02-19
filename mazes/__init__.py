"""
Central registry for all available maze configurations.
This is the single source of truth for the application.
"""
from mazes.y_maze import YMaze
from mazes.epm import EPM
from mazes.freestyle import Freestyle
from mazes.tst import TST
from mazes.base_maze import Maze

# The application will use this dictionary to populate the maze selection dialog.
# To add a new maze, simply create its class inheriting from Maze and add an instance here.
AVAILABLE_MAZES: dict[str, Maze] = {
    "Y-Maze": YMaze(),
    "Elevated Plus Maze": EPM(),
    "Freestyle / Open Field": Freestyle(),
    "Tail Suspension Test": TST(),
}