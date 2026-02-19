"""
Shared utility functions for the Y-Maze Analyzer application.
"""
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import List


def get_user_input(prompt: str, default: str = "", input_type=str):
    """Gets validated user input from the console."""
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input and default:
            user_input = default
        if not user_input:
            print("Input is required. Please try again.")
            continue
        try:
            return input_type(user_input)
        except ValueError:
            print(f"Invalid input. Expected type: {input_type.__name__}.")


def get_yes_no(prompt: str, default: bool = True) -> bool:
    """Gets a yes/no answer from the console."""
    default_str = "y" if default else "n"
    response = input(f"{prompt} (y/n) [{default_str}]: ").strip().lower()
    if not response:
        return default
    return response.startswith('y')


def select_multiple_videos() -> List[Path]:
    """Opens a file dialog to select multiple video files."""
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        file_paths = filedialog.askopenfilenames(
            title="Select Video Files for Batch Processing",
            filetypes=(("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*"))
        )
        root.destroy()
        return [Path(p) for p in file_paths] if file_paths else []
    except Exception as e:
        print(f"Error opening file dialog: {e}")
        return []