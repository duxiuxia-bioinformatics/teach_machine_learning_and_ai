# project_imports.py

import sys
import os

def find_project_root(current_path=None, markers=("setup.py", "pyproject.toml", ".git")):
    """
    Walk upward until we find a folder containing one of the marker files.
    This folder is treated as the project root.
    """
    if current_path is None:
        # Try __file__ (works in .py scripts)
        try:
            current_path = os.path.abspath(os.path.dirname(__file__))
        except NameError:
            # Fallback for Jupyter notebooks
            current_path = os.getcwd()

    path = current_path
    while path != os.path.dirname(path):  # stop at filesystem root
        if any(os.path.exists(os.path.join(path, marker)) for marker in markers):
            return path
        path = os.path.dirname(path)
    return current_path  # fallback: current directory

# Detect project root
project_root = find_project_root()

# Insert into sys.path if not already present
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Optional: expose project_root for use elsewhere
__all__ = ["project_root"]

# Optional: debug print
# print(f"[project_imports] Project root set to: {project_root}")
