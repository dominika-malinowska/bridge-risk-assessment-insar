from pathlib import Path


# Function to find the root of the repository by looking for a .git directory
# This function can be used to determine the base directory of the project
def find_repo_root(path=None):
    if path is None:
        path = Path.cwd()
    for parent in [path] + list(path.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("Could not find the repository root (no .git directory found)")
