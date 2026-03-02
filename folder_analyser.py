
from pathlib import Path
from typing import Union
from collections.abc import Iterable

def print_directory_tree(
        start_path: Union[str, Path],
        prefix: str = "",
        max_depth: int | None = None,
        include_files: bool = True,
        directory_first: bool = True,
) -> None:
    """
    Recursively print a simple tree view of directory contents.

    Parameters
    ----------
    start_path : str or pathlib.Path
        Root directory to display as a tree.
    prefix : str, optional
        ASCII prefix used to visually align nested levels, by default "".
    max_depth : int or None, optional
        Maximum recursion depth relative to ``start_path``. If None, the full
        tree is printed, by default None.
    include_files : bool, optional
        If True, include files in the tree, otherwise only directories are
        shown, by default True.
    directory_first : bool, optional
        If True, list subdirectories before files at each level, by default
        True.

    Notes
    -----
    This function is intended for quick inspection of project folder
    structures (e.g. per-animal fibre photometry data layouts).
    """
    start_path = Path(start_path)

    if not start_path.exists():
        raise FileNotFoundError(f"Path does not exist: {start_path}")

    if not start_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {start_path}")

    def _iter_entries(path: Path) -> Iterable[Path]:
        entries = list(path.iterdir())
        if not include_files:
            entries = [p for p in entries if p.is_dir()]

        if directory_first:
            entries.sort(key=lambda p: (p.is_file(), p.name.lower()))
        else:
            entries.sort(key=lambda p: p.name.lower())
        return entries

    def _print_tree(current_path: Path, current_prefix: str, current_depth: int) -> None:
        if max_depth is not None and current_depth > max_depth:
            return

        entries = list(_iter_entries(current_path))
        n_entries = len(entries)
        for idx, entry in enumerate(entries):
            connector = "└── " if idx == n_entries - 1 else "├── "
            print(f"{current_prefix}{connector}{entry.name}")

            if entry.is_dir():
                extension = "    " if idx == n_entries - 1 else "│   "
                _print_tree(entry, current_prefix + extension, current_depth + 1)

    # Print the root name first for clarity
    print(start_path.name)
    _print_tree(start_path, prefix, current_depth=1)
if __name__ == "__main__":
    # Your hardcoded params
    BASE_PROJECT_PATH = Path(r"\\cmvm.datastore.ed.ac.uk\cmvm\sbms\users\s2830349\Win7\Desktop\Fibre_phot_project_2026")
    ANIMAL_ID = "4879"

    def project_root(animal_id: str) -> Path:
        """Construct the root directory for a given animal ID."""
        return BASE_PROJECT_PATH / animal_id

    root = project_root(ANIMAL_ID)
    print(f"Project tree for {ANIMAL_ID} at {root}:")

    # Debug first
    print("DEBUG: Root exists?", root.exists())
    if root.exists() and root.is_dir():
        print("DEBUG: Root contents:")
        for p in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            print(f"  {p.name} ({'DIR' if p.is_dir() else 'file'})")
        print("\nFull tree:")
        print_directory_tree(root, max_depth=None, include_files=True)
    else:
        print("DEBUG: Falling back to current script dir")
        print_directory_tree(Path(__file__).parent, max_depth=2)
