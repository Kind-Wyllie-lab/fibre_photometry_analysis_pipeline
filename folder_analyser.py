from pathlib import Path
from typing import Union
from collections.abc import Iterable
from params import ANIMAL_ID, BASE_PROJECT_PATH

def print_directory_tree(
        start_path: Union[str, Path],
        prefix: str = "",
        max_depth: Union[int, None] = None,
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

    print(start_path.name)
    _print_tree(start_path, prefix, current_depth=1)

if __name__ == "__main__":
    # Project paths
  #  BASE_PROJECT_PATH = Path(r"\\cmvm.datastore.ed.ac.uk\cmvm\sbms\users\s2830349\Win7\Desktop\Fibre_phot_project_2026")
   # ANIMAL_ID = "4987"  # or "Rat_4879"

    def project_root(animal_id: str) -> Path:
        """Flexible animal root: '4879' ↔ 'Rat_4879'."""
        base = BASE_PROJECT_PATH
        candidate = base / animal_id
        if candidate.exists():
            return candidate

        # Rat_ toggle
        alt_id = f"Rat_{animal_id}" if not animal_id.startswith("Rat_") else animal_id[4:]
        alt_candidate = base / alt_id
        if alt_candidate.exists():
            print(f"INFO: project_root('{animal_id}') → '{alt_id}'")
            return alt_candidate

        raise FileNotFoundError(f"No '{animal_id}' or '{alt_id}' in {base}")

    root=project_root(ANIMAL_ID)
    print(f"Project tree for {ANIMAL_ID} at {root}:")

    # Debug
    print("DEBUG: Root exists?", root.exists())
    if root.exists() and root.is_dir():
        print("DEBUG: Root contents:")
        for p in sorted(root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            print(f"  {p.name} ({'DIR' if p.is_dir() else 'file'})")
        print("\nFull tree:")
        print_directory_tree(root)
    else:
        print("DEBUG: Fallback to script dir")
        print_directory_tree(Path(__file__).parent, max_depth=2)

print("\n=== DEBUG PATH STRUCTURE ===")
animal_num = ANIMAL_ID.replace("Rat_", "")  # "4987"
print(f"Animal num: '{animal_num}'")

csv_paths = list(root.rglob("*.csv"))
print(f"Found {len(csv_paths)} CSVs:")

for csv_path in csv_paths[:5]:  # First 5
    print(f"  {csv_path}")
    print(f"    parent: {csv_path.parent.name}")
    print(f"    parent.parent: {csv_path.parent.parent.name}")
    print(f"    grandparent: {csv_path.parent.parent.parent.name}")

paradigms = {"Cond", "Hab1", "Hab2", "Recall"}
matching = [p for p in csv_paths if p.parent.parent.name in paradigms]
print(f"\nMatching paradigm dirs: {len(matching)}")
for p in matching[:3]:
    print(f"  {p.parent.parent.name}/{p.name}")

# === BATCH RENAME CSVs ===
print("\n=== RENAMING CSVs ===")
animal_num = ANIMAL_ID.replace("Rat_", "")  # "4987"

PARADIGMS = {"Cond", "Hab1", "Hab2", "Recall"}

renamed_count = 0
for csv_path in root.rglob("*.csv"):
    grandparent = csv_path.parent.parent.parent
    if grandparent.name in PARADIGMS:
        paradigm = grandparent.name

        # Skip if already renamed (starts with animal_num_paradigm_)
        if csv_path.name.startswith(f"{animal_num}_{paradigm}_"):
            print(f"SKIP (already renamed): {csv_path.name}")
            continue

        new_name = f"{animal_num}_{paradigm}_{csv_path.name}"
        new_path = csv_path.with_name(new_name)

        if new_path.exists():
            print(f"SKIP (conflict): {new_name}")
        else:
            csv_path.rename(new_path)
            print(f"RENAMED: {csv_path.name} → {new_name}")
            renamed_count += 1

print(f"\n {renamed_count} new renames")

# === REVERT NAMES (uncomment to use) ===
""""
print("\n=== REVERTING NAMES ===")
reverted = 0
for csv_path in root.rglob("*.csv"):
    grandparent = csv_path.parent.parent.parent
    if grandparent.name in PARADIGMS:
        paradigm = grandparent.name
        name_parts = csv_path.stem.split(f"{animal_num}_{paradigm}_", 1)
        if len(name_parts) > 1:
            orig_name = name_parts[1] + csv_path.suffix
            orig_path = csv_path.with_name(orig_name)

            if orig_path.exists():
                print(f"SKIP (orig exists): {orig_name}")
            else:
                csv_path.rename(orig_path)
                print(f"REVERTED: {csv_path.name} → {orig_name}")
                reverted += 1

print(f"\n {reverted} files reverted")"""

# === CREATE OUTPUT FOLDER ===
print("\n=== CREATING OUTPUT FOLDER ===")
output_base = BASE_PROJECT_PATH / f"output_{ANIMAL_ID}"
output_base.mkdir(exist_ok=True)
print(f" Created: {output_base}")

