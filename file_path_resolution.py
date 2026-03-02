"""from pathlib import Path
from typing import Iterable, List
from params import BASE_PROJECT_PATH, ANIMAL_ID

def setup_and_resolve_data_paths(
        animal: str,
        patterns: Iterable[str] | None = None,
) -> tuple[List[str], Path, Path]:  # 3 returns

    Setup paths + resolve fluorescence files using global BASE_PROJECT_PATH.

    animal_dir = BASE_PROJECT_PATH / animal
    output_dir = animal_dir / "output"  # Fixed name
    output_dir.mkdir(exist_ok=True, parents=True)

    figures_dir = output_dir / "figures"  # Fixed reference
    figures_dir.mkdir(exist_ok=True, parents=True)

    if patterns is None:
        patterns = [f"{animal}_Fluorescence*.csv", "*Fluorescence*.csv"]

    paths = []
    for pat in patterns:
        if any(c in pat for c in "*?[]"):
            matches = sorted(animal_dir.glob(pat))
            paths.extend(matches)
        else:
            p_path = animal_dir / pat
            if p_path.exists():
                paths.append(p_path)

    paths = sorted({p.resolve() for p in paths})
    files = [str(p) for p in paths]

    print(f"Data root: {animal_dir.absolute()}")
    print(f"Output dir: {output_dir.absolute()} (created)")
    print(f"Figures dir: {figures_dir.absolute()} (created)")  # Added
    print(f"Patterns: {patterns}")
    print(f"Found {len(files)} files:")

    for i, p in enumerate(paths, 1):
        print(f"  {i:2d}. {p.name}")

    if not files:
        raise ValueError(f"No files matching {patterns} in {animal_dir}")

    print(f"\nFirst: {Path(files[0]).name}")
    print(f"COPY: DATA_FILES = {repr(files)}")

    return files, output_dir, figures_dir  # Return all 3

if __name__ == "__main__":
    files, output_dir, figures_dir = setup_and_resolve_data_paths(ANIMAL_ID)"""


def setup_and_resolve_data_paths(
        animal: str,
        patterns: Iterable[str] | None = None,
) -> tuple[List[str], Path, Path]:
    """Setup paths + resolve files using BASE_PROJECT_PATH."""
    animal_dir = BASE_PROJECT_PATH / animal
    output_dir = animal_dir / "output"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(exist_ok=True, parents=True)
    figures_dir.mkdir(exist_ok=True, parents=True)

    if patterns is None:
        patterns = [f"{animal}_Fluorescence*.csv", "*Fluorescence*.csv"]

    paths = []
    for pat in patterns:
        if any(c in pat for c in "*?[]"):
            matches = sorted(animal_dir.glob(pat))
            paths.extend(matches)
        else:
            p_path = animal_dir / pat
            if p_path.exists():
                paths.append(p_path)

    paths = sorted({p.resolve() for p in paths})
    files = [str(p) for p in paths]

    print(f"Data root: {animal_dir.absolute()}")
    print(f"Output dir: {output_dir.absolute()} (created)")
    print(f"Figures dir: {figures_dir.absolute()} (created)")
    print(f"Found {len(files)} files:")
    for i, p in enumerate(paths, 1):
        print(f"  {i:2d}. {p.name}")

    if not files:
        raise ValueError(f"No files in {animal_dir}")

    print(f"COPY: DATA_FILES = {repr(files)}")
    return files, output_dir, figures_dir

# Auto-setup paths
DATA_FILES, OUTPUT_DIR, FIGURES_DIR = setup_and_resolve_data_paths(ANIMAL_ID)















