"""
    Analysis pipeline for Fiber Photometry recordings
    Copyright (C) 2026 Dr Paul Rignanese, Kind Lab, University of Edinburgh

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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



def animal_output_dirs(animal_num: str) -> tuple[Path, Path]:
    """Return (figures_dir, csv_dir). Create if missing."""
    out_root = BASE_PROJECT_PATH / f"{animal_num}_output"
    figs_dir = out_root / "figures"
    csv_dir = out_root / "csv"

    created = False
    if not figs_dir.exists():
        figs_dir.mkdir(parents=True)
        created = True
    if not csv_dir.exists():
        csv_dir.mkdir(parents=True)
        created = True

    status = "created" if created else "exists"
    print(f"{animal_num}_output/ [{status}]")

    return figs_dir, csv_dir

def select_data_files(recording_type: str, events_type: str, animal_id: str = ANIMAL_ID, base_path: Path = BASE_PROJECT_PATH) -> list[Path]:
    """
    Select fibre photometry data files based on recording type and events type.

    Parameters
    ----------
    recording_type : str
        Recording session identifier ('hab1', 'hab2', 'cond', 'recall').
    events_type : str
        Event data type ('fluorescence', 'fluorescence_event', 'csv').
    animal_id : str, optional
        Animal identifier (default from global ANIMAL_ID).
    base_path : Path, optional
        Base project path (default from global BASE_PROJECT_PATH).

    Returns
    -------
    list[Path]
        List of selected data file paths in the appropriate folder.

    Notes
    -----
    Assumes data files are organized in subfolders like 'data/{animal_id}/{recording_type}'
    with files matching patterns. Falls back to space files if dir missing.
    For neuroscience fibre photometry analysis in GCaMP/dLight recordings with TTL events.
    """
    data_dir = base_path / "data" / f"{animal_id}" / recording_type
    file_patterns = {
        "fluorescence": [f"{animal_id}_Fluorescence.csv", "Fluorescence.csv", "cleaned_fluorescence.csv"],
        "fluorescence_event": ["4879_Fluorescence_Event_freezing_cs.csv", "fluorescence_event*.csv"],
        "csv": ["*.csv"]
    }

    data_files = []
    if data_dir.exists():
        for pattern in file_patterns.get(events_type, ["*.csv"]):
            data_files.extend(list(data_dir.glob(pattern)))

    # Fallback to known space files if no matches (adapt as needed)
    if not data_files:
        fallback_patterns = file_patterns[events_type]
        print(f"Warning: No files in {data_dir}. Using fallback patterns: {fallback_patterns}")
        # In practice, load from space-attached files here or adjust BASE_PROJECT_PATH to include them

    if not data_files:
        raise FileNotFoundError(f"No matching files found for {recording_type}/{events_type}")

    return sorted(data_files)













