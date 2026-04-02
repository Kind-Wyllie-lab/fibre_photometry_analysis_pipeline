from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from params import animal_output_dirs
from session import PhotometrySession


@dataclass
class PhotometryPipeline:
    """
    End-to-end pipeline for batch processing fiber photometry data across animals and sessions.

    Parameters
    ----------
    base_directory : str or Path
        Root directory containing animal folders.
    session_names : sequence of str
        Session names to evaluate for each animal.
    animal_names : sequence of str or None, default=None
        Explicit animal identifiers to process. If ``None``, all folders beginning
        with ``"Rat"`` in `base_directory` are used.
    run_preprocessing : bool, default=True
        Whether to execute preprocessing.
    run_event_sorting : bool, default=True
        Whether to execute event sorting.
    run_signal_processing : bool, default=True
        Whether to execute signal processing.
    run_plotting : bool, default=True
        Whether to execute plotting.

    Attributes
    ----------
    results : list of PhotometrySession
        Completed session objects in processing order.
    """

    base_directory: Path | str
    session_names: Sequence[str]
    animal_names: Optional[Sequence[str]] = None
    run_preprocessing: bool = True
    run_event_sorting: bool = True
    run_signal_processing: bool = True
    run_plotting: bool = True

    results: list[PhotometrySession] = field(default_factory=list, init=False)

    def get_animal_names(self) -> list[str]:
        """
        Resolve the list of animals to process.

        Returns
        -------
        list of str
            Sorted animal identifiers.
        """
        if self.animal_names is not None:
            return list(self.animal_names)

        base_directory = Path(self.base_directory)
        return sorted(
            [name for name in os.listdir(base_directory) if name.startswith("Rat")]
        )

    def build_session(self, animal: str, session_name: str) -> PhotometrySession:
        """
        Create a session object for one animal/session pair.

        Parameters
        ----------
        animal : str
            Animal identifier.
        session_name : str
            Session identifier.

        Returns
        -------
        PhotometrySession
            Configured session object.
        """
        return PhotometrySession(
            animal=animal,
            session_name=session_name,
            base_directory=self.base_directory,
            output_directory=animal_output_dirs(animal, session_name),
            run_preprocessing=self.run_preprocessing,
            run_event_sorting=self.run_event_sorting,
            run_signal_processing=self.run_signal_processing,
            run_plotting=self.run_plotting,
        )

    def run(self) -> list[PhotometrySession]:
        """
        Run the full pipeline across all requested animals and sessions.

        Returns
        -------
        list of PhotometrySession
            Completed session objects.
        """
        self.results = []

        for animal in self.get_animal_names():
            print(animal)
            for session_name in self.session_names:
                print(session_name)
                session_processor = self.build_session(animal, session_name)
                session_processor.run()
                self.results.append(session_processor)
        return self.results
