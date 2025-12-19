"""ASE calculator for Wannier90."""

import itertools
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from ase import Atoms
from ase.calculators.genericfileio import BaseProfile, CalculatorTemplate, GenericFileIOCalculator
from ase.cell import Cell
from ase.dft.kpoints import BandPath
from ase.io import read, write
from ase.spectrum.band_structure import BandStructure


class Wannier90Profile(BaseProfile):
    """Profile for Wannier90 calculator."""

    def get_calculator_command(self, inputfile: str) -> list[str]:
        """Return the arguments to run the calculator."""
        return [inputfile]

    def version(self) -> str:
        """Return the version of the Wannier90 executable."""
        raise NotImplementedError()


class Wannier90Template(CalculatorTemplate):
    """Template for Wannier90 calculator."""

    _label = "wannier90"

    def __init__(self) -> None:
        super().__init__("wannier90", ["spreads", "band structure"])
        self.inputname = self._label + ".win"
        self.outputname = self._label + ".wout"
        self.errorname = self._label + ".werr"

    def write_input(
        self,
        profile: Wannier90Profile,
        directory: Path,
        atoms: Atoms,
        parameters: dict[str, Any],
        properties: list[str],
    ) -> None:
        """Generate a Wannier90 input file."""
        dst = directory / self.inputname

        write(dst, atoms, format="wannier90-in", **parameters)

    def execute(self, directory: Path, profile: Wannier90Profile) -> None:
        """Run the Wannier90 calculation."""
        profile.run(directory, self.inputname, self.outputname, errorfile=self.errorname)

    def read_results(self, directory: os.PathLike[Any]) -> Mapping[str, Any]:
        """Read results from a Wannier90 calculation."""
        directory_path = Path(directory)
        atoms_in = read(directory_path / self.inputname, format="wannier90-in")
        if not isinstance(atoms_in, Atoms):
            raise ValueError("Input file did not return an Atoms object")

        atoms_out = read(directory_path / self.outputname, format="wannier90-out")
        if not isinstance(atoms_out, Atoms):
            raise ValueError("Output file did not return an Atoms object")
        if atoms_out.calc is None:
            raise ValueError("Output Atoms object has no calculator attached")

        results = atoms_out.calc.results
        if not isinstance(results, dict):
            raise ValueError("Output file did not return a results dictionary")

        if atoms_in.calc.parameters.get("bands_plot", False):
            bs = _read_bandstructure(
                directory_path, self._label, atoms_in.cell, atoms_in.calc.parameters["kpoint_path"]
            )
            if bs is not None:
                results["band_structure"] = bs

        return results

    def load_profile(self, cfg: dict[str, Any], **kwargs: Any) -> Wannier90Profile:
        """Load a Wannier90Profile from a configuration."""
        return cast(Wannier90Profile, Wannier90Profile.from_config(cfg, self.name, **kwargs))


class Wannier90(GenericFileIOCalculator):
    """An ASE calculator for Wannier90."""

    def __init__(
        self, profile: Wannier90Profile | None = None, directory: str = ".", **kwargs: Any
    ):
        template = Wannier90Template()
        super().__init__(profile=profile, template=template, directory=directory, parameters=kwargs)


def _read_bandstructure(
    directory: Path, label: str, cell: Cell, path: BandPath
) -> BandStructure | None:
    if not os.path.isfile(directory / f"{label}_band.dat"):
        return None

    # Construct the higher-resolution bandpath from the *_band.labelinfo.dat file
    with open(directory / f"{label}_band.labelinfo.dat") as fd:
        flines = fd.readlines()
    kpts = []

    for start, end in itertools.pairwise(flines):
        start_label, i_start = start.split()[:2]
        end_label, i_end = end.split()[:2]
        kpts += (
            cell.bandpath(start_label + end_label, int(i_end) - int(i_start) + 1).kpts[:-1].tolist()
        )
    kpts.append([float(x) for x in flines[-1].split()[-3:]])
    kpath = BandPath(cell, kpts, path=path.path, special_points=path.special_points)

    # Read in the eigenvalues from the *_band.dat file
    with open(directory / f"{label}_band.dat") as fd:
        flines = fd.readlines()
    eigs: list[list[str]] = [[]]
    for line in flines[:-1]:
        splitline = line.strip().split()
        if len(splitline) == 0:
            eigs.append([])
        else:
            eigs[-1].append(line.strip().split()[-1])
    eigs_np = np.array(eigs, dtype=float).T

    # Construct the bandstructure
    return BandStructure(kpath, eigs_np[np.newaxis, :, :])
