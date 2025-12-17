"""ASE calculator for Wannier90."""

import os
import numpy as np
from os import PathLike
from typing import Any, Mapping
from ase import Atoms
from ase.cell import Cell
from ase.dft.kpoints import BandPath
from ase.spectrum.band_structure import BandStructure
from ase.calculators.genericfileio import GenericFileIOCalculator, BaseProfile, CalculatorTemplate
from ase.io import write, read

class Wannier90Profile(BaseProfile):

    def get_calculator_command(self, inputfile: str) -> list[str]:
        return [inputfile]
    
    def version(self):
        raise NotImplementedError()


class Wannier90Template(CalculatorTemplate):
    """Template for Wannier90 calculator."""

    _label = 'wannier90'

    def __init__(self):
        super().__init__( 'wannier90', ['spreads', 'band structure'])
        self.inputname = self._label + '.win'
        self.outputname = self._label + '.wout'
        self.errorname = self._label + '.werr'

    def write_input(self, profile: Wannier90Profile, directory: PathLike, atoms: Atoms,
                    parameters: dict[str, Any], properties: list[str]) -> None:
        dst = directory / self.inputname

        write(dst,
              atoms,
              format='wannier90-in',
              **parameters)

    def execute(self, directory: PathLike, profile: Wannier90Profile):
        profile.run(directory, self.inputname, self.outputname, errorfile=self.errorname)

    def read_results(self, directory: PathLike) -> Mapping[str, Any]:
        atoms_in = read(directory / self.inputname, format='wannier90-in')
        assert isinstance(atoms_in, Atoms)

        atoms_out = read(directory / self.outputname, format='wannier90-out')
        results = atoms_out.calc.results

        if atoms_in.calc.parameters.get('bands_plot', False):
            bs = _read_bandstructure(directory, self._label, atoms_in.cell, atoms_in.calc.parameters['kpoint_path'])
            results['band_structure'] = bs

        return {k: v for k, v in results.items() if v not in [None, []]}


    def load_profile(self, cfg, **kwargs) -> Wannier90Profile:
        return Wannier90Profile.from_config(cfg, self.name, **kwargs)


class Wannier90(GenericFileIOCalculator):
    """An ASE calculator for Wannier90."""

    def __init__(self, profile=None, directory='.', **kwargs):
        template = Wannier90Template()
        super().__init__(
            profile=profile,
            template=template,
            directory=directory,
            parameters=kwargs
        )


def _read_bandstructure(directory: PathLike, label: str, cell: Cell, path: BandPath) -> BandStructure | None:
    if not os.path.isfile(f'{directory}/{label}_band.dat'):
        return

    # Construct the higher-resolution bandpath from the *_band.labelinfo.dat file
    with open(f'{label}_band.labelinfo.dat') as fd:
        flines = fd.readlines()
    kpts = []

    for start, end in zip(flines[:-1], flines[1:]):
        start_label, i_start = start.split()[:2]
        end_label, i_end = end.split()[:2]
        kpts += cell.bandpath(start_label + end_label, int(i_end) - int(i_start) + 1).kpts[:-1].tolist()
    kpts.append([float(x) for x in flines[-1].split()[-3:]])
    kpath = BandPath(cell, kpts, path=path.path, special_points=path.special_points)
    import ipdb; ipdb.set_trace()

    # Read in the eigenvalues from the *_band.dat file
    with open(f'{directory}/{label}_band.dat') as fd:
        flines = fd.readlines()
    eigs = [[]]
    for line in flines[:-1]:
        splitline = line.strip().split()
        if len(splitline) == 0:
            eigs.append([])
        else:
            eigs[-1].append(line.strip().split()[-1])
    eigs = np.array(eigs, dtype=float).T

    # Construct the bandstructure
    return BandStructure(kpath, eigs[np.newaxis, :, :])
