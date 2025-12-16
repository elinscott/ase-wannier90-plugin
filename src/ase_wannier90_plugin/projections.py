"""Functions related to handling Wannier90 projections."""

from ase import Atoms
from wannier90_input.projections import Projection

from ase_wannier90_plugin.utils import formatted_str_to_list


def num_wann_lookup(proj: str) -> int:
    """Determine the number of Wannier functions for a given projection string."""
    database = {'l=0': 1, 's': 1,
                 'l=1': 3, 'p': 3, 'pz': 1, 'px': 1, 'py': 1,
                 'l=2': 5, 'd': 5, 'dxy': 1, 'dxz': 1, 'dyz': 1, 'dx2-y2': 1, 'dz2': 1,
                 'l=3': 7, 'f': 7, 'fz3': 1, 'fxz2': 1, 'fyz2': 1, 'fz(x2-y2)': 1, 'fxyz': 1,
                 'fx(x2-3y2)': 1, 'fy(3x2-y2)': 1,
                 'l=-1': 2, 'sp': 2, 'sp-1': 1, 'sp-2': 1,
                 'l=-2': 3, 'sp2': 3, 'sp2-1': 1, 'sp2-2': 1, 'sp2-3': 1,
                 'l=-3': 4, 'sp3': 4, 'sp3-1': 1, 'sp3-2': 1, 'sp3-3': 1, 'sp3-4': 1,
                 'l=-4': 5, 'sp3d': 5, 'sp3d-1': 1, 'sp3d-2': 1, 'sp3d-3': 1, 'sp3d-4': 1,
                 'sp3d-5': 1,
                 'l=-5': 6, 'sp3d2': 6, 'sp3d2-1': 1, 'sp3d2-2': 1, 'sp3d2-3': 1, 'sp3d2-4': 1,
                 'sp3d2-5': 1, 'sp3d2-6': 1}
    if proj in database:
        return database[proj]
    elif proj.split(',')[0] in database:
        ang_mtm, mr = proj.split(',')
        if ang_mtm in database and '=' in mr:
            _, val = mr.split('=')
            return len(formatted_str_to_list(val))

    raise NotImplementedError(f'Unrecognised Wannier90 projection {proj}')


def num_wann_from_projections(projections: list[Projection], atoms: Atoms) -> int:
    """Determine the number of Wannier functions for a list of Projections."""
    num_wann = 0
    for proj in projections:
        if proj.site is not None:
            if len(set(atoms.get_tags())) > 0:
                labels = [s + str(t) if t != 0 else s for s, t in
                          zip(atoms.symbols, atoms.get_tags(), strict=True)]
            else:
                labels = atoms.symbols
            num_sites = labels.count(proj.site)
        else:
            num_sites = 1
        num_wann += num_sites * proj.number_of_orbitals()
    return num_wann
