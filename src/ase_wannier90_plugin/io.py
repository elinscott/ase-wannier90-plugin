"""Writing and reading functions for Wannier90."""

import re
from typing import TextIO
import itertools
import numpy as np

from ase import Atoms
from ase.cell import Cell
from wannier90_input.projections import Projection

from ase_wannier90_plugin.utils import list_to_formatted_str, parse_value, formatted_str_to_list
from ase_wannier90_plugin.path import path_str_to_list, construct_kpoint_path


def write_wannier90_in(fd: TextIO, atoms: Atoms):
    """Write Wannier90 input to file."""
    settings = atoms.calc.parameters.items

    for kw, opt in settings.items():

        if kw == 'kpoints':
            if np.ndim(opt) == 2:
                rows_str = [' '.join([str(value) for value in row] + [str(1 / len(opt))]) for row in opt]
                fd.write(block_format('kpoints', rows_str))

        elif kw == 'projections':
            rows_str = [str(proj) for proj in opt]
            fd.write(block_format(kw, rows_str))

        elif kw == 'mp_grid':
            opt_str = ' '.join([str(value) for value in opt])
            fd.write(f'{kw} = {opt_str}\n')

        elif kw == 'kpoint_path':
            if settings.get('bands_plot', False):
                rows_str = []
                path_list = path_str_to_list(opt.path, opt.special_points)
                for start, end in zip(path_list[:-1], path_list[1:]):
                    if ',' in [start, end]:
                        continue
                    rows_str.append('')
                    for point in (start, end):
                        rows_str[-1] += ' ' + point + ' '.join([f'{v:9.5f}' for v in opt.special_points[point]])
                fd.write(block_format(kw, rows_str))

        elif isinstance(opt, list) and kw != 'exclude_bands':
            if np.ndim(opt) == 1:
                opt = [opt]
            rows_str = [' '.join([str(value) for value in row]) for row in opt]
            fd.write(block_format(kw, rows_str))

        else:
            if kw == 'exclude_bands' and isinstance(opt, list):
                opt_str = list_to_formatted_str(opt)
            elif isinstance(opt, bool):
                if opt:
                    opt_str = '.true.'
                else:
                    opt_str = '.false.'
            else:
                opt_str = str(opt)
            fd.write(f'{kw} = {opt_str}\n')

    # atoms_frac
    if atoms.has('tags'):
        labels = [sym + str(tag) if tag != 0 else sym for sym, tag in zip(atoms.symbols, atoms.get_tags())]
    else:
        labels = atoms.get_chemical_symbols()
    rows_str = [label + ' ' + ' '.join([str(coord) for coord in position]) for label, position in
                zip(labels, atoms.get_scaled_positions(), strict=True)]
    fd.write(block_format('atoms_frac', rows_str))

    # unit_cell_cart
    rows_str = ['ang'] + [' '.join([str(v) for v in row]) for row in atoms.cell]
    fd.write(block_format('unit_cell_cart', rows_str))


def block_format(name: str, rows_str: list[str]) -> str:
    """Format a Wannier90 input block."""
    joined_rows = '\n'.join(['  ' + r for r in rows_str])
    return f'\nbegin {name}\n{joined_rows}\nend {name}\n'


def read_wannier90_in(fd: TextIO) -> Atoms:
    """Read a wannier90 input file.

    Based on ase.io.castep.read_freeform
    """
    filelines = fd.readlines()

    keyw = None
    read_block = False
    block_lines = None
    kpoint_path = None

    calc = Wannier90()

    for i, line in enumerate(filelines):

        # Strip all comments, aka anything after a hash
        line = line.split('#', 1)[0].strip()

        if line == '':
            # Empty line... skip
            continue

        lsplit = re.split(r'\s*[:=]*\s+', line, maxsplit=1)

        keyword_parsers = {
            'projections': _parse_projections,
            'kpoints': _parse_kpoints
        }

        if read_block:
            if lsplit[0].lower() == 'end':
                if len(lsplit) == 1 or lsplit[1].lower() != keyw:
                    raise ValueError(f'Out of place end of block at line {i+1}')
                else:
                    read_block = False
                    if keyw == 'unit_cell_cart':
                        cell = Cell(_parse_unit_cell_cart(block_lines))
                    elif keyw == 'atoms_frac':
                        symbols, tags, scaled_positions = _parse_atoms_frac(block_lines)
                    elif keyw == 'kpoint_path':
                        kpoint_path = _parse_kpoint_path(block_lines)
                    else:
                        parser = keyword_parsers.get(keyw, parse_value)
                        calc.parameters[keyw] = parser(block_lines)
            else:
                block_lines += [line.split()]
        else:
            # Check the first word
            # Is it a block?
            read_block = (lsplit[0].lower() == 'begin')
            if read_block:
                if len(lsplit) == 1:
                    raise ValueError(f'Unrecognizable block at line {i+1}')
                keyw = lsplit[1].lower()
            else:
                keyw = lsplit[0].lower()

            # Now save the value
            if read_block:
                block_lines = []
            elif keyw == 'mp_grid':
                calc.parameters[keyw] = [parse_value(v) for v in lsplit[1].split()]
            elif keyw == 'exclude_bands':
                calc.parameters[keyw] = formatted_str_to_list(' '.join(lsplit[1:]))
            else:
                calc.parameters[keyw] = parse_value(' '.join(lsplit[1:]))

    # Convert kpoint_path to a BandPath object
    if kpoint_path is not None:
        calc.parameters.kpoint_path = construct_kpoint_path(
            path=kpoint_path, cell=cell,
            bands_point_num=calc.parameters.get('bands_point_num', 100))

    atoms = Atoms(symbols=symbols, scaled_positions=scaled_positions, cell=cell, calculator=calc,
                  tags=tags)
    atoms.calc.atoms = atoms

    return atoms

def _parse_unit_cell_cart(lines: list[list[str]]) -> Cell:
    if lines.pop(0) != ['ang']:
        raise ValueError('Only angstrom unit_cell_cart currently supported')
    return Cell(parse_value(lines))

def _parse_atoms_frac(lines: list[list[str]]) -> tuple[list[str], list[int], np.ndarray]:
    symbols = []
    tags = []
    for line in lines:
        symbols.append(''.join([char for char in line[0] if char.isalpha()]))
        tag = ''.join([char for char in line[0] if char.isnumeric()])
        if tag:
            tags.append(int(tag))
        else:
            tags.append(0)
    scaled_positions = parse_value([line[1:] for line in lines])
    return symbols, tags, scaled_positions

def _parse_projections(lines: list[list[str]]) -> list[Projection]:
    return [Projection.from_string(' '.join(line)) for line in lines]

def _parse_kpoints(lines: list[list[str]]) -> np.ndarray:
    return np.array([line[:-1] for line in lines], dtype=float)

def _parse_kpoint_path(block_lines: list[list[str]]) -> str:
    if len(block_lines) == 1:
        kpoint_path = block_lines[0][0] + block_lines[0][4]
    else:
        kpoint_path = block_lines[0][0]
        for prev_line, line in itertools.pairwise(block_lines):
            if line[0] != prev_line[4]:
                kpoint_path += prev_line[4] + ','
            kpoint_path += line[0]
        kpoint_path += line[4]
    return kpoint_path


def read_wannier90_out(fd: TextIO | str):
    """Read wannier90 output files.

    Parameters
    ----------
    fd : file|str
        A file like object or filename

    Yields
    ------
    structure : atoms
        An Atoms object with an attached SinglePointCalculator containing
        any parsed results
    """
    if isinstance(fd, str):
        fd = open(fd)

    flines = fd.readlines()

    job_done = False
    walltime = None
    convergence = False
    convergence_dis = None
    imre_ratio: list[float] = []
    centers: list[list[float]] = []
    spreads: list[float] = []

    for i, line in enumerate(flines):
        if 'All done' in line:
            job_done = True
        elif 'Exiting...' in line and '.nnkp written' in line:
            job_done = True
        elif 'Wannierisation convergence criteria satisfied' in line:
            convergence = True
        elif 'DISENTANGLE' in line:
            convergence_dis = False
        elif 'Disentanglement convergence criteria satisfied' in line:
            convergence_dis = True
        elif 'Total Execution Time' in line or 'Time to write kmesh' in line:
            walltime = float(line.split()[-2])
        elif 'Maximum Im/Re Ratio' in line:
            imre_ratio.append(float(line.split()[-1]))
        elif 'Final State' in line:
            j = 1
            while 'WF centre and spread' in flines[i + j]:
                splitline = [x.rstrip(',') for x in re.sub('[(),]', ' ', flines[i + j]).split()]
                centers.append([float(x) for x in splitline[5:8]])
                spreads.append(float(splitline[-1]))
                j += 1

    atoms = Atoms()
    calc = Wannier90(atoms=atoms)
    calc.results['job done'] = job_done
    calc.results['walltime'] = walltime
    calc.results['convergence'] = convergence and convergence_dis in [True, None]
    calc.results['Im/Re ratio'] = imre_ratio
    calc.results['centers'] = centers
    calc.results['spreads'] = spreads

    atoms.calc = calc

    yield atoms
