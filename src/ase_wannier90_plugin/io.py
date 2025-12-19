"""Writing and reading functions for Wannier90."""

import itertools
import re
from collections.abc import Callable, Generator
from typing import Any, ClassVar, TextIO

import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.calculators.calculator import BadConfiguration, Calculator, PropertyNotImplementedError
from ase.cell import Cell
from ase.dft.kpoints import BandPath
from ase.utils.plugins import ExternalIOFormat
from wannier90_input.models.parameters import Projection

from ase_wannier90_plugin.path import construct_kpoint_path, path_str_to_list
from ase_wannier90_plugin.utils import formatted_str_to_list, list_to_formatted_str, parse_value


def _write_kpoints(fd: TextIO, kpoints: npt.NDArray[np.float64]) -> None:
    if np.ndim(kpoints) == 2:
        rows_str = [
            " ".join([str(value) for value in row] + [str(1 / len(kpoints))]) for row in kpoints
        ]
        fd.write(block_format("kpoints", rows_str))


def _write_projections(fd: TextIO, projections: list[Projection]) -> None:
    rows_str = [str(proj) for proj in projections]
    fd.write(block_format("projections", rows_str))


def _write_mp_grid(fd: TextIO, mp_grid: list[int]) -> None:
    opt_str = " ".join([str(value) for value in mp_grid])
    fd.write(f"mp_grid = {opt_str}\n")


def _write_kpoint_path(fd: TextIO, kpoint_path: BandPath) -> None:
    rows_str = []
    path_list = path_str_to_list(kpoint_path.path, kpoint_path.special_points)
    for start, end in itertools.pairwise(path_list):
        if "," in [start, end]:
            continue
        rows_str.append("")
        for point in (start, end):
            rows_str[-1] += (
                " " + point + " ".join([f"{v:9.5f}" for v in kpoint_path.special_points[point]])
            )
    fd.write(block_format("kpoint_path", rows_str))


def _write_exclude_bands(fd: TextIO, exclude_bands: list[int] | str) -> None:
    if isinstance(exclude_bands, list):
        opt_str = list_to_formatted_str(exclude_bands)
    else:
        opt_str = exclude_bands
    fd.write(f"exclude_bands = {opt_str}\n")


def _write_list(fd: TextIO, name: str, values: list[Any]) -> None:
    if np.ndim(values) == 1:
        values = [values]
    rows_str = [" ".join([str(value) for value in row]) for row in values]
    fd.write(block_format(name, rows_str))


def _write_value(fd: TextIO, name: str, value: Any) -> None:
    if isinstance(value, bool):
        if value:
            opt_str = ".true."
        else:
            opt_str = ".false."
    else:
        opt_str = str(value)
    fd.write(f"{name} = {opt_str}\n")


def _write_atoms(fd: TextIO, atoms: Atoms) -> None:
    """Write the atoms to file in Wannier90 format."""
    if atoms.has("tags"):
        labels = [
            sym + str(tag) if tag != 0 else sym
            for sym, tag in zip(atoms.symbols, atoms.get_tags(), strict=True)
        ]
    else:
        labels = atoms.get_chemical_symbols()
    rows_str = [
        label + " " + " ".join([str(coord) for coord in position])
        for label, position in zip(labels, atoms.get_scaled_positions(), strict=True)
    ]
    fd.write(block_format("atoms_frac", rows_str))


def _write_unit_cell_cart(fd: TextIO, cell: Cell) -> None:
    """Write the unit cell to file in Wannier90 format."""
    rows_str = ["ang"] + [" ".join([str(v) for v in row]) for row in cell.array]
    fd.write(block_format("unit_cell_cart", rows_str))


def write_wannier90_in(fd: TextIO, atoms: Atoms, **parameters: dict[str, Any]) -> None:
    """Write Wannier90 input to file."""
    writers: dict[str, Callable[[TextIO, Any], None]] = {
        "kpoints": _write_kpoints,
        "projections": _write_projections,
        "mp_grid": _write_mp_grid,
        "exclude_bands": _write_exclude_bands,
    }
    for kw, opt in parameters.items():
        if kw in writers:
            writers[kw](fd, opt)
        elif kw == "kpoint_path":
            if parameters.get("bands_plot", False):
                if not isinstance(opt, BandPath):
                    raise ValueError(
                        "kpoint_path must be a BandPath object when bands_plot is True"
                    )
                _write_kpoint_path(fd, opt)
        elif isinstance(opt, list):
            _write_list(fd, kw, opt)
        else:
            _write_value(fd, kw, opt)

    # atoms_frac
    _write_atoms(fd, atoms)

    # unit_cell_cart
    _write_unit_cell_cart(fd, atoms.cell)


def block_format(name: str, rows_str: list[str]) -> str:
    """Format a Wannier90 input block."""
    joined_rows = "\n".join(["  " + r for r in rows_str])
    return f"\nbegin {name}\n{joined_rows}\nend {name}\n"


def read_wannier90_in(fd: TextIO) -> Atoms:
    """Read a wannier90 input file.

    Based on ase.io.castep.read_freeform
    """
    from ase_wannier90_plugin.calculator import Wannier90, Wannier90Profile

    filelines = fd.readlines()

    keyw = None
    read_block = False
    block_lines: list[list[str]] = []
    kpoint_path = None

    try:
        calc = Wannier90()
    except BadConfiguration:
        calc = Wannier90(profile=Wannier90Profile(command="wannier90.x"))

    parser: Callable[[Any], Any]
    keyw = None
    for i, line in enumerate(filelines):
        # Strip all comments
        line = line.split("#", 1)[0].strip()

        # Skip empty lines
        if line == "":
            continue

        lsplit = re.split(r"\s*[:=]*\s+", line, maxsplit=1)

        keyword_parsers: dict[str, Callable[[Any], Any]] = {
            "projections": _parse_projections,
            "kpoints": _parse_kpoints,
            "mp_grid": _parse_mp_grid,
            "exclude_bands": _parse_exclude_bands,
        }

        if read_block:
            if lsplit[0].lower() == "end":
                if len(lsplit) == 1 or lsplit[1].lower() != keyw:
                    raise ValueError(f"Out of place end of block at line {i + 1}")
                else:
                    read_block = False
                    if keyw == "unit_cell_cart":
                        cell = Cell(_parse_unit_cell_cart(block_lines))
                    elif keyw == "atoms_frac":
                        symbols, tags, scaled_positions = _parse_atoms_frac(block_lines)
                    elif keyw == "kpoint_path":
                        kpoint_path = _parse_kpoint_path(block_lines)
                    elif keyw is not None:
                        parser = keyword_parsers.get(keyw, parse_value)
                        calc.parameters[keyw] = parser(block_lines)
            else:
                block_lines += [line.split()]
        else:
            # Check the first word
            # Is it a block?
            read_block = lsplit[0].lower() == "begin"
            if read_block:
                keyw = lsplit[1].lower()
                block_lines = []
            else:
                keyw = lsplit[0].lower()
                parser = keyword_parsers.get(keyw, _parse_other)
                calc.parameters[keyw] = parser(lsplit)

    # Convert kpoint_path to a BandPath object
    if kpoint_path is not None:
        calc.parameters["kpoint_path"] = construct_kpoint_path(
            path=kpoint_path, cell=cell, bands_point_num=calc.parameters.get("bands_point_num", 100)
        )

    atoms = Atoms(
        symbols=symbols,
        scaled_positions=scaled_positions,
        cell=cell,
        calculator=calc,
        tags=tags,
        pbc=True,
    )
    atoms.calc.atoms = atoms

    return atoms


def _parse_mp_grid(lsplit: list[str]) -> list[int]:
    return [int(v) for v in lsplit[1].split()]


def _parse_exclude_bands(lsplit: list[str]) -> list[int]:
    return formatted_str_to_list(" ".join(lsplit[1:]))


def _parse_other(lsplit: list[str]) -> Any:
    return parse_value(" ".join(lsplit[1:]))


def _parse_unit_cell_cart(lines: list[list[str]]) -> Cell:
    if lines.pop(0) != ["ang"]:
        raise ValueError("Only angstrom unit_cell_cart currently supported")
    return Cell(parse_value(lines))


def _parse_atoms_frac(
    lines: list[list[str]],
) -> tuple[list[str], list[int], npt.NDArray[np.float64]]:
    symbols = []
    tags = []
    for line in lines:
        symbols.append("".join([char for char in line[0] if char.isalpha()]))
        tag = "".join([char for char in line[0] if char.isnumeric()])
        if tag:
            tags.append(int(tag))
        else:
            tags.append(0)
    scaled_positions = parse_value([line[1:] for line in lines])
    return symbols, tags, scaled_positions


def _parse_projections(lines: list[list[str]]) -> list[Projection]:
    raise NotImplementedError("Parsing projections is not yet implemented.")


def _parse_kpoints(lines: list[list[str]]) -> npt.NDArray[np.float64]:
    return np.array([line[:-1] for line in lines], dtype=float)


def _parse_kpoint_path(block_lines: list[list[str]]) -> str:
    if len(block_lines) == 1:
        kpoint_path = block_lines[0][0] + block_lines[0][4]
    else:
        kpoint_path = block_lines[0][0]
        for prev_line, line in itertools.pairwise(block_lines):
            if line[0] != prev_line[4]:
                kpoint_path += prev_line[4] + ","
            kpoint_path += line[0]
        kpoint_path += line[4]
    return kpoint_path


class Wannier90ResultsCalculator(Calculator):
    """Container for Wannier90 post-processing results."""

    implemented_properties: ClassVar[list[str]] = []  # type: ignore[misc]
    # Ignoring type override because ASE wrongly does not declare this as a ClassVar

    def __init__(self, atoms: Atoms, **results: Any):
        super().__init__(atoms=atoms.copy())
        self.results: dict[str, Any] = {}
        for key, value in results.items():
            if value in [None, []]:
                continue
            self.results[key] = value

    def get_property(
        self, name: str, atoms: Atoms | None = None, allow_calculation: bool = True
    ) -> Any:
        """Return a property."""
        if atoms is None:
            atoms = self.atoms
        if name not in self.results or self.check_state(atoms):
            if allow_calculation:
                raise PropertyNotImplementedError(f"The property {name} is not available.")
            return None
        return self.results[name]


def read_wannier90_out(fd: TextIO | str) -> Generator[Atoms, None, None]:
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
    converged: bool | None = False
    converged_dis: bool | None = None
    imre_ratio: list[float] = []
    centers: list[list[float]] = []
    spreads: list[float] = []
    nnkp_file = None

    for i, line in enumerate(flines):
        if "All done" in line:
            job_done = True
        elif "Exiting..." in line and ".nnkp written" in line:
            job_done = True
            nnkp_file = line.split()[-2]
            converged = None
        elif "Wannierisation convergence criteria satisfied" in line:
            converged = True
        elif "DISENTANGLE" in line:
            converged_dis = False
        elif "Disentanglement convergence criteria satisfied" in line:
            converged_dis = True
        elif "Total Execution Time" in line or "Time to write kmesh" in line:
            walltime = float(line.split()[-2])
        elif "Maximum Im/Re Ratio" in line:
            imre_ratio.append(float(line.split()[-1]))
        elif "Final State" in line:
            j = 1
            while "WF centre and spread" in flines[i + j]:
                splitline = [x.rstrip(",") for x in re.sub("[(),]", " ", flines[i + j]).split()]
                centers.append([float(x) for x in splitline[5:8]])
                spreads.append(float(splitline[-1]))
                j += 1

    atoms = Atoms()
    calc = Wannier90ResultsCalculator(atoms=atoms)
    calc.results["job_done"] = job_done
    calc.results["walltime"] = walltime
    calc.results["converged"] = converged and converged_dis in [True, None]
    calc.results["imre_ratio"] = imre_ratio
    calc.results["centers"] = centers
    calc.results["spreads"] = spreads
    calc.results["nnkp_file"] = nnkp_file

    atoms.calc = calc

    yield atoms


WANNIER90_INPUT_FORMAT = ExternalIOFormat(
    "Input file for Wannier90", "1F", module="ase_wannier90_plugin.io", ext=".win"
)

WANNIER90_OUTPUT_FORMAT = ExternalIOFormat(
    "Output file for Wannier90", "1F", module="ase_wannier90_plugin.io", ext=".wout"
)
