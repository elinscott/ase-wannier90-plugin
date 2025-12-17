"""Functions for handling k-point paths for Wannier90 in ASE."""

import itertools
import math
from typing import Iterable

from ase.cell import Cell
from ase.dft.kpoints import BandPath, bandpath


def _path_lengths(path: str, cell: Cell, bands_point_num: int) -> list[int]:
    # Construct the metric for reciprocal space (based off Wannier90's "utility_metric" subroutine)
    recip_lat = 2 * math.pi * cell.reciprocal().array
    recip_metric = recip_lat @ recip_lat.T

    # Work out the lengths Wannier90 will assign each path (based off Wannier90's
    # "plot_interpolate_bands" subroutine)
    kpath_pts: list[int] = []
    kpath_len: list[float] = []
    special_points = cell.bandpath().special_points

    path_list = path_str_to_list(path, special_points.keys())

    for i, (start, end) in enumerate(itertools.pairwise(path_list)):
        if start == ',':
            kpath_pts.append(0)
        elif end == ',':
            kpath_pts.append(1)
        else:
            vec = special_points[end] - special_points[start]
            kpath_len.append(math.sqrt(vec.T @ recip_metric @ vec))
            if i == 0:
                kpath_pts.append(bands_point_num)
            else:
                kpath_pts.append(round(bands_point_num * kpath_len[-1] / kpath_len[0]))
    return kpath_pts


def path_str_to_list(path: str, special_points: Iterable[str]) -> list[str]:
    """Break down a k-point path specified by a string into a list of special points.

    e.g. 'GLB1,BZGX,QFP1Z,LP' -> ['G', 'L', 'B1', ',', 'B', 'Z', 'G', 'X', ',', 'Q', 'F',
                                  'P1', 'Z', ',', 'L', 'P']
    """
    path_list: list[str] = []
    while len(path) > 0:
        found = False
        for option in [*special_points, ',']:
            if path.endswith(option):
                path = path[:-len(option)]
                path_list.append(option)
                found = True
                break
        if not found:
            raise ValueError(f'Could not deconstruct {path} into individual high-symmetry points')
    return path_list[::-1]


def construct_kpoint_path(path: str, cell: Cell, bands_point_num: int) -> BandPath:
    """Construct a BandPath object from a path string and cell."""
    path_lengths = _path_lengths(path, cell, bands_point_num)
    special_points = cell.bandpath().special_points

    path_list = path_str_to_list(path, special_points)

    kpts = []
    for start, end, npoints in zip(path_list[:-1], path_list[1:], path_lengths, strict=True):
        if start == ',':
            pass
        elif end == ',':
            kpts.append(special_points[start].tolist())
        else:
            bp = bandpath(start + end, cell, npoints + 1)
            kpts += bp.kpts[:-1].tolist()
    # Don't forget about the final kpoint
    kpts.append(bp.kpts[-1].tolist())

    if len(kpts) != sum(path_lengths) + 1:
        raise AssertionError(
            'Did not get the expected number of kpoints; this suggests there is a bug in the code')

    return BandPath(cell=cell, kpts=kpts, path=path, special_points=special_points)
