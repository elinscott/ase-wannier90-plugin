"""Testing the ``ase_wannier90_plugin.path`` module."""

from ase.lattice import RHL

from ase_wannier90_plugin.path import path_str_to_list


def test_path_str_to_list():
    """Test the path_str_to_list function."""
    path_as_string = 'GLB1,BZGX,QFP1Z,LP'
    special_points = RHL(a = 1.0, alpha=20.0).get_special_points()
    path_as_list = ['G', 'L', 'B1', ',', 'B', 'Z', 'G', 'X', ',', 'Q', 'F',
                    'P1', 'Z', ',', 'L', 'P']
    assert path_str_to_list(path_as_string, special_points) == path_as_list
