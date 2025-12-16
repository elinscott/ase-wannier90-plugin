"""Testing the ``ase_wannier90_plugin.utils`` module."""
import pytest

from ase_wannier90_plugin.utils import formatted_str_to_list, list_to_formatted_str, parse_value


def test_list_to_formatted_str():
    """Test the list_to_formatted_str function."""
    lst = [1, 2, 3, 5]
    formatted_str = list_to_formatted_str(lst)
    assert formatted_str == "1-3,5"

def test_roundtrip():
    """Test that formatting and parsing a list is reversible."""
    original_list = [1, 2, 3, 5, 7, 8, 10]
    formatted_str = list_to_formatted_str(original_list)
    parsed_list = formatted_str_to_list(formatted_str)
    assert original_list == parsed_list

@pytest.mark.parametrize("input_value, expected", [
    ("true", True),
    ("False", False),
    ("[1, 2, 3]", [1, 2, 3]),
    ("42", 42),
    ("3.14", 3.14),
    ("some string", "some string")])
def test_parse_value(input_value, expected):
    """Test the parse_value function."""
    assert parse_value(input_value) == expected
