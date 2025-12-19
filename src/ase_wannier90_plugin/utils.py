"""Utility functions for the ASE Wannier90 plugin."""

import itertools
import json
from typing import Any


def list_to_formatted_str(values: list[int]) -> str:
    """Convert a list of integers into the format expected by Wannier90.

    e.g. list_to_formatted_str([1, 2, 3, 4, 5, 7]) = "1-5,7"
    """
    if len(values) == 0:
        raise ValueError("list_to_formatted_str() should not be given an empty list")
    if not all(a < b for a, b in itertools.pairwise(values)):
        raise ValueError("values must be monotonically increasing")
    indices: list[int | None] = [None]
    indices += [i + 1 for i in range(len(values) - 1) if values[i + 1] != values[i] + 1]
    indices += [None]
    sectors = [values[slice(a, b)] for a, b in itertools.pairwise(indices)]
    out: list[str] = []
    for sector in sectors:
        if len(sector) == 1:
            out.append(str(sector[0]))
        else:
            out.append(f"{sector[0]}-{sector[-1]}")
    return ",".join(out)


def formatted_str_to_list(string: str) -> list[int]:
    """Perform the inverse of list_to_formatted_str."""
    out: list[int] = []
    for section in string.split(","):
        if "-" in section:
            out += list(range(int(section.split("-")[0]), int(section.split("-")[1]) + 1))
        else:
            out.append(int(section))
    return out


def parse_value(value: Any) -> Any:
    """Parse a value from a string to an appropriate Python type."""
    parsed_value: Any
    if isinstance(value, list):
        parsed_value = []
        for v in value:
            parsed_value.append(parse_value(v))
    else:
        if isinstance(value, str):
            if value.lower() in ["t", "true", ".true."]:
                return True
            elif value.lower() in ["f", "false", ".false."]:
                return False
        try:
            parsed_value = json.loads(value)
        except (TypeError, json.decoder.JSONDecodeError):
            parsed_value = value
    return parsed_value
