import functools
from typing import Callable
from functools import reduce


def compose(*funcs: Callable) -> Callable:
    """
    Composes a group of functions (f(g(h(...)))) into a single composite function.
    The functions are applied from right to left (h then g then f).
    """
    return reduce(lambda f, g: lambda x: f(g(x)), funcs)