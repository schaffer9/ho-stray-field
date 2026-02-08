from typing import Any, Callable
import os
import csv
from pathlib import Path

import jax
import chex


def write_csv_row(file: str | Path, data: dict[str, Any]) -> None:
    fieldnames = list(data.keys())
    with open(file, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if os.stat(file).st_size == 0:
            writer.writeheader()

        writer.writerow(data)



def apply_along_last_dims[T](
    func: Callable[..., T], *arr: chex.ArrayTree, dims=1
) -> T:
    """Uses vmap to apply a function over all axes of an array
    except the last ones specified by `dims`.

    Parameters
    ----------
    func1d : Callable
    *args : ArrayTree
    dims : int, optional
        The number of remaining axes. If 0 the function is
        applied on each scalar of the array. Default is 1

    Returns
    -------
    ArrayTree
    """
    _dims = [a.ndim for a in jax.tree.leaves(arr)]
    max_dims = max(_dims)
    vmap_dims = max_dims - dims

    for axis in range(0, vmap_dims):
        func = jax.vmap(func, in_axes=axis, out_axes=axis)

    return func(*arr)
