import numpy as np
from numpy import (
    array
)
from typing import Iterable

def demagnetization_tensor_kernel(
        position: Iterable,
        position_prime: Iterable,
        dim: int | None=3,
) -> array:
    position, position_prime = array(position), array(position_prime)
    diff = position - position_prime
    abs_diff = np.linalg.norm(diff)
    _element = lambda i, j: (
        -3 * (diff[i] * diff[j]) / abs_diff ** 5 + (1 / abs_diff ** 3 if i == j else 0) if abs_diff != 0 else 0
    )
    evaluated = [
        [_element(i, j) for j in range(dim)]
        for i in range(dim)
    ]
    return np.array(evaluated)

