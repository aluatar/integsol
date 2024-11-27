from typing import (
    Literal,
    Iterable
)

from numpy import (
    array,
    float128,
)
import numpy as np

def is_element_dim_valide(dim: int, elements: Iterable) -> bool:
    bool_arr = [
        len(e) == dim + 1 if isinstance(e,Iterable) else False 
        for e in elements
        ]
    return all(bool_arr)


def is_point_in_placement(
    point: float | Iterable,
    array: Iterable | dict,
    placement: Literal["centers", "nodes"],
    fill: Literal["vtx", "edge", "boundary", "domain"],
) -> bool:
    if placement == "centers": 
        match fill:
            case "vtx":
                raise TypeError
            case "edge":
                if np.round(point, 4) in np.round(array["edg"], 4):
                    return True
            case "boundary":
                if np.round(point, 4) in np.round(array["tri"], 4):
                    return True
            case "domain":
                if np.round(point, 4) in np.round(array["tet"], 4):
                    return True
    
    elif placement == "nodes":
        if np.round(point, 4) in np.round(array, 4):
            return True
                
    else:
        return False
    

def is_mesh_filled(
    values: Iterable,
    array: Iterable | dict,
    placement: Literal["centers", "nodes"],
    fill: Literal["vtx", "edge", "boundary", "domain"],
) -> bool:
    if placement == "centers": 
        match fill:
            case "vtx":
                raise TypeError
            case "edge":
                if len(values) == len(array["edg"]):
                    return True
            case "boundary":
                if len(values) == len(array["tri"]):
                    return True
            case "domain":
                if len(values) == len(array["tet"]):
                    return True
    
    elif placement == "nodes":
        if len(values) == len(array):
            return True
                
    else:
        return False
    