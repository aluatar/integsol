from typing import (
    Literal,
    Iterable
)

from numpy import (
    array,
    float32,
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
                if float32(point) in float32(array["edg"]):
                    return True
            case "boundary":
                if float32(point) in float32(array["tri"]):
                    return True
            case "domain":
                if float32(point) in float32(array["tet"]):
                    return True
    
    elif placement == "nodes":
        if float32(point) in float32(array):
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
    