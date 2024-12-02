from integsol.base import BaseClass
from integsol.mesh.mesh import Mesh
from typing import (
    Any, 
    Literal,
    Iterable,
)
from numpy import (
    array
)

class IntegralConvolutionOperator(BaseClass):
    def __init__(
        self,
        kernel: Any,
        dim: int | None=3,
        mesh: Mesh | None=None,
        mesh_matrix: Iterable | None=None,
    ):
        self.kernel = kernel
        self.dim = dim
        self.Mesh = mesh
        self.mesh_matrix = mesh_matrix

    def to_mesh_matrix(
        self,
        mesh: Mesh,
        placement: Literal["centers", "nodes"] | None="centers",
    ) -> array:
        
        self.mesh = mesh        
        
        if placement == "centers":
            points_array = mesh.elements_centers
        elif placement == "nodes":
            points_array = mesh.coordinates
        else:
            raise AttributeError(name="placement type error")
        
        matrix = []
        for point in points_array:
            row = []
            for point_prime in points_array:
                row.append(self.kernel(point, point_prime))
            
            matrix.append(row)
        
        self.mesh_matrix = array(matrix)
        return self.mesh_matrix
    


