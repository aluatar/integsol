from integsol.base import BaseClass
from integsol.mesh.mesh import Mesh
from typing import (
    Any
)
from numpy import (
    array
)

class Operator(BaseClass):
    def __init__(
        self,
        kernel: Any,
    ):
        self.kernel = kernel

    
    def act(
        self,
        vector: Any
    ):
        pass


class ConvolutionOperator(BaseClass):
    def __init__(
        self,
        kernel: Any,
    ):
        self.kernel = kernel

    def to_mesh_matrix(
        self,
        mesh: Mesh,
        dim: int | None=3
    ) -> array:
        pass
        
    


