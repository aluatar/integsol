from integsol.mesh.mesh import Mesh
from integsol.structures.kernels import demagnetization_tensor_kernel as demag_tensor
from integsol.structures.vectors import VectorField
from integsol.structures.operators import (
    IntegralConvolutionOperator as ICO,
    CrossProductOperator as CPO
) 
import numpy as np
from torch.linalg import eig
from torch import (
    Tensor,
    dot,
    matmul,
    mv,
)

def get_comsol_mesh(path: str) -> Mesh:
    mesh = Mesh.read(path=path)
    return mesh


def get_precalculated_magnetization(path: str, mesh: Mesh) -> VectorField:
    magnetization = VectorField.read_to_mesh(path=path, mesh=mesh)
    return magnetization