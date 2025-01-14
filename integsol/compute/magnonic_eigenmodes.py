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
from typing import (
    Any,
    Iterable,
)

SATURATION_MAGNETIZATION: float =  1.45 * 1e4 # [A / m]
GYROMAGNETIC_CONSTANT: float = 2.25 * 1e4 # [m /(A * s)]

def get_comsol_mesh(path: str) -> Mesh:
    mesh = Mesh.read(path=path)
    return mesh


def get_precalculated_magnetization(path: str, mesh: Mesh) -> Tensor:
    magnetization = VectorField.read_to_mesh(path=path, mesh=mesh)
    magnetization_matrix = CPO(mesh=mesh, left_vector=magnetization).to_mesh_matrix()
    return magnetization_matrix


def get_precalculated_demagnetization_field(path: str, mesh: Mesh) -> Tensor:
    demagtetization_field = VectorField.read_to_mesh(path=path, mesh=mesh)
    demagtetization_field_matrix = CPO(mesh=mesh, left_vector=demagtetization_field).to_mesh_matrix()
    return demagtetization_field_matrix


def get_uniform_external_field(field_vector: list[float], mesh: Mesh) -> Tensor:
    centers = mesh.elements_centers['tet']
    uniform_external_field = VectorField(
        mesh=mesh,
        coordinates=centers,
        values= np.array([field_vector for _ in centers])
    )
    uniform_external_field_matrix = CPO(mesh=mesh, left_vector=uniform_external_field).to_mesh_matrix()
    return uniform_external_field_matrix


def get_dipole_integral_convolution(
    kernel: Any,
    mesh: Mesh,
) -> Tensor:
    integral_convolution_operator = ICO(kernel=kernel)
    integral_convolution_matrix = integral_convolution_operator.to_mesh_matrix(mesh=mesh)
    return integral_convolution_matrix


def get_llg_operator(
    external_field_matrix: Tensor,
    demagnetization_field_matrix: Tensor,
    magnetization_matrix: Tensor,
    dipole_iontegral_convolution: Tensor,
) -> Iterable:
    dipole_term = matmul(magnetization_matrix,dipole_iontegral_convolution)
    operator = GYROMAGNETIC_CONSTANT * (external_field_matrix + demagnetization_field_matrix + SATURATION_MAGNETIZATION * dipole_term)
    return np.array(operator)

