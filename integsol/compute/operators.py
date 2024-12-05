from integsol.base import BaseClass
from integsol.mesh.mesh import Mesh
from typing import (
    Any, 
    Literal,
    Iterable,
)
from numpy import (
    array,
    zeros,
)
import sys
from time import time
from torch import Tensor

FillElementTypesMap = {
    "vtx": "vtx",
    "edge": "edg",
    "boundary": "tri",
    "domain": "tet",
}

class IntegralConvolutionOperator(BaseClass):
    def __init__(
        self,
        kernel: Any,
        dim: int | None=3,
        mesh: Mesh | None=None,
        mesh_matrix: Tensor | None=None,
    ):
        self.kernel = kernel
        self.dim = dim
        self.mesh = mesh
        self.mesh_matrix = mesh_matrix

    def to_mesh_matrix(
        self,
        mesh: Mesh,
        placement: Literal["centers", "nodes"] | None="centers",
        fill: Literal["vtx", "edge", "boundary", "domain"] | None="domain",
    ) -> Tensor:
        start = time()
        if self.mesh is None:
            self.mesh = mesh

        if placement == "centers":
            points_array = self.mesh.elements_centers.get(FillElementTypesMap[fill])
            measures_array = self.mesh.elements_measures.get(FillElementTypesMap[fill])
        elif placement == "nodes":
            points_array = self.mesh.coordinates.get(FillElementTypesMap[fill])
        else:
            raise AttributeError(name="placement type error")
        
        matrix = []
        _len = len(points_array)
        print(f"Begin placement of operator on mesh elements' {placement}.")
        for step, point in enumerate(points_array):
            row = [[] for _ in range(self.dim)]
            for point_prime, measure in zip(points_array, measures_array):
                kernel_evaluated = self.kernel(point, point_prime)
                if not isinstance(kernel_evaluated, Iterable):
                    raise Exception
                
                for i in range(self.dim):
                    row[i].extend(measure * array(kernel_evaluated[i]))
            
            matrix.extend(row)
            sys.stdout.write(f"\rProgress: {round(100 * step / _len, 2)}%")
            sys.stdout.flush()
        print('\n')
        self.mesh_matrix = Tensor(matrix)
        finish = time()
        print(f"Mesh matric of the operator generated in {finish - start} seconds.")
        return self.mesh_matrix
    

class CrossProductOperator(BaseClass):

    def __init__(
            self,
            mesh: Mesh | None=None,
            dim: int | None=3,
            left_vector: Tensor | None=None,
            mesh_matrix: Tensor | None=None
    ):
        self.mesh = mesh
        self.dim = dim
        self.left_vector = left_vector
        self.mesh_matrix = mesh_matrix


    @staticmethod
    def get_cross_product_blocks(vector: Iterable, dim: int) -> list:
        sigrature = [
            [(0,0), (-1,2), (1,1)],
            [(1,2), (0,1), (-1,0)],
            [(-1,1), (1,0), (0,2)]
        ]
        _range = len(vector) // dim
        blocks_array = []
        for i in range(_range):
            block = []
            for ei in range(dim):
                sgnt = sigrature[ei]
                block.append(
                    [
                        sgnt[0][0] * vector[i*dim + sgnt[0][1]],
                        sgnt[1][0] * vector[i*dim + sgnt[1][1]],
                        sgnt[2][0] * vector[i*dim + sgnt[2][1]],
                    ]
                )
            blocks_array.append(block)
        return blocks_array


    def to_mesh_matrix(
        self,
        mesh: Mesh | None=None,
        left_vector: Tensor | None=None,        
        dim: int | None=None,
        placement: Literal["centers", "nodes"] | None="centers",
        fill: Literal["vtx", "edge", "boundary", "domain"] | None="domain",
    ) -> Tensor:
        
        if mesh is None:
            mesh = self.mesh
        if left_vector is None:
            left_vector = self.left_vector
        if dim is None:
            dim = self.dim

        if placement == "centers":
            points_array = self.mesh.elements_centers.get(FillElementTypesMap[fill])
        elif placement == "nodes":
            points_array = self.mesh.coordinates.get(FillElementTypesMap[fill])
        else:
            raise AttributeError(name="placement type error")
        
        mesh_matrix = zeros(shape=(len(left_vector), len(left_vector)))
        blocks = self.get_cross_product_blocks(vector=left_vector, dim=dim)

        for i in range(len(points_array)):
            for ei in range(dim):
                for ej in range(dim):
                    mesh_matrix[i * dim + ei][i * dim + ej] = blocks[i][ei][ej]
        
        self.mesh_matrix = Tensor(mesh_matrix)
        return self.mesh_matrix

        

