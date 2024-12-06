from integsol.base import BaseClass
from integsol.mesh.mesh import Mesh
from copy import deepcopy
from typing import (
    Any, 
    Literal,
    Iterable,
)
from numpy import (
    array,
    concatenate,
    zeros,
    isnan,
    nan,
    sign,
)
from integsol.compute.vectors import VectorField
from abc import abstractmethod
import sys
from time import time
from torch import (
    Tensor,
    double,
)
import torch
from integsol.compute.algebra import levi_chitiva_3

torch.set_default_dtype(double)


class BaseLinearOperator(BaseClass):
    def __init__(
        self,
        kernel: Any,
        dim: int | None=3,
        mesh: Mesh | None=None,
        mesh_matrix: Any | None=None,

    ):
        self.dim = dim
        self.kernel = kernel
        self.mesh = mesh
        self.mesh_matrix = mesh_matrix

    @abstractmethod
    def to_mesh_matrix(
        self,
        mesh: Mesh,
        placement: Literal["centers", "nodes"] | None="centers",
        fill: Literal["vtx", "edge", "boundary", "domain"] | None="domain",
    ) -> Tensor:
        raise NotImplementedError
        

class IntegralConvolutionOperator(BaseLinearOperator):
    def __init__(
        self,
        kernel: Any,
        dim: int | None=3,
        mesh: Mesh | None=None,
        mesh_matrix: Tensor | None=None,
    ):
        super().__init__(
            kernel=kernel,
            dim=dim,
            mesh=mesh,
            mesh_matrix=mesh_matrix
        )
    
    @staticmethod
    def get_principle_value(
        _kernel: Iterable,
        diff_prev: Iterable | None,
        diff_next: Iterable | None,
        dim: int,
    ) -> Iterable:
        result = []
        for row in _kernel:
            result_row = []
            for ei in range(dim):
                if isnan(row[ei]):
                    prev_sign = sign(diff_prev[ei]) if diff_prev is not None else 0
                    next_sign = sign(diff_next[ei]) if diff_next is not None else 0
                    tot_sign = prev_sign + next_sign

                    if tot_sign > 0:
                        result_row.append(-2)
                    elif tot_sign < 0:
                        result_row.append(4)
                    else:
                        result_row.append(1)
                else:
                    result_row.append(row[ei])
            result.append(result_row)
        
        return array(result)
        
    
    def to_mesh_matrix(
        self,
        mesh: Mesh | None=None,
        placement: Literal["centers", "nodes"] | None="centers",
        fill: Literal["vtx", "edge", "boundary", "domain"] | None="domain",
    ) -> Tensor:
        start = time()
        if self.mesh is None:
            self.mesh = mesh

        if placement == "centers":
            points_array = self.mesh.elements_centers.get(mesh.FillElementTypesMap[fill])
            measures_array = self.mesh.elements_measures.get(mesh.FillElementTypesMap[fill])
        elif placement == "nodes":
            points_array = self.mesh.coordinates.get(mesh.FillElementTypesMap[fill])
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
                if any(isnan(concatenate(kernel_evaluated, axis=0))):
                    diff_prev = point - points_array[step - 1] if step != 0 else None
                    diff_next = point - points_array[step + 1] if step != len(points_array) - 1 else None
                    kernel_evaluated = self.get_principle_value(
                        _kernel=kernel_evaluated,
                        diff_prev=diff_prev,
                        diff_next=diff_next,
                        dim=self.dim
                    )
                    for i in range(self.dim):
                        row[i].extend(array(kernel_evaluated[i]))

                else:
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
    

class CrossProductOperator(BaseLinearOperator):

    def __init__(
        self,
        mesh: Mesh,
        left_vector: VectorField, 
        dim: int | None=3,
        mesh_matrix: Tensor | None=None,
        placement: Literal["centers", "nodes"] | None="centers",
        fill: Literal["vtx", "edge", "boundary", "domain"] | None="domain",
    ):
        kernel = self.get_kernel(
            mesh=mesh,
            left_vector=left_vector,
            placement=placement,
            fill=fill,
        )
        super().__init__(
            kernel=kernel,
            mesh=mesh,
            dim=dim,
            mesh_matrix=mesh_matrix,
        )
    
    @staticmethod
    def get_kernel(
        mesh: Mesh,
        left_vector: VectorField,
        placement: Literal["centers", "nodes"] | None="centers",
        fill: Literal["vtx", "edge", "boundary", "domain"] | None="domain",
    ) -> Any:
        signature = [
            [(0,0), (-1,2), (1,1)],
            [(1,2), (0,1), (-1,0)],
            [(-1,1), (1,0), (0,2)]
        ]

        if mesh is not left_vector.mesh:
            raise AttributeError(name="Meshes are different")
        
        if placement == "centers":
            points_array = mesh.elements_centers.get(mesh.FillElementTypesMap[fill])
        elif placement == "nodes":
            points_array = mesh.coordinates
        else:
            raise AttributeError(name="placement type error")
        
        if len(points_array) != len(left_vector.coorrdinates) != 0:
            raise AttributeError("Coordicates of placement don't match coordinates of the lest vector")
        
        point_value_map = {}        
        for point, value in zip(points_array, left_vector.values):
            block = [
                [
                    signature[ei][ej][0] * value[signature[ei][ej][1]] 
                    for ej in range(mesh.dim)
                ]
                 for ei in range(mesh.dim)
            ]
            point_value_map[tuple(point)] = array(block)
        
        _kernel = lambda p: point_value_map[tuple(p)]

        return _kernel
    

    def to_mesh_matrix(
        self,
        mesh: Mesh | None=None,
        placement: Literal["centers", "nodes"] | None="centers",
        fill: Literal["vtx", "edge", "boundary", "domain"] | None="domain",
    ) -> Tensor:
        start = time()
        if mesh is None:
            mesh = self.mesh

        if placement == "centers":
            points_array = mesh.elements_centers.get(mesh.FillElementTypesMap[fill])
        elif placement == "nodes":
            points_array = mesh.coordinates.get(mesh.FillElementTypesMap[fill])
        else:
            raise AttributeError(name="placement type error")
        
        matrix = zeros(shape=(self.dim * len(points_array), self.dim * len(points_array)))
        _len = len(points_array)
        print(f"Begin placement of operator on mesh elements' {placement}.")
        for step, point in enumerate(points_array):
            kernel_evaluated = self.kernel(point)
            for ei in range(self.dim):
                for ej in range(self.dim):
                    matrix[step * self.dim + ei][step * self.dim + ej] = kernel_evaluated[ei][ej]
            
            sys.stdout.write(f"\rProgress: {round(100 * step / _len, 2)}%")
            sys.stdout.flush()
        print('\n')
        finish = time()
        print(f"Mesh matric of the operator generated in {finish - start} seconds.")
        
        self.mesh_matrix = Tensor(matrix)
        return self.mesh_matrix

        

