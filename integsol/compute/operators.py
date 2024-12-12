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
    ones,
    sum,
    float64,
    float128,
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
    def get_dipole_superposition(
        kernel: Any,
        measure: float64 | float128,
        point: Iterable,
        center: Iterable,
        element: Iterable,
        dim: int,
    ) -> Iterable:
        center_kernel = kernel(point, center)
        if not isinstance(center_kernel, Iterable):
            raise Exception
        
        if any(isnan(concatenate(center_kernel, axis=0))):
            center_kernel = ones(shape=(dim,dim)) 
        else:
            center_kernel = center_kernel * measure
        
        nodes_kernel = sum([kernel(point, en) for en in element], axis=0) * measure

        result = center_kernel + nodes_kernel

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
            nodes_array = self.mesh.elements_coordinates.get(mesh.FillElementTypesMap[fill])
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
            for element , point_prime, measure in zip(nodes_array, points_array, measures_array):
                element_dipol_superposition = self.get_dipole_superposition(
                    kernel=self.kernel,
                    measure=measure,
                    point=point,
                    center=point_prime,
                    element=element,
                    dim=self.dim
                ) 

                for i in range(self.dim):
                    row[i].extend(element_dipol_superposition[i])
            
            matrix.extend(row)
            sys.stdout.write(f"\rProgress: {round(100 * step / _len, 2)}%")
            sys.stdout.flush()
        print('\n')
        self.mesh_matrix = Tensor(matrix).T
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

        

