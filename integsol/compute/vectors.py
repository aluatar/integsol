from integsol.base import BaseClass
from integsol.mesh.mesh import Mesh
from integsol.mesh.mesh_validators import (
    is_point_in_placement,
    is_mesh_filled,
)

from typing import (
    Literal,
    Iterable,
)
from numpy import (
    array,
    float64,
)
import numpy as np
from typing import Any
from torch import Tensor

class VectorField(BaseClass):

    def __init__(
        self,
        mesh: Mesh | None=None,
        coordinates: Any | None=None,
        values: Any | None=None,
        dim: int | None=3,
        values_on_mesh: Iterable | None=None,
        vectorized: tuple[Iterable, Tensor] | None=None,
    ):
        self.mesh = mesh
        self.dim = dim

        if coordinates is not None:
            self.coorrdinates = coordinates
        elif coordinates is None and self.mesh is not None:
            self.coorrdinates = self.mesh.coordinates
        else:
            self.coorrdinates = np.zeros(shape=(1, self.dim))

        self.values = values if values is not None else np.zeros(shape=(len(self.coorrdinates), self.dim))
        self.values_on_mesh = values_on_mesh
        self.vectorized = vectorized

    @classmethod
    def read(
        cls, 
        path: str, 
        dim: int | None=3,
        #################
        #   OPTIONAL    #
        #################
        comment_char: str | None="%",

    ): 
        with open(path, 'r') as f:
            field = f.read()
        field = field.splitlines()

        line_i = 0
        while field[line_i][0] == comment_char:
            line_i += 1 
        
        coordinates = []
        values = []
        for line_i in range(line_i, len(field)):
            line = field[line_i].split(" ")
            point_and_value = [float64(v) for v in line if v != ""]
            point = point_and_value[:3]
            value = point_and_value[3:]
            
            values.append(value)
            coordinates.append(point)

        return VectorField(
            coordinates=array(coordinates),
            values=array(values),
            dim=dim
        )
        
    @classmethod
    def read_to_mesh(
        cls,
        path: str,
        mesh: Mesh,
        dim: int | None=3,
        placement: Literal["centers", "nodes"] | None="centers",
        fill: Literal["vtx", "edge", "boundary", "domain"] | None="domain",
        #################
        #   OPTIONAL    #
        #################
        comment_char: str | None="%",

    ):
        with open(path, 'r') as f:
            field = f.read()
        field = field.splitlines()

        line_i = 0
        while field[line_i][0] == comment_char:
            line_i += 1 
        
        coordinates = []
        values = []
        for line_i in range(line_i, len(field)):
            line = field[line_i].split(" ")
            point_and_value = [float64(v) for v in line if v != ""]
            point = point_and_value[:3]
            value = point_and_value[3:]
            
            values.append(value)
            coordinates.append(point)
        

        vector =  VectorField(
            coordinates=array(coordinates),
            values=array(values),
            dim=dim,
            mesh=mesh,
        )
        vector.place_on_mesh(
            mesh=mesh,
            placement=placement,
            fill=fill,
            comment_char=comment_char
        )

        return vector
    

    def place_on_mesh(
        self,
        mesh: Mesh,
        placement: Literal["centers", "nodes"] | None="centers",
        fill: Literal["vtx", "edge", "boundary", "domain"] | None="domain",
        #################
        #   OPTIONAL    #
        #################
        comment_char: str | None="%",

    ):
        self.mesh = mesh        
        
        if placement == "centers":
            points_array = mesh.elements_centers
        elif placement == "nodes":
            points_array = mesh.coordinates
        else:
            raise AttributeError(name="placement type error")
        
        value_at_point = []
        for point, value in zip(self.coorrdinates, self.values):
            if is_point_in_placement(
                point=point,
                array=points_array,
                placement=placement,
                fill=fill
            ):
                value_at_point.append(value)
        
        if not is_mesh_filled(
            values=value_at_point,
            array=points_array,
            placement=placement,
            fill=fill
        ):
            raise Exception
        
        self.values_on_mesh = np.array(value_at_point)
    

    def vectorize(
        self
    ) -> tuple[array, Tensor]:
        vectorized_points, vectorized_values = [], []
        for point, value in zip (self.coorrdinates, self.values):
            for component in value:
                vectorized_points.append(point)
                vectorized_values.append(component)
        
        self.vectorized = (array(vectorized_points), Tensor(vectorized_values))

        return self.vectorized
    
    def devectorize(
        self,
        values: Iterable | None=None,
    ) -> None:
        if values is None:
            values = self.vectorized
        dim = self.dim
        if len(values) % 3 != 0:
            raise Exception
        
        devectorized_values = []
        for i in range(len(values) // dim):
            devectorized_values.append(
                [values[dim*i], values[dim*i + 1], values[dim*i + 2]]
            )

        self.values = np.array(devectorized_values)



        


    
