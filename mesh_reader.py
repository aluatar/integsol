import meshio
import numpy as np
from numpy import (
    array,
    int32,
    float128,
)
from .base import BaseClass


class Mesh(BaseClass):
    verteses: list | None = None
    volume_elements: list[list] | None = None
    boundary_elements: list[list] | None = None

    @staticmethod
    def get_mph_mesh_coordinates(
        file: list[str],
        coordinates_header: str | None="coordinates",
        comment_char: str | None="#" 
    ) -> array:
        # find vertecies coordinates entry
        entry_i = 0
        while coordinates_header not in file[entry_i]:
            entry_i += 1

        #collect coordinates into numpy array
        line_i = entry_i + 1
        coordinates = []
        while not (file[line_i] == "" or comment_char in file[line_i]):
            line = file[line_i].split(" ")
            print(line)
            coordinates.append(
                [float128(v) for v in line if v != ""]
            )
            line_i += 1

        return np.array(coordinates)
    
    @staticmethod
    def get_mph_mesh_elements(
        file: list[str],
        elements_type: int,
        elements_header: str | None="Elements"
    ) -> array:
        pass

    @classmethod
    def read(path: str, **kwargs) -> dict[str, array]:
        if path.split(".")[-1] == "mphtxt":
            #read .mphtxt file
            with open(path, 'r') as _m:
                mesh_file = _m.read()
            mesh_file = mesh_file.splitlines()

             
