from integsol.validators.mesh_validators import *
import numpy as np
from numpy import (
    array,
    int32,
    float64,
    cross,
    dot
)
from numpy.linalg import (
    norm,    
)
from typing import (
    Iterable,
    Literal,
)
from integsol.base import BaseClass
from dataclasses import dataclass
import time
import sys
from datetime import datetime


def oneD_measure(vectors: Iterable):
    return norm(vectors[0])


def twoD_measure(vectors: Iterable):
    return (1 / 2) * norm(cross(vectors[0], vectors[1]))

def threeD_measure(vectors: Iterable, type: str):
    if type == "tet":
        volume = (1 / 6) * abs(
            dot(
                vectors[0], cross(vectors[1], vectors[2])
            )
        )

    return volume



@dataclass
class Mesh(BaseClass):
    FillElementTypesMap = {
        "vtx": "vtx",
        "edge": "edg",
        "boundary": "tri",
        "domain": "tet",
    }
    def __init__(
            self,
            coordinates: Iterable,
            dim: int | None=3,
            elements: dict[str, array] | None=None,
            elements_coordinates: dict[str, array] | None=None,
            elements_measures: dict[str, array] | None=None,
            elements_centers: dict[str, array] | None=None,
    ):
        self.coordinates = coordinates if coordinates is None else array(coordinates)
        self.elements = elements
        self.values_on_mesh = {}
        self.dim = dim

        if elements_coordinates is None:
            self.elements_coordinates = self.fill_elements_coordinates(
                elements=elements,
                coordinates=coordinates,
            )
        else:
            self.elements_coordinates = elements_coordinates 

        if elements_measures is None:
            self.elements_measures = self.get_measures_of_elements(
                elements_coordinates=self.elements_coordinates
            )
        else:
            self.elements_measures = elements_measures  

        if elements_centers is None:
            self.elements_centers = self.get_center_of_masses(
                elements_coordinates=self.elements_coordinates
            )
        else:
            self.elements_centers = elements_centers  

    @staticmethod
    def get_mph_mesh_coordinates(
        file: list[str],     
        coordinates_header: str,  
        comment_char: str 
    ) -> array:
        # find vertecies coordinates entry
        entry_i = 0
        while coordinates_header not in file[entry_i]:
            entry_i += 1

        #collect coordinates into numpy array
        line_i = entry_i + 1
        coordinates = []
        while file[line_i] != "" and comment_char not in file[line_i]:
            line = file[line_i].split(" ")
            coordinates.append(
                [float64(v) for v in line if v != ""]
            )
            line_i += 1

        return np.array(coordinates)
    
    @staticmethod
    def get_mph_mesh_elements(
        file: list[str],
        elements_type_number: int | None,
        elements_header: str,
        element_type_number_entry: str,
        type_header: str,
        comment_char: str, 
    ) -> dict[str, array]:
        # get number of element types
        if elements_type_number is None:
            line_i = 0
            while element_type_number_entry not in file[line_i]:
                line_i += 1
            line = file[line_i].split(" ")
            elements_type_number = int32(line[0])

        # assembe {Type: Elements} dict
        type_elements_map = {}
        line_i = 0
        for type in range(elements_type_number):
            # first find begining of the Type N partof the file to makesure we are in the right place
            while f"# {type_header} #{type}" not in file[line_i]:
                line_i += 1
            # obtain element type name
            type_name = file[line_i + 2].split(comment_char)[0].split(" ")[1]
            # from this line find Element header
            while elements_header not in file[line_i]:
                line_i += 1
            line_i += 1
            # start fill in Elements array
            elements = []
            while file[line_i] != "" and comment_char not in file[line_i]:
                line = file[line_i].split(" ")
                elements.append(
                    [int32(v) for v in line if v != ""]
                )
                line_i += 1
            
            type_elements_map[type_name] = np.array(elements)
        
        return type_elements_map
    
    @staticmethod
    def fill_elements_coordinates(
        elements: dict[str, array],
        coordinates: array,
    ) -> dict[str, array]:
        
        elements_coordinates = {}
        for type in elements:
            _elements = []
            for element in elements[type]:
                _elements.append(
                    [coordinates[node] for node in element]
                )
            elements_coordinates[type] = array(_elements)
        
        return elements_coordinates
    
    @staticmethod
    def get_measures_of_elements(
        elements_coordinates: dict[str, array]
    ) -> dict[str, array]:
        start = time.time()
        elements_measures: dict = {} 
        print("Begin calculation of elements measures.")
        for type in elements_coordinates:
            print(f"Calculate measures for {type} type of elements")
            if is_element_dim_valide(dim=0, elements=elements_coordinates[type]):
                continue
            element_measures = []
            _len = len(elements_coordinates[type])
            for step, element in enumerate(elements_coordinates[type]):
                element_vectors = [
                    np.array(element[0]) - np.array(element[i])
                    for i in range(1,len(element))
                ]

                if is_element_dim_valide(1,elements=elements_coordinates[type]):
                    measure = oneD_measure(vectors=element_vectors)
                elif is_element_dim_valide(2,elements=elements_coordinates[type]):
                    measure = twoD_measure(vectors=element_vectors)
                elif is_element_dim_valide(3,elements=elements_coordinates[type]):
                    measure = threeD_measure(vectors=element_vectors, type=type)

                element_measures.append(measure)
                sys.stdout.write(f"\rProgress: {round(100 * step / _len, 2)}%")
                sys.stdout.flush()
            print('\n')
            
            elements_measures[type] = np.array(element_measures)
        finish = time.time()
        print(f"Calculation of measures of all elements finished in {finish - start} seconds.")
        return elements_measures
    
    @staticmethod
    def get_center_of_masses(
        elements_coordinates: Iterable,
    ) -> dict[str, array]:
        elements_centers = {}
        for type in elements_coordinates:
            if is_element_dim_valide(dim=0, elements=elements_coordinates[type]):
                continue
                
            centers = []
            for element in elements_coordinates[type]:
                centers.append(
                    np.sum(element, axis=0) / len(element)
                )
            
            elements_centers[type] = np.array(centers)

        return elements_centers
   
   
    @classmethod
    def read(cls, path: str,
             ################
             #   OPTIONAL   # 
             ################
             pref: str | None='nm',
             coordinates_header: str | None="coordinates",
             elements_type_number: int | None=None,
             elements_header: str | None="Elements",
             element_type_number_entry: str | None="# number of element types",
             type_header: str | None="Type",
             comment_char: str | None="#", 
             ):
        
        if path.split(".")[-1] == "mphtxt":
            #read .mphtxt file
            with open(path, 'r') as _m:
                mesh_file = _m.read()
            mesh_file = mesh_file.splitlines()

            coordinates = cls.get_mph_mesh_coordinates(
                file=mesh_file,
                coordinates_header=coordinates_header,
                comment_char=comment_char,
            )

            elements = cls.get_mph_mesh_elements(
                file=mesh_file,
                elements_type_number=elements_type_number,
                elements_header=elements_header,
                element_type_number_entry=element_type_number_entry,
                type_header=type_header,
                comment_char=comment_char
            )

            return Mesh(
                coordinates=coordinates,
                elements=elements)
            

    # ad hoc

    def dump(self, path: str) -> None:
        import json

        _mesh = {
            "metainfo": "Mesh file " + path.split("/")[-1].replace(".json", "") + f" created at {datetime.now()}",
            "coordinates": self.coordinates.tolist(),
            "elements": {
                type: arr.tolist() for type, arr in self.elements.items()
            },
            "elements_coordinates": {
                type: arr.tolist() for type, arr in self.elements_coordinates.items()
            },
            "elements_centers": {
                type: arr.tolist() for type, arr in self.elements_centers.items()
            },
            "elements_measures": {
                type: arr.tolist() for type, arr in self.elements_measures.items()
            }
        }

        print(json.dumps(_mesh))
        with open (path, 'w') as mesh_json:
            mesh_json.write(json.dumps(_mesh))


    def load(path: str):
        import json

        with open(path, 'r') as mesh_file:
            _mesh = json.load(mesh_file)

        coordinates = array(_mesh['coordinates'])
        elements = { type: array(arr) for type, arr in _mesh['elements'].items()}
        element_coordinates = { type: array(arr) for type, arr in _mesh['elements_coordinates'].items()}
        element_centers = { type: array(arr) for type, arr in _mesh['elements_centers'].items()}
        element_measures = { type: array(arr) for type, arr in _mesh['elements_measures'].items()}

        return Mesh(
            coordinates=coordinates,
            elements=elements,
            elements_centers=element_centers,
            elements_coordinates=element_coordinates,
            elements_measures=element_measures,
            dim=3,
        )

        


                 
            




        

                  
        

        



             
