import meshio
import numpy as np
from numpy import (
    array,
    int32,
    float128,
    cross,
    dot
)
from numpy.linalg import (
    norm,
        
)
from typing import Iterable
from .base import BaseClass
import time

def element_dim_validate(dim: int, elements: Iterable):
    bool_arr = [
        len(e) == dim + 1 if isinstance(e,Iterable) else False 
        for e in elements
        ]
    return all(bool_arr)

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




class Mesh(BaseClass):
    def __init__(
            self,
            pref: str | None='nm',
            coordinates: Iterable | None=None,
            elements: dict[str, array] | None=None,
            elements_coordinates: dict[str, array] | None=None,
            elements_measures: dict[str, array] | None=None,
            elements_centers: dict[str, array] | None=None,
    ):
        self.coordinates = coordinates if coordinates is None else array(coordinates)
        self.elements = elements

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
        while not (file[line_i] == "" or comment_char in file[line_i]):
            line = file[line_i].split(" ")
            coordinates.append(
                [float128(v) for v in line if v != ""]
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
            while not (file[line_i] == "" or comment_char in file[line_i]):
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
        for type in elements_coordinates:
            print(f"Calculate measures for {type} type of elements")
            if element_dim_validate(dim=0, elements=elements_coordinates[type]):
                continue
            element_measures = []
            for element in elements_coordinates[type]:
                element_vectors = [
                    np.array(element[0]) - np.array(element[i])
                    for i in range(1,len(element))
                ]

                if element_dim_validate(1,elements=elements_coordinates[type]):
                    measure = oneD_measure(vectors=element_vectors)
                elif element_dim_validate(2,elements=elements_coordinates[type]):
                    measure = twoD_measure(vectors=element_vectors)
                elif element_dim_validate(3,elements=elements_coordinates[type]):
                    measure = threeD_measure(vectors=element_vectors, type=type)

                element_measures.append(measure)
            
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
            if element_dim_validate(dim=0, elements=elements_coordinates[type]):
                continue
                
            centers = []
            for element in elements_coordinates[type]:
                centers.append(
                    np.sum(element, axis=0) / len(element)
                )
            
            elements_centers[type] = np.array(centers)

        return elements_centers
   
   
    @classmethod
    def read(self, path: str,
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

            coordinates = self.get_mph_mesh_coordinates(
                file=mesh_file,
                coordinates_header=coordinates_header,
                comment_char=comment_char,
            )

            elements = self.get_mph_mesh_elements(
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
        



             
