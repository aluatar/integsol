from integsol.mesh.mesh import Mesh

def is_compatible(
    cover_mesh: Mesh,
    sub_mesh: Mesh,
) -> bool:
    if sub_mesh.coordinates not in cover_mesh.coordinates:
        return False
    if set(sub_mesh.elements.keys()) not in set(cover_mesh.elements.keys()):
        return False
    
    bool_array = [sub_mesh.elements[type] in cover_mesh.elements[type] for type in sub_mesh.elements]
    if not all(bool_array):
        return False
    
    return True