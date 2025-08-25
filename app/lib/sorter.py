import trimesh
import numpy as np

from .load import load_translation

def sort_by_size(meshes):

    sizes = []
    for mesh in meshes:
        _, extent = trimesh.bounds.oriented_bounds(mesh, angle_digits=3)
        sizes.append(np.prod(extent))

    return np.argsort(sizes)[::-1]

def sort_by_dist(assembly_dir):
    dists = load_translation(assembly_dir)
    return np.argsort([np.linalg.norm(dist) for dist in dists])[::-1]