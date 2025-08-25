import numpy as np
import trimesh
from sklearn.cluster import DBSCAN
from collections import defaultdict
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from networkx.algorithms.community import girvan_newman

from .load import load_assembly

def apply_dbscan(subassemblies, part_positions, epsilon, min_samples):
    # Calculate the mean position of each subassembly
    part_positions = [np.mean([part_positions[idx] for idx in subassembly], axis=0) for subassembly in subassemblies]
    # Cluster the subassemblies using DBSCAN
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(part_positions)
    return clustering.labels_

def analyze_clusters(subassemblies, labels):
    # Create a dictionary of clusters where the keys are the cluster labels and the values are lists of subassemblies in that cluster
    cluster_dict = defaultdict(list)
    for idx, label in enumerate(labels):
        if label != -1:
            cluster_dict[label].append(subassemblies[idx])
    return cluster_dict

def recursive_clustering(part_positions, epsilon, min_samples, subassembly_history=None, unassigned_parts=None, max_depth=10, current_depth=0):
    if subassembly_history is None:
        subassembly_history = []

    if unassigned_parts is None:
        unassigned_parts = set(range(len(part_positions)))

    if len(unassigned_parts) == 0 or current_depth >= max_depth:
        return subassembly_history

    subassemblies = [[idx] for idx in unassigned_parts]
    labels = apply_dbscan(subassemblies, part_positions, epsilon, min_samples)
    clusters = analyze_clusters(subassemblies, labels)

    new_assembly = []

    if current_depth == 0:
        for part_idx in list(unassigned_parts):  # Create a copy of unassigned_parts
            new_assembly.append([part_idx])
            unassigned_parts.discard(part_idx)
    else:
        for cluster_id, part_indices in clusters.items():
            new_group = []
            for part_idx in part_indices:
                unassigned_parts.discard(part_idx[0])
                new_group.append(part_idx[0])
            if new_group:
                new_assembly.append(new_group)

    subassembly_history.append(new_assembly)

    if len(unassigned_parts) > 0:
        return recursive_clustering(part_positions, epsilon, min_samples, subassembly_history, unassigned_parts, max_depth, current_depth + 1)
    else:
        return subassembly_history

def proximity_matrix(part_positions, epsilon):
    n_parts = len(part_positions)
    matrix = np.zeros((n_parts, n_parts))
    for i in range(n_parts):
        for j in range(i+1, n_parts):
            distance = np.linalg.norm(part_positions[i] - part_positions[j])
            if distance <= epsilon:
                matrix[i, j] = matrix[j, i] = 1
    return matrix

def apply_girvan_newman(part_positions, epsilon):
    matrix = proximity_matrix(part_positions, epsilon)
    graph = nx.from_numpy_matrix(matrix)
    communities = list(girvan_newman(graph))
    return communities


def generate_subassemblies(source_dir, type='DBSCAN', epsilon_percentile=10, min_samples=3):
    # Load objs
    meshes, names = load_assembly(source_dir, translate=False, return_names=True)

    # part_positions = [mesh.mass_properties["center_mass"] for mesh in meshes]
    # distances = pdist(part_positions)
    # epsilon = np.percentile(distances, epsilon_percentile)  # Adjust the percentile value as needed

    # if type == 'DBSCAN':
    #     return recursive_clustering(part_positions, epsilon, min_samples)

    # elif type == 'GN':
    #     communities = apply_girvan_newman(part_positions, epsilon)
    #     subassemblies = []
    #     for level in communities:
    #         subassembly_level = [[idx] for idx in range(len(part_positions))]
    #         for community in level:
    #             group = []
    #             for part_idx in community:
    #                 group.append(part_idx)
    #                 subassembly_level.remove([part_idx])
    #             subassembly_level.append(group)
    #         subassemblies.append(subassembly_level)
    #     return subassemblies

    # This code is not working but sometimes giving errors. Commenting it out until it works
    return [[[i] for i in range(len(meshes))]]

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--source-dir', type=str, required=True)
    parser.add_argument('--type', type=str, default='DBSCAN')
    parser.add_argument('--epsilon-percentile', type=float, required=False)
    parser.add_argument('--min-samples', type=int, required=False)
    args = parser.parse_args()

    kwargs = {}
    if args.epsilon_percentile is not None:
        kwargs['epsilon_percentile'] = args.epsilon_percentile
    if args.min_samples is not None:
        kwargs['min_samples'] = args.min_samples

    print(generate_subassemblies(args.source_dir, args.type, **kwargs))



