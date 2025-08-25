import sys
sys.path.append('/usr/lib/freecad-daily/lib')

from collections import deque
from collections.abc import Iterable

import FreeCAD as App
import Part as Part
import Import
import MeshPart
import numpy as np
import os
import shutil
import json
from argparse import ArgumentParser
import logging

def run_conversion(filepath, output_path):
    ####################
    # Initial Conversion
    ####################

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        shutil.rmtree(output_path)
        os.makedirs(output_path)

    Import.open(filepath)
    doc = App.ActiveDocument

    root_objects = [obj for obj in doc.Objects if not obj.InList]
    part_hierarchy = []
    index_hierarchy = []
    flat_parts = []
    current_index = 0

    for root_obj in root_objects:
        if hasattr(root_obj, "Group"):
            part_list, index_list, flat_parts = get_part_hierarchy(root_obj, current_index, flat_parts)
            part_hierarchy.append(part_list)
            index_hierarchy.append(index_list)
            current_index = list(fully_flatten(index_list))[-1] + 1
        else:
            part_hierarchy.append(root_obj)
            flat_parts.append(root_obj)
            index_hierarchy.append(current_index)
            current_index += 1

    logging.info("Index Hierarchy: " + str(index_hierarchy))

    while len(index_hierarchy) == 1:
        if (isinstance(index_hierarchy[0], list)):
            part_hierarchy = part_hierarchy[0]
            index_hierarchy = index_hierarchy[0]
        else:
            return None, None, index_hierarchy
    
    subassembly_graph, subassembly_order = hierarchy_to_graph_and_order(index_hierarchy)
    logging.info("Subassemblies: " + str(subassembly_order))

    max_dimension_threshold = 2.0
    largest_dimension = 0
    for obj in flat_parts:
        bounding_box = obj.Shape.BoundBox
        current_largest_dimension = max(bounding_box.XLength, bounding_box.YLength, bounding_box.ZLength)
        largest_dimension = max(largest_dimension, current_largest_dimension)
    # Set the scale such that the largest dimension among all parts is within the threshold
    scale = max_dimension_threshold / largest_dimension

    names_from_doc = []
    assembled_translations = {}
    for part_idx, part in enumerate(flat_parts):
        # scale part appropriately
        mat = App.Matrix()
        mat.scale(scale, scale, scale)
        scaled_shape = part.Shape.transformGeometry(mat)

        names_from_doc.append(part.Label)

        scaled_translated_part = part.Document.addObject("Part::Feature", "ScaledTranslatedPart" + str(part_idx))
        scaled_translated_part.Shape = scaled_shape
        scaled_translated_part.Placement.Rotation = part.Placement.Rotation
        base = [scaled_translated_part.Placement.Base.x * scale, scaled_translated_part.Placement.Base.y * scale, scaled_translated_part.Placement.Base.z * scale]
        assembled_translations[part_idx] = base
        scaled_translated_part.Placement.Base -= part.Placement.Base * scale
        part_file = os.path.join(output_path, str(part_idx) + '.obj')
        MeshPart.meshFromShape(scaled_translated_part.Shape, 0.001).write(part_file)
        logging.info("Part " + str(part_idx) + " saved")

    doc.recompute()

    with open(os.path.join(output_path, "translation.json"), 'w') as trans_file:
        trans_file.write(json.dumps(assembled_translations))

    ####################
    # Subassembly saving
    ####################

    for subassembly_idx, subassembly in enumerate(subassembly_order):
        subassembly_path = os.path.join(output_path, str(subassembly_idx))
        if not os.path.exists(subassembly_path):
            os.makedirs(subassembly_path)
        else:
            shutil.rmtree(subassembly_path)
            os.makedirs(subassembly_path)
        translations = {}
        for idx in range(len(subassembly)):
            translations[idx] = [0, 0, 0]
        with open(os.path.join(subassembly_path, "translation.json"), 'w') as trans_file:
            trans_file.write(json.dumps(translations))

    part_name_to_indices_dict = {}
    # Combine desired parts into one shape
    for subassembly_idx, subassembly in enumerate(subassembly_order):
        for subpart_idx, subpart in enumerate(subassembly):
            part_name = ''
            if len(subpart) == 1:
                part_name = str(subpart[0] + 1) + "_" + names_from_doc[subpart[0]]
            else:
                part_name = 'combined_' + '_'.join([str(part_idx + 1) for part_idx in subpart])
            part_name_to_indices_dict[part_name] = subpart
            # Save combined shape as OBJ file
            logging.info("Combining parts " + str([idx for idx in range(len(flat_parts)) if idx in subpart]) + " into subpart " + str(subpart_idx) + " for subassembly " + str(subassembly_idx))
            input_files = [os.path.join(output_path, str(part_idx) + '.obj') for part_idx in range(len(flat_parts)) if part_idx in subpart]
            merge_translations = [assembled_translations[part_idx] for part_idx in range(len(flat_parts)) if part_idx in subpart]
            output_file = os.path.join(output_path, str(subassembly_idx), str(subpart_idx) + '.obj')
            merge_obj_files(input_files, merge_translations, output_file)

    part_names = []
    part_name_indices = []
    sorted_keys = sorted(list(part_name_to_indices_dict.keys()))
    resorted_keys = sorted(sorted_keys[:len(flat_parts)], key=lambda x: int(x.split('_')[0])) + sorted(sorted_keys[len(flat_parts):], key=len)
    for idx, part_name in enumerate(resorted_keys):
        if idx < len(flat_parts):
            part_names.append('_'.join(part_name.split('_')[1:]))
        else:
            part_names.append(part_name)
        part_name_indices.append(part_name_to_indices_dict[part_name])

    return subassembly_order, part_names, part_name_indices


def get_part_hierarchy(obj, current_index, flat_shapes, depth=0):
    shape_list = []
    index = current_index
    index_list = []

    for child_obj in obj.Group:
        if hasattr(child_obj, "Group"):
            child_shape_list, child_index_list, flat_shapes = get_part_hierarchy(child_obj, index, flat_shapes, depth=depth+1)
            index = list(fully_flatten(child_index_list))[-1] + 1
            if len(child_index_list) == 1:
                child_index_list = child_index_list[0]
                child_shape_list = child_shape_list[0]
            shape_list.append(child_shape_list)
            index_list.append(child_index_list)
        elif child_obj.isDerivedFrom("Part::Feature"):
            if child_obj.Shape.Volume > 0:
                shape_list.append(child_obj)
                index_list.append(index)
                flat_shapes.append(child_obj)
                index += 1

    return shape_list, index_list, flat_shapes

def fully_flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from fully_flatten(x)
        else:
            yield x

def hierarchy_to_graph_and_order(hierarchy):
    def build_graph_and_order(graph, order, node, depth):
        if isinstance(node, list):
            depth += 1
            children = []
            for child in node:
                child_id = build_graph_and_order(graph, order, child, depth)
                children.append(child_id)
            node_tuple = tuple(children)
            graph[node_tuple] = children
            if depth not in order:
                order[depth] = []
            order[depth].append(children)
            return node_tuple
        else:
            return node

    def flatten(node):
        if isinstance(node, list):
            result = []
            for child in node:
                result.extend(flatten(child))
            return result
        elif isinstance(node, tuple):
            result = []
            for child in node:
                result.extend(flatten(child))
            return result
        else:
            return [node]

    graph = {}
    order = {}
    root_id = build_graph_and_order(graph, order, hierarchy, 0)
    graph[root_id] = hierarchy

    assembly_order = []
    for depth in sorted(order.keys(), reverse=True):
        for item in order[depth]:
            combined = []
            for x in item:
                if isinstance(x, int):
                    combined.append([x])
                elif isinstance(x, tuple) and x in graph:
                    combined.extend([flatten(graph[x])])
            assembly_order.append(combined)

    return graph, assembly_order

def merge_obj_files(input_files, translations, output_file):
    vertex_offset = 0
    uv_offset = 0
    normal_offset = 0

    with open(output_file, 'w') as outfile:
        for input_file, translation in zip(input_files, translations):
            with open(input_file, 'r') as infile:
                for line in infile:
                    if line.startswith('v '):
                        parts = line.strip().split()
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        x += translation[0]
                        y += translation[1]
                        z += translation[2]
                        outfile.write(f"v {x} {y} {z}\n")
                    elif line.startswith('vt '):
                        outfile.write(line)
                    elif line.startswith('vn '):
                        outfile.write(line)
                    elif line.startswith('f '):
                        parts = line.strip().split()
                        new_parts = ['f']
                        for part in parts[1:]:
                            indices = part.split('/')
                            new_indices = []
                            if indices[0]:
                                v = int(indices[0]) + vertex_offset
                                new_indices.append(str(v))
                            else:
                                new_indices.append('')
                            if len(indices) > 1:
                                if indices[1]:
                                    vt = int(indices[1]) + uv_offset
                                    new_indices.append(str(vt))
                                else:
                                    new_indices.append('')
                            if len(indices) > 2:
                                if indices[2]:
                                    vn = int(indices[2]) + normal_offset
                                    new_indices.append(str(vn))
                                else:
                                    new_indices.append('')
                            new_parts.append('/'.join(new_indices))
                        outfile.write(' '.join(new_parts) + '\n')
            vertex_count = sum(1 for line in open(input_file) if line.startswith('v '))
            vertex_offset += vertex_count
            uv_count = sum(1 for line in open(input_file) if line.startswith('vt '))
            uv_offset += uv_count
            normal_count = sum(1 for line in open(input_file) if line.startswith('vn '))
            normal_offset += normal_count


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    subassemblies, part_names, part_names_indices = run_conversion(args.filename, args.output_dir)
    for name, indices in zip(part_names, part_names_indices):
        logging.info("Name: " + str(name) + ", Indices: " + str(indices) + "\n")