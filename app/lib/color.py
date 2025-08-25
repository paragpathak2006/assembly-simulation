'''
Predefined colors for assembly meshes
'''

import numpy as np


def get_joint_color(index, normalize=True):
    '''
    Get color for 2-part assembly
    '''
    index = int(index)
    assert index in [0, 1]
    colors = np.array([
        [107, 166, 161, 255],
        [209, 184, 148, 255],
    ], dtype=int)
    if normalize: colors = colors.astype(float) / 255.0
    return colors[int(index)]


def get_multi_color(index, normalize=True):
    '''
    Get color for multi-part assembly
    '''
    index = int(index)
    colors = np.array([
        [0, 130, 200, 255], # Blue
        [230, 25, 75, 255], # Red
        [60, 180, 75, 255], # Green
        [170, 110, 40, 255], # Brown
        [145, 30, 180, 255], # Purple
        [128, 128, 128, 255], # Grey
        [0, 0, 0, 255], # Black
        [255, 255, 25, 255], # Yellow
        [70, 240, 240, 255], # Cyan
        [128, 0, 0, 255], # Maroon
    ], dtype=int)
    if normalize: colors = colors.astype(float) / 255.0
    return colors[index % 10]

def get_color_string(index):
    index = int(index)
    color_strings = [
        "Blue",
        "Red",
        "Green",
        "Brown",
        "Purple",
        "Grey",
        "Black",
        "Yellow",
        "Cyan",
        "Maroon",
    ]
    return color_strings[index % 10]
