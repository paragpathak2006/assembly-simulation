import os
import json
import numpy as np

from .color import get_joint_color, get_multi_color

def arr_to_str(arr):
    return ' '.join([str(x) for x in arr])

def get_xml_string(assembly_dir, move_id, still_ids, move_joint_type, body_type, sdf_dx, col_th, save_sdf):
    with open(os.path.join(assembly_dir, 'translation.json'), 'r') as fp:
        translation = json.load(fp)
    body_type = body_type.upper()
    get_color = get_joint_color if len(translation.keys()) <= 2 else get_multi_color
    sdf_args = 'load_sdf="true" save_sdf="true"' if save_sdf else ''
    string = f'''
<redmax model="assemble">
<option integrator="BDF1" timestep="1e-3" gravity="0. 0. 1e-12"/>
<render bg_color="255 255 255 255"/>

<default>
    <general_{body_type}_contact kn="1e6" kt="1e3" mu="0" damping="0"/>
</default>
'''
    for part_id in [move_id, *still_ids]:
        joint_type = move_joint_type if part_id == move_id else 'fixed'
        string += f'''
<robot>
    <link name="part{part_id}">
        <joint name="part{part_id}" type="{joint_type}" axis="0. 0. 0." pos="{arr_to_str(translation[str(part_id)])}" quat="1 0 0 0" frame="WORLD" damping="0"/>
        <body name="part{part_id}" type="{body_type}" filename="{assembly_dir}/{part_id}.obj" {sdf_args} pos="0 0 0" quat="1 0 0 0" scale="1 1 1" transform_type="OBJ_TO_JOINT" density="1" dx="{sdf_dx}" col_th="{col_th}" mu="0" rgba="{arr_to_str(get_color(part_id))}"/>
    </link>
</robot>
'''
    string += f'''
<contact>
'''
    for part_id in still_ids:
        string += f'''
    <general_{body_type}_contact general_body="part{move_id}" {body_type}_body="part{part_id}"/>
    <general_{body_type}_contact general_body="part{part_id}" {body_type}_body="part{move_id}"/>
'''
    string += f'''
</contact>
</redmax>
    '''
    return string


def unit_vector(vector):
    norm = np.linalg.norm(vector)
    if np.isclose(norm, 0):
        random_vector = np.random.random(len(vector))
        return random_vector / np.linalg.norm(random_vector)
    else:
        return vector / norm