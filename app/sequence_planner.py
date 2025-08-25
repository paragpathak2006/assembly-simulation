import os
os.environ['OMP_NUM_THREADS'] = '1'
import shutil

import numpy as np
import networkx as nx
from time import time
import logging

from .lib.load import load_assembly, load_translation
from .lib.save import clear_saved_sdfs, interpolate_path
from .lib.renderer import SimRenderer
from .lib.sorter import sort_by_size, sort_by_dist

from .path_planner import BFSPlanner as Planner

class SequencePlanner:

    def __init__(self, assembly_dir):

        self.assembly_dir = assembly_dir
        self.assembly_id = os.path.basename(assembly_dir)

        self.graph = nx.DiGraph()
        
        meshes, names = load_assembly(assembly_dir, return_names=True)
        # self.size_order = sort_by_size(meshes)
        # self.dist_order = sort_by_dist(self.assembly_dir)

        part_ids = [int(name.replace('.obj', '')) for name in names]
        for i in range(len(part_ids)):
            self.graph.add_node(part_ids[i])

        self.num_parts = len(part_ids)
        assert self.num_parts > 1
        self.max_seq_count = (1 + self.num_parts) * self.num_parts // 2 - 1

        self.success_status = ['Success', 'Start with goal']
        self.failure_status = ['Timeout', 'Failure']
        self.valid_status = self.success_status + self.failure_status

    def draw_graph(self):
        import matplotlib.pyplot as plt
        nx.draw(self.graph, with_labels=True)
        plt.show()

    def plan_path(self, move_id, still_ids, rotation, body_type, sdf_dx, collision_th, force_mag, frame_skip,
        max_time, max_depth, seed, render, record_path, reverse, save_dir, n_save_state, return_contact=False):

        path_planner = Planner(self.assembly_dir, move_id, still_ids, rotation, body_type, sdf_dx, collision_th, force_mag, frame_skip, save_sdf=True)
        status, t_plan, path = path_planner.plan(max_time, max_depth=max_depth, seed=seed, return_path=True, render=render, record_path=record_path, reverse=reverse, multi=True)

        assert status in self.valid_status, f'unknown status {status}'
        if save_dir is not None:
            path_planner.save_path(path, save_dir, n_save_state)

        if return_contact:
            contact_parts = path_planner.get_contact_bodies(move_id)
            return status, t_plan, contact_parts, path
        else:
            return status, t_plan, path, path_planner

    def plan_sequence(self, rotation, body_type, sdf_dx, collision_th, force_mag, frame_skip, seq_max_time, 
                      path_max_time, seed, render, record_dir, reverse, save_dir, n_save_state, verbose=False, proposed_order=None):

        t_start_total = time()

        np.random.seed(seed)

        if render and record_dir is not None:
            os.makedirs(record_dir, exist_ok=True)

        seq_status = 'Failure'
        sequence = []
        seq_count = 0
        t_plan_all = 0
        paths = []
        path_planners = []
        record_paths = []

        active_queue = []
        if proposed_order is not None:
            active_queue = [(order_idx, 1) for order_idx in proposed_order]
        else:
            active_queue = [(node, 1) for node in self.graph.nodes] # [(id, depth)] nodes going to try
            np.random.shuffle(active_queue)
        inactive_queue = [] # [(id, depth)] nodes tried

        while True:

            all_ids = list(self.graph.nodes)
            move_id, max_depth = active_queue.pop(0)
            still_ids = all_ids.copy()
            still_ids.remove(move_id)

            if record_dir is None:
                record_path = None
            elif render:
                record_path = os.path.join(record_dir, f'{self.assembly_id}', f'{seq_count}_{move_id}.gif')

            if save_dir is not None:
                curr_save_dir = os.path.join(save_dir, f'{self.assembly_id}', f'{seq_count}_{move_id}')
            else:
                curr_save_dir = None

            curr_seed = np.random.randint(self.max_seq_count)

            # Try first without rotation
            status, t_plan, path, planner = self.plan_path(move_id, still_ids,
                False, body_type, sdf_dx, collision_th, force_mag, frame_skip, path_max_time,
                max_depth, curr_seed, render, record_path, reverse, curr_save_dir, n_save_state)
            assert status in self.valid_status
            
            # If this fails, try again with rotation
            if status in self.failure_status and rotation:
                status, t_plan_rot, path, planner = self.plan_path(move_id, still_ids,
                    True, body_type, sdf_dx, collision_th, force_mag, frame_skip, path_max_time, 
                    max_depth, curr_seed, render, record_path, reverse, curr_save_dir, n_save_state)
                t_plan += t_plan_rot

            if reverse:
                if path:
                    path = path[::-1]

            t_plan_all += t_plan
            seq_count += 1
            if status in self.success_status:
                paths.append(path)
                path_planners.append(planner)
                record_paths.append(record_path)

            if verbose:
                logging.info(f'# trials: {seq_count} | Move id: {move_id} | Status: {status} | Current planning time: {t_plan} | Total planning time: {t_plan_all}')
                if status in self.success_status:
                    logging.info("Path: From " + str([round(val+offset,3) for val, offset in zip(path[0][:3], planner.com_move)] + [round(val) for val in path[0][3:]]) + " to " + str([round(val+offset,3) for val, offset in zip(path[-1][:3], planner.com_move)] + [round(val) for val in path[-1][3:]]))

            if status in self.success_status:
                self.graph.remove_node(move_id)
                sequence.append(int(move_id))
            else:
                inactive_queue.append([move_id, max_depth + 1])

            if verbose:
                logging.info('Active queue: ' + str(active_queue))
                logging.info('Inactive queue: ' + str(inactive_queue))

            if len(self.graph.nodes) == 1:
                seq_status = 'Success'
                sequence.append(int(list(self.graph.nodes)[0]))
                paths.append([])
                break

            if len(active_queue) == 0:
                active_queue = inactive_queue.copy()
                inactive_queue = []

            if t_plan_all > seq_max_time:
                seq_status = 'Timeout'
                break

        if reverse:
            sequence = sequence[::-1]

        total_execution_time = time() - t_start_total

        if verbose:
            logging.info(f'Result: {seq_status} | Disassembled: {len(sequence)}/{self.num_parts} | Total # trials: {seq_count} | Total planning time: {t_plan_all}')
            logging.info(f'Sequence: {sequence}')
            logging.info("\n\nTotal Execution Time: " + str(total_execution_time) + "\n\n")

        if render:
            render_paths = paths[:-1]
            if reverse:
                render_paths = render_paths[::-1]
                path_planners = path_planners[::-1]
                record_paths = record_paths[::-1]
            for path, planner, record_path in zip(render_paths, path_planners, record_paths):
                if path is not None:
                    if not reverse:
                        path = path[::-1]
                    planner.sim.set_state_his(path, [np.zeros(planner.ndof) for _ in range(len(path))])
            SimRenderer.replay_all([planner.sim for planner in path_planners])

        return seq_status, sequence, seq_count, paths, t_plan_all
    
def run_sequence_planner(run_dir, id=None, rotation=False, reverse=False, body_type='sdf', sdf_dx=0.05,
                         collision_th=1e-2, force_mag=100, frame_skip=100, seq_max_time=3600, path_max_time=120, 
                         seed=1, render=False, record_dir=None, save_dir=None, n_save_state=100, proposed_order=None):
    subassembly_dir = os.path.join(run_dir, str(id)) if (not id==None) else run_dir
    if rotation: seq_max_time *= 2
        
    clear_saved_sdfs(subassembly_dir)
    seq_planner = SequencePlanner(subassembly_dir)
    _, sequence, _, paths, _ = seq_planner.plan_sequence( 
        rotation, body_type, sdf_dx, collision_th, force_mag, frame_skip,
        seq_max_time, path_max_time, seed, render, record_dir, reverse, save_dir, n_save_state, verbose=True, proposed_order=proposed_order)
    for idx, path in enumerate(paths):
        paths[idx] = interpolate_path(path, n_save_state)
    clear_saved_sdfs(subassembly_dir)

    return sequence, paths

def parallel_sequence_planner(run_dir, subassembly_idx, subassembly, proposed_orders):
    logging.info("Running parallel sequence planner")
    sequence, paths = run_sequence_planner(run_dir, id=subassembly_idx, reverse=True, proposed_order=proposed_orders[subassembly_idx] if proposed_orders else None)
    logging.info("Done running parallel sequence planner")
    shutil.rmtree(os.path.join(run_dir, str(subassembly_idx)))
    return subassembly_idx, subassembly, sequence, [[pos.tolist() for pos in path] for path in paths]

def finish_buildit(sequence_results, run_dir, part_names, part_indices_list):
    output = {}
    output["steps"] = []
    step_counter = 0
    for subassembly_idx, subassembly, sequence, paths in sequence_results:
        assert(len(sequence) == len(paths))
        for step_idx, subpart_idx in enumerate(sequence):
            step = {}
            moveIndices = subassembly[subpart_idx]
            stillIndices = []
            for other_subpart in [other_subpart for other_subpart_idx, other_subpart in enumerate(subassembly) if sequence.index(other_subpart_idx) < step_idx]:
                stillIndices += other_subpart
            if (step_counter == 0):
                step["description"] = "Start with part " + str(part_indices_list.index(moveIndices) + 1) + " (" + part_names[part_indices_list.index(moveIndices)] + ")"
            else:
                step["description"] = "Add part " + str(part_indices_list.index(moveIndices) + 1) + " (" + part_names[part_indices_list.index(moveIndices)] + ") to the assembly"
            step["moveIndices"] = moveIndices
            step["stillIndices"] = stillIndices
            step["paths"] = [paths[::-1][step_idx]]*len(moveIndices)
            step["group"] = subassembly_idx
            output["steps"].append(step)
            step_counter += 1

    coms = load_translation(run_dir)
    output["translations"] = [coms[key].tolist() for key in sorted(coms.keys())]

    output["parts"] = []
    for part_name, part_indices in zip(part_names, part_indices_list):
        output["parts"].append({"name": part_name, "indices": part_indices})

    return output

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--run-dir', type=str, help='directory storing all assemblies')
    parser.add_argument('--id', type=int, default=None, help='assembly id (e.g. 0)')
    parser.add_argument('--rotation', default=False, action='store_true')
    parser.add_argument('--reverse', default=False, action='store_true', help='reverse sequence and paths to assembly instead of disassembly')
    parser.add_argument('--body-type', type=str, default='sdf', choices=['bvh', 'sdf'], help='simulation type of body')
    parser.add_argument('--sdf-dx', type=float, default=0.05, help='grid resolution of SDF')
    parser.add_argument('--collision-th', type=float, default=1e-2)
    parser.add_argument('--force-mag', type=float, default=100, help='magnitude of force')
    parser.add_argument('--frame-skip', type=int, default=100, help='control frequency')
    parser.add_argument('--seq-max-time', type=float, default=3600, help='sequence planning timeout')
    parser.add_argument('--path-max-time', type=float, default=120, help='path planning timeout')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--render', default=False, action='store_true', help='if render the result')
    parser.add_argument('--record-dir', type=str, default=None, help='directory to store rendering results')
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--n-save-state', type=int, default=100)
    args = parser.parse_args()

    sequence, path = run_sequence_planner(args.run_dir, args.id, args.rotation, args.reverse, args.body_type, args.sdf_dx, 
                                          args.collision_th, args.force_mag, args.frame_skip, args.seq_max_time, args.path_max_time,
                                          args.seed, args.render, args.record_dir, args.save_dir, args.n_save_state)
