import os
import shutil
from pathlib import Path

class SimRenderer:
    @staticmethod
    def replay(sim, record = False, record_path = None):
        if record:
            temp_folder_name = os.path.basename(record_path) + '_tmp'
            record_folder = os.path.join(Path(record_path).parent, temp_folder_name)
            os.makedirs(record_folder, exist_ok = True)
            sim.viewer_options.record = True
            sim.viewer_options.record_folder = record_folder
            loop = sim.viewer_options.loop
            infinite = sim.viewer_options.infinite
            sim.viewer_options.loop = False
            sim.viewer_options.infinite = False
        
        sim.replay()

        if record:
            images_path = os.path.join(record_folder, r"%d.png")
            palette_path = os.path.join(record_folder, 'palette.png')
            os.remove(os.path.join(record_folder, "0.png"))
            os.system("ffmpeg -y -i {} -vf palettegen {}".format(images_path, palette_path))
            os.system("ffmpeg -y -i {} -i {} -lavfi paletteuse {}".format(images_path, palette_path, record_path))

            shutil.rmtree(record_folder)

            sim.viewer_options.record = False
            sim.viewer_options.loop = loop
            sim.viewer_options.infinite = infinite

    @staticmethod
    def replay_all(sims):
        i = 0
        continue_sim = True
        next_inc = sims[0].replay_next(sims[0], True)
        if next_inc == 0:
            continue_sim = False
        i += next_inc
        while continue_sim:
            if i < 0:
                i = 0
                next_inc = 0
            if i >= len(sims):
                i = len(sims) - 1
                next_inc = 0
            
            next_inc = sims[i].replay_next(sims[i - next_inc], False)
            if next_inc == 0:
                continue_sim = False
            i += next_inc
            

            
