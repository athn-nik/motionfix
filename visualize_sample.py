from src.render.mesh_viz import render_motion
from src.model.utils.tools import pack_to_render
from src.render.video import get_offscreen_renderer
import os 
import argparse
import numpy as np
import torch
def main(path_to_motion):
    # GET A RENDERER
    r = get_offscreen_renderer('./data/body_models') # this path if you followed my setup should be data/body_models
    gen_motion = np.load(path_to_motion, allow_pickle=True).item()['pose']
    # Load the npy and get the translation aka trans_can and the rotations aka rots_can
    # CREATE A DICT TO GIVE TO RENDERING FUNCTION
    smpl_params = pack_to_render(trans=torch.from_numpy(gen_motion[..., :3]), 
                                 rots=torch.from_numpy(gen_motion[..., 3:]))
    # RENDER A MOTION AND SAVE VIDEO! [Do not put a suffix just the name!]
    # you can change colors and put text on top of it.
    render_motion(r, smpl_params, pose_repr='aa',
                              filename='output')

    # If you want to render them overlaid motion1 motion2 you can do:
    #smpl_params1 = pack_to_render(trans=trans_can1, rots=rots_can1, pose_repr='aa')
    #smpl_params2 = pack_to_render(trans=trans_can2, rots=rots_can2, pose_repr='aa')

    # RENDER A MOTION AND SAVE VIDEO!
    #render_motion(r, [smpl_params1, smpl_params2], pose_repr='aa', colors = [color1, color2]
    #                            filename='output.mp4', text_for_vid='if you want to put some text')

    # to give the correct colors look at src/utils/art_utils.py


if __name__ == "__main__":
    os.system("Xvfb :12 -screen 1 640x480x24 &")
    os.environ['DISPLAY'] = ":12"
    parser = argparse.ArgumentParser(description="Read motions from path1 and write to path2.")
    parser.add_argument("--path", type=str, help="Path to the input JSON file")
    # parser.add_argument("path2", type=str, help="Path to the output JSON file")
    args = parser.parse_args()
    main(args.path)
