import sys
from rich import print
import time
import subprocess
import re
import argparse
from tqdm import tqdm 

def run(cmd):
    # print(f"Executing: {' '.join(cmd)}")
    x = subprocess.run(cmd)

def get_guidances(s=1.0, e=5, no=5, t2m=False):
    import itertools
    import numpy as np
    if t2m:
        gd_text = np.linspace(s, e, no, endpoint=True)
        all_combs = [round(c,2) for c in list(gd_text)]
    else:
        gd_motion = np.linspace(s, e, no, endpoint=True)
        gd_text_motion = np.linspace(s, e, no, endpoint=True)

        all_combinations = list(itertools.product(gd_motion, gd_text_motion))
        all_combs = [(round(c[0],2), round(c[1],2)) for c in all_combinations]
    
    return all_combs


def main_loop(command, exp_paths,
              guidance_vals,
              init_from, data):

    cmd_no=0
    cmd_sample = command
    from itertools import product
    exp_grid = list(product(exp_paths,
                            guidance_vals,
                            init_from,
                            data))
    print("Number of different experiments is:", len(exp_grid))
    print('---------------------------------------------------')
    ckt_name = 'last'
    if data[0] != 'hml3d':
        arg1 = 'guidance_scale_text_n_motion'
        arg0 = 'guidance_scale_motion'
        t2m = False
    else:
        arg0 = 'guidance_scale_text'
        arg1 = 'guidance_scale_motion'
        t2m = True
    for fd, gd, in_lat, data_type in tqdm(exp_grid):
        cur_cmd = list(cmd_train)
        idx_of_exp = cur_cmd.index("FOLDER")
        cur_cmd[idx_of_exp] = str(fd)
        if t2m:
            list_of_args = ' '.join([f"init_from={in_lat}",
                                 f"ckpt_name={ckt_name}",
                                 f"{arg1}={gd[1]}",
                                 f"{arg0}={gd[0]}",
                                 f"data={data_type}"])
        else:
            list_of_args = ' '.join([f"init_from={in_lat}",
                                 f"ckpt_name={ckt_name}",
                                 f"{arg1}={gd[1]}",
                                 f"{arg0}={gd[0]}",
                                 f"data={data_type}"])
        cur_cmd.extend([list_of_args])
         
        run(cur_cmd)
        time.sleep(0.01)
        cmd_no += 1
        # import ipdb;ipdb.set_trace()

if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', required=True, type=str,
                        help='what to do')
    parser.add_argument('--ds', required=True, type=str,
                        help='dataset')
    parser.add_argument('--bid', required=False, default=30, type=int,
                        help='bid money for cluster')
    parser.add_argument(
            "--runs",
            nargs="*",  # expects arguments
            type=str,
            default=[],  # default list if no arg value
        )
    args = parser.parse_args()
    bid_for_exp = args.bid
    subdirs = args.runs
    mode = args.mode
    exp_paths = subdirs
    datasets = args.ds
    print('The current folders are---->\n', '\n'.join(subdirs))
    print('---------------------------------------------------')
    assert mode in ['viz_t2m', 'sample_t2m', 'viz', 'sample', 'eval']
    assert datasets in ['bodilex', 'sinc_synth', 'hml3d']
    if mode == 'viz':
        script = 'visualize_motionfix'
    elif mode == 'viz_t2m':
        script = 'visualize_hml3d'
    elif mode in ['sample', 'eval']:
        script = 'motionfix_evaluate'
    else:
        script = 'hml3d_evaluate'
    guidances = get_guidances()
    parser = argparse.ArgumentParser()
    if datasets == 'bodilex':
        bid_for_data = 2000
    cmd_train = ['python', 'cluster/single_run.py',
                '--folder', 'FOLDER',
                '--mode', mode,
                '--prog', script,
                '--bid', '20',
                '--extras']
    init_from = ['noise']
    data = [datasets]
    main_loop(cmd_train, exp_paths, guidances, init_from,data)
