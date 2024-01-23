import sched
import sys
from rich import print
import time
import subprocess
import re
import argparse

# the list() things is a hack to avoid by reference assignment
def end_script(no_of_cmds):
    from inspect import currentframe, getframeinfo
    frameinfo = getframeinfo(currentframe())
    sys.exit(f'\nNumber of commands executed: {no_of_cmds}\nExited after line {frameinfo.filename} {frameinfo.lineno}')

def run(cmd):
    print(f"Executing: {cmd}")
    x = subprocess.run(cmd)

def main_loop(command, exp_paths,
              text_guidance_vals,
              motion_guidance_vals, schedulers,
              init_from, condition_modes, data, steps_nos):
    
    cmd_no=0
    cmd_sample = command
    from itertools import product
    exp_grid = list(product(exp_paths, 
                           schedulers,
                           text_guidance_vals,
                           motion_guidance_vals,
                           init_from,
                           condition_modes,
                           data,
                           steps_nos))
    # for fd in exp_paths:
    #     for sched in schedulers:
    #         for gd_t in text_guidance_vals:
    #             for gd_m in motion_guidance_vals:
    #                 for in_lat in init_from:
    #                     for cond_mode in condition_modes:
    #                         for stp_no in steps_nos:
    print("Number of different experiments is:", len(exp_grid))
    print('---------------------------------------------------')
    # exit()
    ckt_name = 'last'
    for fd, sched, gd_t, gd_m, in_lat, cond_mode, data_type, stp_no in exp_grid:
        cur_cmd = list(cmd_train)
        idx_of_exp = cur_cmd.index("FOLDER")
        cur_cmd[idx_of_exp] = str(fd)
        
        list_of_args = ' '.join([f"condition_mode={cond_mode}",
                                 f"init_from={in_lat}",
                                 f"ckpt_name={ckt_name}",
                                 f"guidance_scale_text={gd_t}",
                                 f"guidance_scale_motion={gd_m}",
                                 f"model/infer_scheduler={sched}",
                                 f"data={data_type}",
                                 f"steps={stp_no}"])
        cur_cmd.extend([list_of_args])
        run(cur_cmd)
        time.sleep(0.2)
        cmd_no += 1

    end_script(cmd_no)


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser()

    # parser.add_argument('--mode', required=True, type=str,
    #                    help='what to do')

    parser.add_argument('--bid', required=False, default=30, type=int,
                        help='bid money for cluster')
    parser.add_argument(
            "--runs",
            nargs="*",  # expects arguments
            type=str,
            default=[],  # default list if no arg value
        )
    args= parser.parse_args()
    bid_for_exp = args.bid
    subdirs = args.runs
    # mode = args.mode

    # put base directory
    # base_dir = Path(f'experiments/motionfix-sigg/')
    # print('The base directory is:', str(base_dir))
    
    exp_paths = subdirs
    print('The current folders are---->\n', '\n'.join(subdirs))
    print('---------------------------------------------------')
    parser = argparse.ArgumentParser()
    cmd_train = ['python', 'cluster/single_run.py',
                '--folder', 'FOLDER',
                '--mode', 'eval',
                '--prog', 'motionfix_evaluate',
                '--bid', '20',
                '--extras']
    gd_text = [1.0]
    gd_motion = [2.5, 1.0]
    schedulers = ['ddpm']
    init_from = ['noise', 'source']
    condition_modes = ['full_cond'] #, 'mot_cond', 'text_cond']
    steps_size = [1000] #, 'mot_cond', 'text_cond']
    data = ['sinc_synth', 'bodilex']
    main_loop(cmd_train, exp_paths, gd_text,
              gd_motion, schedulers, init_from, 
              condition_modes, data, steps_size)
