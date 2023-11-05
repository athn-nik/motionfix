import sys
from rich import print
import time
import subprocess
import re
import argparse




# the list() things is a hack to avoid by reference assignment
cmd_no = 0
def end_script(no_of_cmds):
    from inspect import currentframe, getframeinfo
    frameinfo = getframeinfo(currentframe())
    sys.exit(f'\nNumber of commands executed: {no_of_cmds}\nExited after line {frameinfo.filename} {frameinfo.lineno}')

def run(cmd):
    print(f"Executing: {' '.join(re.escape(x) for x in cmd)}")
    x = subprocess.run(cmd)




def main_loop(command, exp_paths,
              text_guidance_vals,
              motion_guidance_vals, schedulers):

    cmd_sample = command

    for sch in schedulers:
        for fd in folders:
            for gd_t in text_guidance_vals:
                for gd_m in motion_guidance_vals:
                    cur_cmd = list(cmd_train)
                    idx_of_exp = cur_cmd.index("FOLDER")
                    cur_cmd[idx_of_exp] = fd
                    
                    if sched == 'ddim':
                        stps = 200
                    else:
                        stps = 100

                    list_of_args = [f" guidance_scale_text={gd_t} guidance_scale_motion={gd_m} model/infer_scheduler={sched} steps={stps} "]
                    cur_cmd.extend(list_of_args)
                    run(cur_cmd)
                    time.sleep(1)
                    cmd_no += 1

    end_script(cmd_no)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--bid', required=False, default=10, type=int,
                        help='bid money for cluster')
    parser.add_argument(
            "--runs",
            nargs="*",  # expects ≥ 0 arguments
            type=str,
            default=[],  # default list if no arg value
        )
    parser.add_argument(
        "--exp",
        type=str,
        default='experiments/space',
    )
    arguments = parser.parse_args()
    bid_for_exp = arguments.bid
    main_fd = args.exp
    subdirs = args.runs

    user_dir = user_dir.replace('/lustre/fast', '')
    'experiments/motion-editing/'
    # put base directory
    base_dir = Path(f'experiments/motion-editing/{main_fd}')
    print('The base directory is:', base_dir)
    exp_paths = [base_dir/subd for subd in subdirs]
    print('The current runs are:', exp_paths)

    parser = argparse.ArgumentParser()
    cmd_train = ['python', 'cluster/single_run.py',
                '--folder', 'FOLDER',
                '--mode', 'sample',
                '--bid', '30',
                '--extras']
    gd_text = [1, 1.5, 2, 2.5, 3]
    gd_motion = [1, 1.5, 2, 2.5, 3]
    schedulers = ['ddim', 'ddpm']

    main_loop(cmd_train, gd_text, exp_paths,
              gd_motion, schedulers)