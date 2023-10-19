import sys
from rich import print
import time
import subprocess
import re
cmd_train = ['python', 'cluster/single_run.py',
             '--expname', 'search-wo-jts-loss',
             '--mode', 'train',
             '--bid', '30',
             '--run-id', 'RUNID',
             '--gpus', '1',
             '--extras']

batch_sizes = [16, 32]
lrs = [8e-4, 3e-4, 1e-4, 1e-3, 1e-5]

# the list() things is a hack to avoid by reference assignment
cmd_no = 0
def end_script(no_of_cmds):
    from inspect import currentframe, getframeinfo
    frameinfo = getframeinfo(currentframe())
    sys.exit(f'\nNumber of commands executed: {no_of_cmds}\nExited after line {frameinfo.filename} {frameinfo.lineno}')

def run(cmd):
    print(f"Executing: {' '.join(re.escape(x) for x in cmd)}")
    x = subprocess.run(cmd)
input_repr = [" 'model.input_feats=[body_transl, body_orient, body_pose]' ", " 'model.input_feats=[body_transl_delta_pelv_xy, body_orient_delta, body_pose_delta]' "]
# SAMPLING
extra_const = 'model.loss_on_positions=false'
for dtype in input_repr:
    for bs in batch_sizes:
        for cur_lr in lrs:
            cur_cmd = list(cmd_train)
            idx_of_runid = cur_cmd.index("--run-id")
            if 'delta_pelv_xy' in dtype:
                repres = 'deltas'
            else:
                repres = 'poses'
            cur_cmd[idx_of_runid + 1] = f'BatchSize{bs}_LR{cur_lr:.1e}_REPR_{repres}'
            list_of_args = [f" {dtype} machine.batch_size={bs} model.optim.lr={cur_lr} {extra_const}"]
            cur_cmd.extend(list_of_args)
            run(cur_cmd)
            time.sleep(2)
            cmd_no += 1

end_script(cmd_no)
