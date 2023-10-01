import sys
from rich import print
import time
import subprocess

cmd_train = ['python', 'cluster/single_run.py',
             '--expname', 'large-grid-search',
             '--mode', 'train',
             '--bid', '40',
             '--run-id', 'RUNID',
             '--gpus', '2',
             '--extras']

batch_sizes = [16, 32]
lrs = [8e-4, 3e-4, 4e-4, 1e-3, 1e-5]

# the list() things is a hack to avoid by reference assignment
cmd_no = 0
def end_script(no_of_cmds):
    from inspect import currentframe, getframeinfo
    frameinfo = getframeinfo(currentframe())
    sys.exit(f'\nNumber of commands executed: {no_of_cmds}\nExited after line {frameinfo.filename} {frameinfo.lineno}')

def run(cmd):
    print(f"Executing: {' '.join(cmd)}")
    x = subprocess.run(cmd)

# SAMPLING
for bs in batch_sizes:
    for cur_lr in lrs:
        cur_cmd = list(cmd_train)
        idx_of_runid = cur_cmd.index("--run-id")
        cur_cmd[idx_of_runid + 1] = f'BatchSize{bs}_LR{cur_lr:.1e}'

        list_of_args = [f'machine.batch_size={bs} model.optim.lr={cur_lr}']
        cur_cmd.extend(list_of_args)
        run(cur_cmd)
        time.sleep(5)
        cmd_no += 1

end_script(cmd_no)
