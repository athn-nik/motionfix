import logging
import hydra
import os
from pathlib import Path
import subprocess
import sys
import stat
from loguru import logger
from typing import List, Set, Dict, Tuple, Optional
import shortuuid
import wandb
MODES = ["training", "sample_eval", "render",
         "compile_results", "relaunch_sample"]

SHELL_SCRIPT_FD = 'cluster_scripts'
CONDOR_FD = 'condor_logs'

GPUS = {
        'v100-p16': ('\"Tesla V100-PCIE-16GB\"', 'volta', 16000),
        'v100-p32': ('\"Tesla V100-PCIE-32GB\"', 'volta', 32000),
        'v100-s32': ('\"Tesla V100-SXM2-32GB\"', 'volta', 32000),
        'a100-sm80': ('\"NVIDIA A100-SXM4-80GB\"', 'nvidia', 80000),
        'a100-sxm40': ('\"NVIDIA A100-SXM4-40GB\"', 'nvidia', 40000),
        'quadro6000': ('\"Quadro RTX 6000\"', 'quadro', 24000),
        'nvidia-h100': ('\"NVIDIA H100\"', 'nvidia', 80000),
        #'rtx2080ti': ('\"NVIDIA GeForce RTX 2080 Ti\"', 'rtx', 11000)
        }
    
SUBMISSION_TEMPLATE = f'Description=DESCRIPTION\n' \
                       'executable = RUN_SCRIPT\n' \
                       'arguments = $(Process) $(Cluster)\n' \
                       'error = CNR_LOG_ID/$(Cluster).$(Process).err\n' \
                       'output = CNR_LOG_ID/$(Cluster).$(Process).out\n' \
                       'log = CNR_LOG_ID/$(Cluster).$(Process).log\n' \
                       'request_memory = 128000\n' \
                       'request_cpus=CPUS\n' \
                       'request_gpus=NO_GPUS\n' \
                       '+BypassLXCfs="true"\n' \
                       'requirements=GPUS_REQS\n' \
                       'queue 1'



SUBMISSION_TEMPLATE_SWEEP = f'executable = RUN_SCRIPT\n' \
                             'arguments = $(Process) $(Cluster)\n' \
                             'error = CNR_LOG_ID/$(Cluster).$(Process).err\n' \
                             'output = CNR_LOG_ID/$(Cluster).$(Process).out\n' \
                             'log = CNR_LOG_ID/$(Cluster).$(Process).log\n' \
                             'request_memory = 128000\n' \
                             'request_cpus=CPUS\n' \
                             'request_gpus=NO_GPUS\n' \
                             '+BypassLXCfs="true"\n' \
                             'requirements=GPUS_REQS\n' \
                             'queue QQJOBS'


def generate_id() -> str:
    # ~3t run ids (36**4)
    run_gen = shortuuid.ShortUUID(alphabet=list("0123456789abcdefghijklmnopqrstuvwxyz"))
    return run_gen.random(4)

ID_TMP = generate_id()
ID_EXP = f'_{ID_TMP}'

def get_gpus(min_mem=32000, arch=('volta', 'quadro', 'rtx', 'nvidia')):
    gpu_names = []
    for k, (gpu_name, gpu_arch, gpu_mem) in GPUS.items():
        if gpu_mem >= min_mem and gpu_arch in arch:
            gpu_names.append(gpu_name)
    print("The selected GPUs to run this job are:", gpu_names)
    assert len(gpu_names) > 0, 'Suitable GPU model could not be found'

    return gpu_names


def launch_task_on_cluster(configs: List[Dict[str, str]],
                           num_exp: int = 1, mode: str = 'train',
                           bid_amount: int = 10, num_workers: int = 32,
                           memory: int = 128000, gpu_min_mem:int = 32000,
                           gpu_arch: Optional[List[Tuple[str, ...]]] = 
                           ('volta', 'quadro', 'rtx', 'nvidia')) -> None:


    gpus_requirements = get_gpus(min_mem=gpu_min_mem, arch=gpu_arch)
    gpus_requirements = ' || '.join([f'CUDADeviceName=={x}' for x in gpus_requirements])
    if 'gpus' in configs[0]:
        req_gpus = configs[0]['gpus']
    else:
        req_gpus = 1
    if req_gpus > 1:
        cpus = 6 * req_gpus
    else:
        cpus = int(num_workers/2)

    # stamp_submission = "{:%Y_%d_%m_%H:%M:%S}".format(datetime.now())    
    
    # if exp_opts is not None:
    #     bash += ' --opts '
    #     for opt in exp_opts:
    #         bash += f'{opt} '
    #     bash += 'SYSTEM.CLUSTER_NODE $2.$1'
    # else:
    #     bash += ' --opts SYSTEM.CLUSTER_NODE $2.$1'
    
    # assert mode in MODES
    condor_dir = Path(CONDOR_FD)
    shell_dir = Path(SHELL_SCRIPT_FD)
    no_gpus = 1

    if mode == "train":
        for experiment in configs: 
            run_id = experiment["run_id"]
            extra_args = experiment["args"]
            no_gpus = experiment["gpus"]
            sub_file = SUBMISSION_TEMPLATE
            sub_file = sub_file.replace('EXPMODE', mode)
            sub_file = sub_file.replace('DESCRIPTION', f'{run_id}')
            if no_gpus > 1:
                strategy = 'ddp'
            else:
                strategy = 'auto'

            bash = 'export HYDRA_FULL_ERROR=1 export PYTHONFAULTHANDLER=1\nexport PYTHONUNBUFFERED=1\nexport PATH=$PATH\n' \
                   'export PATH=/home/nathanasiou/apps/imagemagick/bin:$PATH\n' \
                   'export LD_LIBRARY_PATH=/home/nathanasiou/apps/imagemagick/lib:$LD_LIBRARY_PATH\n' \
                   f'exec {sys.executable} train.py ' \
                   f'run_id={run_id} trainer.strategy={strategy} devices={no_gpus} machine.num_workers={int(cpus/2)} {extra_args}'
            shell_dir.mkdir(parents=True, exist_ok=True)
            run_cmd_path = shell_dir / (run_id + '_' + mode + ID_EXP +".sh")

            with open(run_cmd_path, 'w') as f:
                f.write(bash)
            os.chmod(run_cmd_path, stat.S_IRWXU)

            log = f'{mode}/{run_id}'
            for x, y in [("NO_GPUS", str(no_gpus)), ("GPUS_REQS", gpus_requirements),
                         ("CNR_LOG_ID", f'{CONDOR_FD}/{log}/logs'),
                         ("CPUS", str(cpus)),
                         ("RUN_SCRIPT", os.fspath(run_cmd_path))]:
                sub_file = sub_file.replace(x, y)

            submission_path = condor_dir / log / (run_id + ID_EXP + ".sub")
            logdir_condor = condor_dir / log / 'logs'
            logdir_condor.mkdir(parents=True, exist_ok=True)

            with open(submission_path, 'w') as f:
                f.write(sub_file)

            logger.info('The cluster logs for this experiments can be found under:'\
                        f'{str(logdir_condor)}')
            
            cmd = ['condor_submit_bid', f'{bid_amount}', str(submission_path)]
            logger.info('Executing ' + ' '.join(cmd))
            subprocess.run(cmd)
    elif mode in ["sample"]:
        for experiment in configs: 
            extra_args = experiment["args"]
            no_gpus = experiment["gpus"]
            folder = experiment["folder"]
            run_id = folder.split('/')[-2:]
            run_id = '__'.join(run_id)
            sub_file = SUBMISSION_TEMPLATE
            sub_file = sub_file.replace('EXPMODE', mode)
            sub_file = sub_file.replace('DESCRIPTION', f'{run_id}')

            bash = 'export HYDRA_FULL_ERROR=1 export PYTHONFAULTHANDLER=1\nexport PYTHONUNBUFFERED=1\nexport PATH=$PATH\n' \
                   'export PATH=/home/nathanasiou/apps/imagemagick/bin:$PATH\n' \
                   'export LD_LIBRARY_PATH=/home/nathanasiou/apps/imagemagick/lib:$LD_LIBRARY_PATH\n' \
                   f'exec {sys.executable} demo.py ' \
                   f'folder={folder} {extra_args}'
            shell_dir.mkdir(parents=True, exist_ok=True)
            run_cmd_path = shell_dir / (run_id + '_' + mode + ID_EXP +".sh")

            with open(run_cmd_path, 'w') as f:
                f.write(bash)
            os.chmod(run_cmd_path, stat.S_IRWXU)

            log = f'{mode}/{run_id}'
            for x, y in [("NO_GPUS", str(no_gpus)), ("GPUS_REQS", gpus_requirements),
                         ("CNR_LOG_ID", f'{CONDOR_FD}/{log}/logs'),
                         ("CPUS", str(cpus)),
                         ("RUN_SCRIPT", os.fspath(run_cmd_path))]:
                sub_file = sub_file.replace(x, y)

            submission_path = condor_dir / log / (run_id + ID_EXP + ".sub")
            logdir_condor = condor_dir / log / 'logs'
            logdir_condor.mkdir(parents=True, exist_ok=True)

            with open(submission_path, 'w') as f:
                f.write(sub_file)

            logger.info('The cluster logs for this experiments can be found under:'\
                        f'{str(logdir_condor)}')
            
            cmd = ['condor_submit_bid', f'{bid_amount}', str(submission_path)]
            logger.info('Executing ' + ' '.join(cmd))
            subprocess.run(cmd)

    elif mode in ["sweep"]:
        config_file = configs[0]["config"]

        sub_file = SUBMISSION_TEMPLATE_SWEEP
        sub_file = sub_file.replace('EXPMODE', mode)
        import yaml
        pydict_sweep = yaml.safe_load(Path(config_file).read_text())
        sweep_name = configs[0]["sweep-name"]
        pydict_sweep['parameters']['logger.group']['value'] = sweep_name
        pydict_sweep['parameters']['experiment']['value'] = sweep_name

        sweep_id = wandb.sweep(sweep=pydict_sweep,
                               project="motion-editing")
        bash = 'source /home/nathanasiou/.venvs/modit/bin/activate ' \
               'export HYDRA_FULL_ERROR=1 export PYTHONFAULTHANDLER=1\nexport PYTHONUNBUFFERED=1\nexport PATH=$PATH\n' \
               'export PATH=/home/nathanasiou/apps/imagemagick/bin:$PATH\n' \
               'export LD_LIBRARY_PATH=/home/nathanasiou/apps/imagemagick/lib:$LD_LIBRARY_PATH\n' \
               'wandb agent clockwork_pin/motion-editing/SWEEP_ID'

        # cd ~/hoi-common-sense
        # cmd=`wandb sweep --project $1 $2 2> >(grep -o "wandb agent.*$")`
        # echo "replacing command with ${cmd}"
        # sed -i -r "s#wandb.*#$cmd#g" cluster_run_sweep.sh
        # cat cluster_run_sweep.sh
        # echo ""
        # cd

        shell_dir.mkdir(parents=True, exist_ok=True)
        run_cmd_path = shell_dir / (mode + '_' + sweep_id +".sh")
        bash = bash.replace('SWEEP_ID', sweep_id)
        with open(run_cmd_path, 'w') as f:
            f.write(bash)
        os.chmod(run_cmd_path, stat.S_IRWXU)

        num_of_expers = 1
        for k,v in pydict_sweep['parameters'].items():
            if 'values' in v:
                num_of_expers *= len(v['values'])
        log = f'{mode}/{sweep_id}'
        for x, y in [("NO_GPUS", str(no_gpus)), ("GPUS_REQS", gpus_requirements),
                        ("CNR_LOG_ID", f'{CONDOR_FD}/{log}/logs'),
                        ("CPUS", str(cpus)),
                        ("RUN_SCRIPT", os.fspath(run_cmd_path)),
                        ("QQJOBS", str(num_of_expers))]:
            sub_file = sub_file.replace(x, y)

        submission_path = condor_dir / log / (sweep_id + ".sub")
        logdir_condor = condor_dir / log / 'logs'
        logdir_condor.mkdir(parents=True, exist_ok=True)

        with open(submission_path, 'w') as f:
            f.write(sub_file)

        logger.info('The cluster logs for this experiments can be found under:'\
                    f'{str(logdir_condor)}')
        cmd = ['condor_submit_bid', f'{bid_amount}', str(submission_path)]
        logger.info('Executing ' + ' '.join(cmd))
        subprocess.run(cmd)

        # #!/bin/bash
        # cd /home/mdiomataris/hoi-common-sense
        # export PATH="/home/mdiomataris/mambaforge/envs/hoics11/bin:/home/mdiomataris/mambaforge/condabin:/lustre/home/mdiomataris/.vscode-server/bin/d045a5eda657f4d7b676dedbfa7aab8207f8a075/bin/remote-cli:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/usr/lib/jvm/java-8-oracle/bin:/usr/lib/jvm/java-8-oracle/db/bin:/usr/lib/jvm/java-8-oracle/jre/bin:/home/mdiomataris/.fzf/bin"
        # module load cuda/11.7
        # wandb agent markos_mpi/GoalOpt/t99kcvgj

