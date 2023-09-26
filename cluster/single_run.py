from base import launch_task_on_cluster
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', required=True, choices=['train', 'sample', 
                                                          'eval', 'render', 'fast-render'], type=str,
                            help='Mode is either train or sample or eval!')
    
    parser.add_argument('--folder', required=False, type=str, default=None,
                        help='folder for evaluation')
    parser.add_argument('--expname', required=False, type=str, default='exp_name',
                        help='Experiment Name')
    parser.add_argument('--run-id', required=False, type=str, default='run_id',
                        help='Run ID')
    parser.add_argument('--extras', required=False, default='', type=str, help='args hydra')
    parser.add_argument('--gpus', required=False, default=1, type=int,
                        help='No of GPUS to use')

    parser.add_argument('--bid', required=False, default=10, type=int,
                        help='bid money for cluster')

    parser.add_argument('--mem-gpu', required=False, default=32000, type=int,
                        help='bid money for cluster')

    arguments = parser.parse_args()
    cluster_mode = arguments.mode
    bid_for_exp = arguments.bid
    gpus_no = arguments.gpus
    gpu_mem = arguments.mem_gpu
    if arguments.extras is not None:
        args = arguments.extras
        _args = args.strip().split()
        for i, a in enumerate(_args):
            if '$' in a:
                subst_arg = a.split('=')[-1]
                _args[i] = a.split('=')[0] + '=' + f"'{subst_arg}'"
        args = ' '.join(_args)
    else:
        args = ''

    resume_training = False
    # check if resume is in extras
    if 'resume=' in args:
        path_of_exp_to_resume = args.split("resume=", 1)[1]
        path_of_exp_to_resume = path_of_exp_to_resume.rstrip('/')
        expname_from_prev, run_id_from_prev = path_of_exp_to_resume.split('/')[-3:-1]
        resume_training = True

    if gpus_no > 1: assert cluster_mode == 'train'
    if cluster_mode == 'train':
        if resume_training:
            run_id = run_id_from_prev
            expname =  expname_from_prev
            args += " hydra.output_subdir='.hydra_resume'"
        else:
            run_id = arguments.run_id
            expname = arguments.expname
        experiments = [{"expname": expname, "run_id": run_id, "args": args, "gpus": gpus_no}]
    elif cluster_mode == 'eval':
        pass
    launch_task_on_cluster(experiments, bid_amount=bid_for_exp, 
                           gpu_min_mem=gpu_mem,
                           mode=cluster_mode)
