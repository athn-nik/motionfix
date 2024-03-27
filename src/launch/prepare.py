import sys
import os
# os.environ['HOME']='/home/nathanasiou'
# sys.path.insert(0,'/usr/lib/python3.10/')
# os.environ['PYTHONPATH']='/home/nathanasiou/.venvs/teach/lib/python3.10/site-packages'
import warnings
from pathlib import Path
from omegaconf import OmegaConf
from src.tools.runid import generate_id
import hydra

# Local paths
def code_path(path=""):
    code_dir = hydra.utils.get_original_cwd()
    code_dir = Path(code_dir)
    return str(code_dir / path)


def working_path(path):
    return str(Path(os.getcwd()) / path)


# fix the id for this run
ID = generate_id()
def generate_id():
    return ID

def concat_string_list(l, d1, d2, d3):
    """
    Concatenate the strings of a list in a sorted order
    """
    if d1 == 0: l.remove('hml3d')
    if d2 == 0: l.remove('bodilex')
    if d3 == 0: l.remove('sinc_synth')
    return '_'.join(sorted(l))

def get_last_checkpoint(path, ckpt_name="last"):
    if path is None:
        return None
    output_dir = Path(hydra.utils.to_absolute_path(path))
    if ckpt_name != 'last':
        last_ckpt_path = output_dir / "checkpoints" / f'latest-epoch={ckpt_name}.ckpt'
    else:
        last_ckpt_path = output_dir / "checkpoints/last.ckpt"
    return str(last_ckpt_path)

def get_samples_folder(path):
    output_dir = Path(hydra.utils.to_absolute_path(path))
    samples_path = output_dir / "samples"
    return str(samples_path)

def get_expdir(debug):
    if debug:
        return 'experiments'
    else:
        return 'experiments'

def get_debug(debug):
    if debug:
        return '_debug'
    else:
        return ''
    
# this has to run -- pytorch memory leak in the dataloader associated with #973 pytorch issues
#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (12000, rlimit[1]))
# Solutions summarized in --> https://github.com/Project-MONAI/MONAI/issues/701
OmegaConf.register_new_resolver("get_debug", get_debug)
OmegaConf.register_new_resolver("get_expdir", get_expdir)
OmegaConf.register_new_resolver("code_path", code_path)
OmegaConf.register_new_resolver("working_path", working_path)
OmegaConf.register_new_resolver("generate_id", generate_id)
OmegaConf.register_new_resolver("concat_string_list", concat_string_list)
OmegaConf.register_new_resolver("absolute_path", hydra.utils.to_absolute_path)
OmegaConf.register_new_resolver("get_last_checkpoint", get_last_checkpoint)
OmegaConf.register_new_resolver("get_samples_folder", get_samples_folder)


# Remove warnings
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

warnings.filterwarnings(
    "ignore", ".*does not have many workers which may be a bottleneck*"
)

warnings.filterwarnings(
    "ignore", ".*Our suggested max number of worker in current system is*"
)
