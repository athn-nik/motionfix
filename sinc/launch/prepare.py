import sys
import os
# os.environ['HOME']='/home/nathanasiou'
# sys.path.insert(0,'/usr/lib/python3.10/')
# os.environ['PYTHONPATH']='/home/nathanasiou/.venvs/teach/lib/python3.10/site-packages'
import warnings
from pathlib import Path
from omegaconf import OmegaConf
from sinc.tools.runid import generate_id
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


def get_last_checkpoint(path, ckpt_name="last"):
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


def get_kitname(load_amass_data: bool, load_with_rot: bool):
    if not load_amass_data:
        return "kit-mmm-xyz"
    if load_amass_data and not load_with_rot:
        return "kit-amass-xyz"
    if load_amass_data and load_with_rot:
        return "kit-amass-rot"

# this has to run -- pytorch memory leak in the dataloader associated with #973 pytorch issues
#import resource
#rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
#resource.setrlimit(resource.RLIMIT_NOFILE, (12000, rlimit[1]))
# Solutions summarized in --> https://github.com/Project-MONAI/MONAI/issues/701

OmegaConf.register_new_resolver("code_path", code_path)
OmegaConf.register_new_resolver("working_path", working_path)
OmegaConf.register_new_resolver("generate_id", generate_id)
OmegaConf.register_new_resolver("absolute_path", hydra.utils.to_absolute_path)
OmegaConf.register_new_resolver("get_last_checkpoint", get_last_checkpoint)
OmegaConf.register_new_resolver("get_samples_folder", get_samples_folder)
OmegaConf.register_new_resolver("get_kitname", get_kitname)


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
