from typing import List
import logging

import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric

from sinc.transforms.joints2jfeats import Rifke
from sinc.tools.geometry import matrix_of_angles
from sinc.model.utils.tools import remove_padding
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)


def variance(x, T, dim):
    mean = x.mean(dim)
    out = (x - mean)**2
    out = out.sum(dim)
    return out / (T - 1)

class ComputeMetricsSinc(Metric):
    def __init__(self, eval_model=None, jointstype: str = "smplh",
                 sync_on_compute=False,
                 dist_sync_on_step=False,
                 full_state_update=False,
                 **kwargs):
        super().__init__(
                         # sync_on_compute=sync_on_compute,
                         dist_sync_on_step=dist_sync_on_step,
                        #  full_state_update=full_state_update
                         )
        assert jointstype in ["smplh"]

        from sinc.info.joints import number_of_joints

        ndims = number_of_joints[jointstype]

        self.jointstype = jointstype

        self.eval_model = eval_model
        # self.temos_path = '/is/cluster/fast/nathanasiou/data/motion-language/sinc-checkpoints/temos_score/bs32'
        # def load_temos():
        #     from hydra.utils import instantiate
        #     temos_path = Path(self.temos_path)
        #     temoscfg = OmegaConf.load(temos_path / ".hydra/config.yaml")

        #     # Overload it
        #     logger.info("Loading TEMOS model")
        #     # Instantiate all modules specified in the configs
        #     temos_model = instantiate(temoscfg.model,
        #                             nfeats=135,
        #                             logger_name="none",
        #                             nvids_to_save=None,
        #                             _recursive_=False)

        #     last_ckpt_path = temos_path / "checkpoints/last.ckpt"
        #     # Load the last checkpoint
        #     temos_model = temos_model.load_from_checkpoint(last_ckpt_path)
        #     temos_model.eval()
        #     logger.info("TEMOS Model weights restored")
        #     return temos_model, temoscfg
        # #import ipdb; ipdb.set_trace()
        # #self.temos_model, _ = load_temos()
        # TODO fix this should be smplh
        self.rifke = Rifke(jointstype=jointstype,
                           normalization=False)

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_seq", default=torch.tensor(0), dist_reduce_fx="sum")
        # TODO remove hard-coded stuff from here
        # APE
        self.add_state("APE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("APE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("APE_pose", default=torch.zeros(ndims-1), dist_reduce_fx="sum")
        self.add_state("APE_joints", default=torch.zeros(ndims), dist_reduce_fx="sum")
        self.APE_metrics = ["APE_root", "APE_traj", "APE_pose", "APE_joints"]

        # AVE
        self.add_state("AVE_root", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("AVE_traj", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("AVE_pose", default=torch.zeros(ndims-1), dist_reduce_fx="sum")
        self.add_state("AVE_joints", default=torch.zeros(ndims), dist_reduce_fx="sum")
        self.AVE_metrics = ["AVE_root", "AVE_traj", "AVE_pose", "AVE_joints"]
 
        self.add_state("Temos_Score", default=torch.tensor(0.), dist_reduce_fx="sum")
        # self.temos_score = ["Temos_Score"]
    
    def compute(self, mode='train'):
        count = self.count
        APE_metrics = {metric: getattr(self, metric) / count for metric in self.APE_metrics}

        APE_metrics["APE_mean_pose"] = self.APE_pose.mean() / count
        APE_metrics["APE_mean_joints"] = self.APE_joints.mean() / count

        if mode == 'train':
            APE_metrics.pop("APE_pose")
            APE_metrics.pop("APE_joints")

        count_seq = self.count_seq
        AVE_metrics = {metric: getattr(self, metric) / count_seq for metric in self.AVE_metrics}

        # Compute average of AVEs
        AVE_metrics["AVE_mean_pose"] = self.AVE_pose.mean() / count_seq
        AVE_metrics["AVE_mean_joints"] = self.AVE_joints.mean() / count_seq
        
        # temos_score["Temos_Score"] = self.TEMOS_SCORE.mean()

        if mode == 'train':
            AVE_metrics.pop("AVE_pose")
            AVE_metrics.pop("AVE_joints")
        
        if self.eval_model is not None:
            temos_score = self.Temos_Score / count_seq
            return {**APE_metrics, **AVE_metrics, 'Temos_Score': temos_score}
        return {**APE_metrics, **AVE_metrics}


    def update(self, feat_text_lst: Tensor, feat_ref_lst: Tensor,
            #    rots_text_lst: Tensor, rots_ref_lst: Tensor,
               lengths: List[int]):
        jts_text_lst = feat_text_lst.joints
        jts_ref_lst = feat_ref_lst.joints

        rfeats_text_lst = feat_text_lst.rfeats
        rfeats_ref_lst = feat_ref_lst.rfeats

        from sinc.data.tools import collate_tensor_with_padding
        jts_text = collate_tensor_with_padding(jts_text_lst)
        jts_ref = collate_tensor_with_padding(jts_ref_lst)

        if self.eval_model is not None:
            with torch.no_grad():
                distribution_ref = self.eval_model(rfeats_ref_lst)
                distribution_motion = self.eval_model(rfeats_text_lst)
                mu_ref = distribution_ref.loc.squeeze()
                mu_motion = distribution_motion.loc.squeeze()
        if self.eval_model is not None:
            self.Temos_Score += 2*(1-torch.nn.CosineSimilarity()(mu_motion, mu_ref)).mean()

        self.count += sum(lengths)
        self.count_seq += len(lengths)
        jts_text, poses_text, root_text, traj_text = self.transform(jts_text, lengths)
        jts_ref, poses_ref, root_ref, traj_ref = self.transform(jts_ref, lengths)
        for i in range(len(lengths)):
            self.APE_root += l2_norm(root_text[i], root_ref[i], dim=1).sum()
            self.APE_pose += l2_norm(poses_text[i], poses_ref[i], dim=2).sum(0)
            self.APE_traj += l2_norm(traj_text[i], traj_ref[i], dim=1).sum()
            self.APE_joints += l2_norm(jts_text[i], jts_ref[i], dim=2).sum(0)

            root_sigma_text = variance(root_text[i], lengths[i], dim=0)
            root_sigma_ref = variance(root_ref[i], lengths[i], dim=0)
            self.AVE_root += l2_norm(root_sigma_text, root_sigma_ref, dim=0)

            traj_sigma_text = variance(traj_text[i], lengths[i], dim=0)
            traj_sigma_ref = variance(traj_ref[i], lengths[i], dim=0)
            self.AVE_traj += l2_norm(traj_sigma_text, traj_sigma_ref, dim=0)

            poses_sigma_text = variance(poses_text[i], lengths[i], dim=0)
            poses_sigma_ref = variance(poses_ref[i], lengths[i], dim=0)
            self.AVE_pose += l2_norm(poses_sigma_text, poses_sigma_ref, dim=1)

            jts_sigma_text = variance(jts_text[i], lengths[i], dim=0)
            jts_sigma_ref = variance(jts_ref[i], lengths[i], dim=0)
            self.AVE_joints += l2_norm(jts_sigma_text, jts_sigma_ref, dim=1)

    def transform(self, joints: Tensor, lengths):
        features = self.rifke(joints)
        ret = self.rifke.extract(features)
        root_grav_axis, poses_features, vel_angles, vel_trajectory_local = ret

        # already have the good dimensionality
        angles = torch.cumsum(vel_angles, dim=-1)
        # First frame should be 0, but if infered it is better to ensure it
        angles = angles - angles[..., [0]]

        cos, sin = torch.cos(angles), torch.sin(angles)
        rotations = matrix_of_angles(cos, sin, inv=False)

        # Get back the local poses
        poses_local = rearrange(poses_features, "... (joints xyz) -> ... joints xyz", xyz=3)

        # Rotate the poses
        poses = torch.einsum("...lj,...jk->...lk", poses_local[..., [0, 1]], rotations)
        poses = torch.stack((poses[..., 0], poses[..., 1], poses_local[..., 2]), axis=-1)

        # Rotate the vel_trajectory
        vel_trajectory = torch.einsum("...j,...jk->...k", vel_trajectory_local, rotations)
        # Integrate the trajectory
        # Already have the good dimensionality
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # get the root joint
        root = torch.cat((trajectory[..., :, [0]],
                          trajectory[..., :, [1]],
                          root_grav_axis[..., None],), dim=-1)

        # Add the root joints (which is still zero)
        poses = torch.cat((0 * poses[..., [0], :], poses), -2)
        # put back the root joint y
        poses[..., 0, 2] = root_grav_axis
        # Add the trajectory globally
        poses[..., [0, 1]] += trajectory[..., None, :]

        # return results in meters
        return (remove_padding(poses, lengths),
                remove_padding(poses_local, lengths),
                remove_padding(root, lengths),
                remove_padding(trajectory, lengths))
