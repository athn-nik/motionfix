from typing import Optional

import torch
from torch import Tensor
from einops import rearrange

from sinc.tools.easyconvert import matrix_to, nfeats_of, to_matrix
import sinc.tools.geometry as geometry

from .base import Rots2Rfeats


class Globalvelandy(Rots2Rfeats):
    def __init__(self, path: Optional[str] = None,
                 normalization: bool = False,
                 pose_rep: str = "rot6d",
                 canonicalize: bool = False,
                 offset: bool = True,
                 keep_dim: bool = False, # flatten the features or not
                 **kwargs) -> None:
        super().__init__(path=path, normalization=normalization)
        self.canonicalize = canonicalize
        self.pose_rep = pose_rep
        self.nfeats = nfeats_of(pose_rep)
        self.offset = offset
        self.keep_dim = keep_dim

    def forward(self, data, first_frame=None) -> Tensor:
        matrix_poses, trans = data.rots, data.trans
        # matrix_poses: [nframes, 22, 3, 3]

        # extract the root gravity axis
        # for smpl it is the last coordinate
        root_y = trans[..., 2]
        trajectory = trans[..., [0, 1]]

        # Compute the difference of trajectory
        vel_trajectory = torch.diff(trajectory, dim=-2)
        # 0 for the first one => keep the dimentionality

        if first_frame is None:
            first_frame = 0 * vel_trajectory[..., [0], :]

        vel_trajectory = torch.cat((first_frame, vel_trajectory), dim=-2)

        # first normalize the data
        if self.canonicalize:
            global_orient = matrix_poses[..., 0, :, :]
            # remove the rotation
            rot2d = geometry.matrix_to_axis_angle(global_orient[..., 0, :, :])
            # Remove the fist rotation along the vertical axis
            # construct this by extract only the vertical component of the
            # rotation
            rot2d[..., :2] = 0

            if self.offset:
                # add a bit more rotation
                rot2d[..., 2] += torch.pi/2

            rot2d = geometry.axis_angle_to_matrix(rot2d)

            # turn with the same amount all the rotations
            global_orient = torch.einsum("...kj,...kl->...jl", rot2d,
                                         global_orient)

            matrix_poses = torch.cat((global_orient[..., None, :, :],
                                      matrix_poses[..., 1:, :, :]), dim=-3)

            # Turn the trajectory as well
            vel_trajectory = torch.einsum("...kj,...lk->...lj",
                                          rot2d[..., :2, :2],
                                          vel_trajectory)

        poses = matrix_to(self.pose_rep, matrix_poses)
        if self.keep_dim:
            batch_len = root_y.shape
            extra_feats = torch.cat((root_y[..., None],
                                    vel_trajectory,
                                    torch.zeros((*batch_len,3))),
                                    dim=-1)
            features = torch.cat((extra_feats[..., None, :], poses),
                                dim=-2)
        else:
            features = torch.cat((root_y[..., None],
                                  vel_trajectory,
                                  rearrange(poses,
                                            "... joints rot -> ... (joints rot)")),
                                 dim=-1)
        features = self.normalize(features)

        return features

    def extract_bodyparts(self, features):
        from sinc.info.joints import bp2ids
        bs, seqlen = features.shape[:2]
        vel_traj_global_orient = features[..., :9]
        features_3d = features[..., 9:].reshape(bs, seqlen, 21, 6)
        features_kd = torch.zeros((bs, seqlen, 22, 6), device=features_3d.device)
        features_kd[..., 1:, :] = features_3d
        torso = features_kd[..., bp2ids['torso'], :].flatten(-2)
        la = features_kd[..., bp2ids['left arm'], :].flatten(-2)
        ra = features_kd[..., bp2ids['right arm'], :].flatten(-2)
        ll = features_kd[..., bp2ids['left leg'], :].flatten(-2)
        rl = features_kd[..., bp2ids['right leg'], :].flatten(-2)
        bparts = [vel_traj_global_orient, torso, la, ra, ll, rl]
        return bparts

    def extract(self, features):
        if self.keep_dim:
            root_grav_axis = features[..., 0, 0]
            vel_trajectory = features[..., 0, 1:3]
            poses_features = features[..., 1:, :]
            poses = poses_features
        else:
            root_grav_axis = features[..., 0]
            vel_trajectory = features[..., 1:3]
            poses_features = features[..., 3:]
            poses = rearrange(poses_features,
                            "... (joints rot) -> ... joints rot", rot=self.nfeats)
        return root_grav_axis, vel_trajectory, poses

    def inverse(self, features, last_frame=None):
        features = self.unnormalize(features)

        root_grav_axis, vel_trajectory, poses = self.extract(features)
        # integrate the trajectory
        trajectory = torch.cumsum(vel_trajectory, dim=-2)
        if last_frame is None:
            pass

        # First frame should be 0, but if infered it is better to ensure it
        trajectory = trajectory - trajectory[..., [0], :]

        # Get back the translation
        # if self.keep_dim:
        #     trans = torch.cat([trajectory, root_grav_axis], dim=-1)
        # else:
        trans = torch.cat([trajectory, root_grav_axis[..., None]], dim=-1)

        matrix_poses = to_matrix(self.pose_rep, poses)

        from sinc.transforms.smpl import RotTransDatastruct
        return RotTransDatastruct(rots=matrix_poses, trans=trans)
