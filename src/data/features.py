from src.tools.transforms3d import change_for, transform_body_pose
from src.utils.genutils import to_tensor

def _get_body_pose(self, data):
    """get body pose"""
    # default is axis-angle representation: Frames x (Jx3) (J=21)
    pose = to_tensor(data['rots'][..., 3:3 + 21*3])  # drop pelvis orientation
    pose = transform_body_pose(pose, f"aa->{self.rot_repr}")
    return pose
def _get_body_transl(self, data):
    """get body pelvis tranlation"""
    return to_tensor(data['trans'])

def _get_body_orient(self, data):
    """get body global orientation"""
    # default is axis-angle representation
    pelvis_orient = to_tensor(data['rots'][..., :3])
    if self.rot_repr == "6d":
        # axis-angle to rotation matrix & drop last row
        pelvis_orient = transform_body_pose(pelvis_orient, "aa->6d")
    return pelvis_orient

def _get_body_transl_delta_pelv(self, data):
    """
    get body pelvis tranlation delta relative to pelvis coord.frame
    v_i = t_i - t_{i-1} relative to R_{i-1}
    """
    trans = to_tensor(data['trans'])
    trans_vel = trans - trans.roll(1, 0)  # shift one right and subtract
    pelvis_orient = transform_body_pose(to_tensor(data['rots'][..., :3]), "aa->rot")
    trans_vel_pelv = change_for(trans_vel, pelvis_orient.roll(1, 0))
    trans_vel_pelv[0] = 0  # zero out velocity of first frame
    return trans_vel_pelv
