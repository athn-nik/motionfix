from src.tools.transforms3d import transform_body_pose
import torch

def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def remove_padding(tensors, lengths):
    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensors, lengths)]

def pack_to_render(rots, trans, pose_repr='6d'):
    # make axis-angle
    # global_orient = transform_body_pose(rots, f"{pose_repr}->aa")
    if pose_repr != 'aa':
        body_pose = transform_body_pose(rots, f"{pose_repr}->aa")
    else:
        body_pose = rots
    if trans is None:
        trans = torch.zeros((rots.shape[0], rots.shape[1], 3),
                             device=rots.device)
    render_d = {'body_transl': trans,
                'body_orient': body_pose[..., :3],
                'body_pose': body_pose[..., 3:]}
    return render_d