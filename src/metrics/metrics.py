from torchmetrics import Metric
import torch
from torch import Tensor
import smplx
from src.utils.genutils import freeze
from src.model.utils.smpl_fast import smpl_forward_fast
from typing import Dict, List


def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)

class BaseMetric(Metric):
    def __init__(self, smpl_path: str, ):
        super().__init__()
        self.add_state("local_motion_preservance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("global_motion_preservance", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("local_edit_accuracy", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("global_edit_accuracy", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.body_model = smplx.SMPLHLayer(f'{smpl_path}/smplh', model_type='smplh',
                                           gender='neutral',
                                           ext='npz').to(self.device).eval();

    def run_smpl_fwd(self, body_transl, body_orient, body_pose):
        if len(body_transl.shape) > 2:
            body_transl = body_transl.flatten(0, 1)
            body_orient = body_orient.flatten(0, 1)
            body_pose = body_pose.flatten(0, 1)
  
        batch_size = body_transl.shape[0]
        from src.tools.transforms3d import transform_body_pose
        self.body_model.batch_size = batch_size
        return self.body_model(transl=body_transl,
                               body_pose=transform_body_pose(body_pose,
                                                             'aa->rot'),
                               global_orient=transform_body_pose(body_orient,
                                                                 'aa->rot'))

    def update(self, source: Dict[Tensor], preds: Dict[Tensor],
               target: Dict[Tensor], lengths: List[int]):


        B, S, _ = source['body_pose'].shape
        source_verts  = self.run_smpl_fwd(source['body_transl'],
                                          source['body_orient'],
                                          source['body_pose'].reshape(B, S, 63))
        source_verts = source_verts.vertices

        target_verts  = self.run_smpl_fwd(target['body_transl'],
                                          target['body_orient'],
                                          target['body_pose'].reshape(B, S, 63))
        target_verts = target_verts.vertices

        pred_target_verts  = self.run_smpl_fwd(preds['body_transl'],
                                               preds['body_orient'],
                                               preds['body_pose'].reshape(B,
                                                                          S,
                                                                          63))
        pred_target_verts = pred_target_verts.vertices


        self.count_lens += sum(lengths)
        self.count_seqs += len(lengths) 


        for i in range(len(lengths)):
            self.local_motion_preservance += l2_norm(source_verts[i,
                                                                  :lengths[i]],
                                                     target_verts[i],
                                                     dim=1).sum()
            self.global_motion_preservance += l2_norm(source_verts[i,
                                                                  :lengths[i]],
                                                      target_verts[i],
                                                      dim=1).sum()
            self.local_edit_accuracy += l2_norm(target_verts[i, :lengths[i]],
                                                pred_target_verts[i],
                                                dim=1).sum()
            self.global_edit_accuracy += l2_norm(target_verts[i, :lengths[i]],
                                                 pred_target_verts[i],
                                                 dim=1).sum()

        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.total += target.numel()

    def compute(self):
        final_metrs= {'MoPres_LCL': self.local_motion_preservance / self.count,
                      'MoPres_GLB': self.global_motion_preservance / self.count,
                      'EdAcc_LCL': self.local_edit_accuracy / self.count,
                      'EdAcc_GLB': self.global_edit_accuracy / self.count}

        return final_metrs