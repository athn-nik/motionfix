from torchmetrics import Metric
import torch
from torch import Tensor
import smplx
from src.utils.genutils import freeze
from src.model.utils.smpl_fast import smpl_forward_fast
from typing import Dict, List
from src.utils.file_io import hack_path

def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)

class ComputeMetrics(Metric):
    def __init__(self, smpl_path: str):
        super().__init__()
        self.add_state("loc_motion_preservance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("global_motion_preservance", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        self.add_state("local_motion_preservance_gt", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("global_motion_preservance_gt", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("local_edit_accuracy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("global_edit_accuracy", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        self.add_state("acceleration", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        self.add_state("count_seqs", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_lens_mins", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count_lens_tgt", default=torch.tensor(0), dist_reduce_fx="sum")
        smpl_path = hack_path(smpl_path, keyword='data')
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

    def update(self, source: Dict[str, Tensor], preds: Dict[str, Tensor],
               target: Dict[str, Tensor], lengths_source: List[int],
               lengths_target: List[int]):

        B, S_src, _ = source['body_pose'].shape
        B, S_tgt, _ = target['body_pose'].shape
        min_lens = [min(lengths_source[idx],
                        lengths_target[idx]) for idx in range(B)]


        source_verts  = self.run_smpl_fwd(source['body_transl'].detach(),
                                          source['body_orient'].detach(),
                                          source['body_pose'].detach().reshape(B, S_src,
                                                                      63))
        source_verts = source_verts.vertices.reshape(B, S_src, -1, 3)
        local_source_verts = source_verts - source['body_transl'][:, :, None, :]


        target_verts  = self.run_smpl_fwd(target['body_transl'].detach(),
                                          target['body_orient'].detach(),
                                          target['body_pose'].detach().reshape(B, S_tgt,
                                                                      63))
        target_verts = target_verts.vertices.reshape(B, S_tgt, -1, 3)
        local_target_verts = target_verts - target['body_transl'][:, :, None, :]

        pred_target_verts  = self.run_smpl_fwd(preds['body_transl'].detach(),
                                               preds['body_orient'].detach(),
                                               preds['body_pose'].detach().reshape(B,
                                                                          S_tgt,
                                                                          63))
        pred_target_verts = pred_target_verts.vertices.reshape(B, S_tgt, -1, 3)
        local_pred_verts = pred_target_verts - preds['body_transl'][:, :, None, :]
        # Average the acceleration values across the sequence (S) and batch (B) dimensions
        # This will result in a tensor of shape (J, 3)
        velocity = pred_target_verts[:, 1:] - pred_target_verts[:, :-1]

        # Compute the acceleration (second derivative of position)
        acceleration_tot = velocity[:, 1:] - velocity[:, :-1]
        mean_accel_per_seq = acceleration_tot.mean(dim=1)

        self.acceleration += mean_accel_per_seq.sum()

        self.count_lens_mins += sum(min_lens)
        self.count_lens_tgt += sum(lengths_target)
        self.count_seqs += len(min_lens) 


        for i in range(len(min_lens)):
            self.local_motion_preservance_gt += l2_norm(source_verts[i,
                                                                  :min_lens[i]],
                                                     target_verts[i,
                                                                  :min_lens[i]],
                                                     dim=1).sum()
            self.global_motion_preservance_gt += l2_norm(source_verts[i,
                                                                  :min_lens[i]],
                                                      target_verts[i,
                                                                  :min_lens[i]],
                                                      dim=1).sum()
            
            self.local_motion_preservance += l2_norm(source_verts[i,
                                                                  :min_lens[i]],
                                                     pred_target_verts[i,
                                                                  :min_lens[i]],
                                                     dim=1).sum()
            self.global_motion_preservance += l2_norm(source_verts[i,
                                                                  :min_lens[i]],
                                                      pred_target_verts[i,
                                                                  :min_lens[i]],
                                                      dim=1).sum()
 
            self.local_edit_accuracy += l2_norm(target_verts[i, 
                                                            :lengths_target[i]],
                                                pred_target_verts[i, 
                                                            :lengths_target[i]],
                                                dim=1).sum()
            self.global_edit_accuracy += l2_norm(target_verts[i,
                                                            :lengths_target[i]],
                                                 pred_target_verts[i, 
                                                            :lengths_target[i]],
                                                 dim=1).sum()

    
    def compute(self):
        total_mins = self.count_seqs * self.count_lens_mins
        total_tgt = self.count_seqs * self.count_lens_tgt

        final_metrs= {
                      'MoPres_LCL_golden':
                      self.local_motion_preservance_gt / total_mins ,
                      'MoPres_GLB_golden':
                      self.global_motion_preservance_gt / total_mins,
                      'MoPres_LCL':
                      self.local_motion_preservance / total_mins ,
                      'MoPres_GLB':
                      self.global_motion_preservance / total_mins,
                      'EdAcc_LCL':
                      self.local_edit_accuracy / total_tgt,
                      'EdAcc_GLB':
                      self.global_edit_accuracy / total_tgt,
                      'Acceleration':
                      self.acceleration / total_tgt

                      }

        return final_metrs