from typing import List
import numpy as np
from einops import reduce
import torch
from src.utils.file_io import hack_path
import smplx

def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)

class MotionEditEvaluator:
    def __init__(self, metrics_to_eval: List[str],
                 smplh_path: str):
        self.metrics_to_eval = metrics_to_eval
        self.eval_functions = {
            # 'foot_skating': self.calculate_foot_skating,
            'loc_pres': self.motion_preservance,
            'glo_pres': self.motion_preservance,
            'lc_pre_gt': self.motion_preservance,
            'gb_pre_gt': self.motion_preservance,
            'loc_edit': self.edit_accuracy,
            'glob_edit': self.edit_accuracy,

        }
        # self.eval_functions = {
        #     # 'foot_skating': self.calculate_foot_skating,
        #     'loc_pres': [0, 2],
        #     'glo_pres': [0, 2],
        #     'lc_pre_gt': [0, 1],
        #     'gb_pre_gt': [0, 1],
        #     'loc_edit': [1, 2],
        #     'glob_edit': [1, 2],
        # }

        smpl_path = hack_path(smplh_path, keyword='data')
        self.body_model = smplx.SMPLHLayer(f'{smplh_path}/smplh',
                                           model_type='smplh',
                                           gender='neutral',
                                           ext='npz').to('cuda').eval();

        self.metrics_batch = []
        self.meta_data = []
 
    def filter_verts(vertices):
        velmo = vertices[1:] - vertices[:-1]
        avg_vels = torch.linalg.norm(velmo, dim=-1).mean(1) # power of vels per frame averaged

        velmaxi = avg_vels.max(dim=1)[0][:, None]
        velmini = avg_vels.min(dim=1)[0][:, None]
        avg_vels_norm = (avg_vels-velmini)/(velmaxi - velmini + 1e-5)
        # zero out the vertices that are less than 0.65
        avg_vels_norm[avg_vels_norm < 0.65] = 0
        return avg_vels_norm 

    def motion_preservance(self, x, y, force_cm=True):
        seqlen = x.shape[1]
        if force_cm:
            mult = 100
        else:
            mult = 1
        local_motion_preservance = l2_norm(mult*x, mult*y, dim=1)/seqlen
        # local_motion_preservance_gt = local_motion_preservance_gt.mean()
        return local_motion_preservance.mean((1,2))

    def edit_accuracy(self, x, y, force_cm=True):
        seqlen = x.shape[1]
        if force_cm:
            mult = 100
        else:
            mult = 1
        global_edit_accuracy = l2_norm(mult*x, mult*y, dim=1) / seqlen
        # global_edit_accuracy = global_edit_accuracy.mean()

        return global_edit_accuracy.mean((1,2))

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

    def get_vertices(self, source, target, preds):
        B, S_src, _ = source['body_pose'].shape
        B, S_tgt, _ = target['body_pose'].shape

        if S_src > S_tgt:
            for k, v in source.items():
                source[k] = v[:, :S_tgt]
        else:
            for k, v in target.items():
                target[k] = v[:, :S_src]
                preds[k] = v[:, :S_src]

        B, S_src, _ = source['body_pose'].shape
        B, S_tgt, _ = target['body_pose'].shape

        source_verts  = self.run_smpl_fwd(source['body_transl'].detach(),
                                          source['body_orient'].detach(),
                                          source['body_pose'].detach().reshape(B, S_src,
                                                                      63))
        source_v = source_verts.vertices.reshape(B, S_src, -1, 3)
        lo_source_v = source_v - source['body_transl'][:, :, None, :]


        target_verts  = self.run_smpl_fwd(target['body_transl'].detach(),
                                          target['body_orient'].detach(),
                                          target['body_pose'].detach().reshape(B, S_tgt,
                                                                      63))
        target_v = target_verts.vertices.reshape(B, S_tgt, -1, 3)
        lo_target_v = target_v - target['body_transl'][:, :, None, :]

        pred_target_verts  = self.run_smpl_fwd(preds['body_transl'].detach(),
                                               preds['body_orient'].detach(),
                                               preds['body_pose'].detach().reshape(B,
                                                                          S_tgt,
                                                                          63))
        pred_target_v = pred_target_verts.vertices.reshape(B, S_tgt, -1, 3)
        lo_pred_v = pred_target_v - preds['body_transl'][:, :,
                                                                    None, :]
        return lo_source_v, lo_target_v, lo_pred_v, source_v, target_v, pred_target_v

    def evaluate_motion_batch(self, source, target, preds,
                              meta_data: dict=None):
        # get it to vertices 
        src_lc, tgt_lc, pred_lc, src, tgt, pred = self.get_vertices(source,
                                                                    target, preds)
        metrics = {}
        for metric in self.metrics_to_eval:
            func_metr = self.eval_functions[metric]
            if 'loc' in metric or 'lc' in metric:
                if 'edit' in metric:
                    metrics[metric] = func_metr(tgt_lc, pred_lc)
                else:
                    if '_gt' in metric:
                        metrics[metric] = func_metr(src_lc, tgt_lc)
                    else:
                        metrics[metric] = func_metr(src_lc, pred_lc)
            else:
                if 'edit' in metric:
                    metrics[metric] = func_metr(tgt, pred)
                else:
                    if '_gt' in metric:
                        metrics[metric] = func_metr(src, tgt)
                    else:
                        metrics[metric] = func_metr(src, pred)
        self.metrics_batch.append(metrics)
        # metrics = {metric: self.eval_functions[metric](motion) 
        #            for metric in self.metrics_to_eval}
        return metrics

    def get_metrics(self):
        metrics_batched = {metric: torch.cat([m[metric] for m in self.metrics_batch],
                                          dim=0)#.sort()[0]
                   for metric in self.metrics_to_eval}
        metrics_batched.update({
        'gb_preserv_l2_diff ↓' : (metrics_batched['glo_pres'] -
                                  metrics_batched['gb_pre_gt']).abs(),
        'loc_preserv_l2_diff ↓' : (metrics_batched['loc_pres'] -
                                   metrics_batched['lc_pre_gt']).abs()
        })
        del metrics_batched['glo_pres']
        del metrics_batched['gb_pre_gt']
        del metrics_batched['loc_pres']
        del metrics_batched['lc_pre_gt']

        metrics_indices = {metric: torch.cat([m[metric] for m in self.metrics_batch],
                                          dim=0).sort()[1]
                           for metric in self.metrics_to_eval}
        ### AVERAGING
        metrics_avg = {metric+'_avg': torch.cat([m[metric] for m in self.metrics_batch],
                                          dim=0).mean()
                       for metric in self.metrics_to_eval}
        # for k, v in metrics '↓' '↑'
        metrics_avg.update({
        'gb_preserv_l2_diff_avg ↓' : (metrics_avg['glo_pres_avg'] -
                                       metrics_avg['gb_pre_gt_avg']
                                     ).abs(),
        'loc_preserv_l2_diff_avg ↓' : (metrics_avg['loc_pres_avg'] -
                                       metrics_avg['lc_pre_gt_avg']
                                      ).abs()
        })
        metrics_avg['loc_edit_avg ↓'] = metrics_avg.pop('loc_edit_avg')
        metrics_avg['glob_edit_avg ↓'] = metrics_avg.pop('glob_edit_avg')

        del metrics_avg['glo_pres_avg']
        del metrics_avg['gb_pre_gt_avg']
        del metrics_avg['loc_pres_avg']
        del metrics_avg['lc_pre_gt_avg']

        metrics_batched = {k: v.cpu().numpy() for k, v in metrics_batched.items()}
        metrics_avg = {k: v.cpu().numpy() for k, v in metrics_avg.items()}
        metrics_indices = {k: v.cpu().numpy() for k, v in metrics_indices.items()}

        # meta_data = {k: np.([m[k] for m in self.meta_data],
        #                                dim=0)
        #              for k in self.meta_data[0].keys()}
        return {'metrics': metrics_batched,
                'metrics_avg': metrics_avg,
                # 'meta_data': meta_data
                }, metrics_indices