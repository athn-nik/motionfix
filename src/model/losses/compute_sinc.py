import hydra
import torch
from torch.nn import Module
import logging
from einops import repeat
from hydra.utils import get_original_cwd
logger = logging.getLogger(__name__)



# Define which arguments each loss should use
class LossArguments:
    def __init__(self, _losses_func):
        self._losses_func = _losses_func

    def extract(self, *tensors, lengths):
        return tensors

    # Reconstruction losses
    def recons_jfeats2jfeats(self, *, ds_motion, ds_ref, lengths, **kwargs):
        func = self._losses_func["recons_jfeats2jfeats"]
        args = self.extract(ds_motion.jfeats, ds_ref.jfeats, lengths=lengths)
        return func(*args)

    def recons_text2jfeats(self, *, ds_text, ds_ref, lengths, **kwargs):
        func = self._losses_func["recons_text2jfeats"]
        args = self.extract(ds_text.jfeats, ds_ref.jfeats, lengths=lengths)
        return func(*args)

    def recons_rfeats2rfeats(self, *, ds_motion, ds_ref, lengths, **kwargs):
        func = self._losses_func["recons_rfeats2rfeats"]
        args = self.extract(ds_motion.rfeats, ds_ref.rfeats, lengths=lengths)
        return func(*args)

    def recons_text2rfeats(self, *, ds_text, ds_ref, lengths, **kwargs):
        func = self._losses_func["recons_text2rfeats"]
        args = self.extract(ds_text.rfeats, ds_ref.rfeats, lengths=lengths)
        return func(*args) 

    def kl_text2motion(self, *, dis_text, dis_motion, **kwargs):
        func = self._losses_func["kl_text2motion"]
        return func(dis_text, dis_motion)


    def kl_text2motion_multi(self, *, multi_dis_text, dis_motion, **kwargs):
        func = self._losses_func["kl_text2motion_multi"]
        total_loss = torch.zeros(len(multi_dis_text))
        for i, ds_text in enumerate(multi_dis_text):
            ntexts = ds_text.loc.shape[0]
            motion_dis = torch.distributions.Normal(dis_motion.loc[i][None].tile((ntexts, 1)),
                                                    dis_motion.scale[i][None].tile((ntexts, 1)))
            total_loss[i] = func(ds_text, motion_dis)
        return total_loss.mean()


    def kl_text2motion_multi_bodypart(self, *, multi_dis_text, dis_motion, **kwargs):
        func = self._losses_func["kl_text2motion_multi_bodypart"]
        total_loss = torch.zeros(len(multi_dis_text))
        for i, ds_text in enumerate(multi_dis_text):
            ntexts = ds_text.loc.shape[0]
            motion_dis = torch.distributions.Normal(dis_motion.loc[i,:][None].tile((ntexts, 1, 1)),
                                                    dis_motion.scale[i][None].tile((ntexts, 1, 1)))
            total_loss[i] = func(ds_text, motion_dis)
        return total_loss.mean()


    def kl_motion2text(self, *, dis_motion, dis_text, **kwargs):
        func = self._losses_func["kl_motion2text"]
        return func(dis_motion, dis_text)


    def kl_motion2text_multi(self, *, dis_motion, multi_dis_text, **kwargs):

        func = self._losses_func["kl_motion2text_multi"]
        total_loss = torch.zeros(len(multi_dis_text))
        for i, ds_text in enumerate(multi_dis_text):
            ntexts = ds_text.loc.shape[0]
            motion_dis = torch.distributions.Normal(dis_motion.loc[i][None].tile((ntexts, 1)),
                                                    dis_motion.scale[i][None].tile((ntexts, 1)))
            total_loss[i] = func(motion_dis, ds_text)
        return total_loss.mean()


    def kl_motion2text_multi_bodypart(self, *, dis_motion, multi_dis_text, **kwargs):

        func = self._losses_func["kl_motion2text_multi_bodypart"]
        total_loss = torch.zeros(len(multi_dis_text))
        for i, ds_text in enumerate(multi_dis_text):
            ntexts = ds_text.loc.shape[0]
            motion_dis = torch.distributions.Normal(dis_motion.loc[i][None].tile((ntexts, 1, 1)),
                                                    dis_motion.scale[i][None].tile((ntexts, 1, 1)))
            total_loss[i] = func(motion_dis, ds_text)
        return total_loss.mean()


    def kl_text(self, *, dis_text, dis_ref, **kwargs):
        func = self._losses_func["kl_text"]
        return func(dis_text, dis_ref)

    def kl_text_multi(self, *, multi_dis_text, dis_ref_text, **kwargs):
        func = self._losses_func["kl_text_multi"]
        kl_sum = func(multi_dis_text, dis_ref_text)
        kl_loss = kl_sum / len(multi_dis_text)
        return kl_loss

    def kl_motion(self, *, dis_motion, dis_ref_mot, **kwargs):
        func = self._losses_func["kl_motion"]
        return func(dis_motion, dis_ref_mot)

    # Embedding loss
    def latent_manifold(self, *, lat_text, lat_motion, **kwargs):
        func = self._losses_func["latent_manifold"]
        return func(lat_text, lat_motion)

    def latent_manifold_multi(self, *, multi_lat_text, lat_motion, texts_mask, **kwargs):
        total_loss = 0
        bs, max_texts = texts_mask.shape
        func = self._losses_func["latent_manifold_multi"]

        if multi_lat_text.shape == lat_motion.shape: return func(multi_lat_text, lat_motion)
        for mask, lat_text, lat_mot in zip(texts_mask, multi_lat_text, lat_motion):
            lmot_tile = repeat(lat_mot, "nz latent -> max_texts nz latent", max_texts=max_texts)
            current_loss = torch.nn.functional.smooth_l1_loss(lat_text*mask[:, None, None],
                                                              lmot_tile*mask[:, None, None])
            total_loss += current_loss.mean()
        return (total_loss / bs)


def get_losses_functions(vae: bool, mode: str,
                         modelname: str,
                         separate_latents: bool = False,
                         loss_on_both: bool = False,
                         force_loss_on_jfeats: bool = False,
                         ablation_no_kl_combine: bool = False,
                         ablation_no_motionencoder: bool = False,
                         ablation_no_kl_gaussian: bool = False, 
                        **kwargs):

    losses = []
    if mode == "xyz" or force_loss_on_jfeats:
        if not ablation_no_motionencoder:
            losses.append("recons_jfeats2jfeats")
        losses.append("recons_text2jfeats")
    if mode == "smpl":
        if not ablation_no_motionencoder:
            losses.append("recons_rfeats2rfeats")
        losses.append("recons_text2rfeats")
    else:
        ValueError("This mode is not recognized.")

    if vae or loss_on_both:
        kl_losses = []
        if not ablation_no_kl_combine and not ablation_no_motionencoder:
            if separate_latents:
                if modelname == 'sinc.att_bp':
                    kl_losses.extend(["kl_text2motion_multi_bodypart", "kl_motion2text_multi_bodypart"])
                else:
                    kl_losses.extend(["kl_text2motion_multi", "kl_motion2text_multi"])
            else:
                kl_losses.extend(["kl_text2motion", "kl_motion2text"])

        if not ablation_no_kl_gaussian:
            if ablation_no_motionencoder:
                if separate_latents:
                    kl_losses.extend(["kl_text_multi"])
                else:
                    kl_losses.extend(["kl_text"])
            else:
                if separate_latents:
                    kl_losses.extend(["kl_text_multi", "kl_motion"])
                else:
                    kl_losses.extend(["kl_text", "kl_motion"])

        losses.extend(kl_losses)
    if not vae or loss_on_both:
        if not ablation_no_motionencoder:
            if separate_latents:
               losses.append("latent_manifold_multi")
            else:
                losses.append("latent_manifold")
    logger.info('\n'.join([f'--> {val}' for val in losses]))
    return losses


class SincLosses(Module):
    def __init__(self, **kwargs):
        super().__init__()

        losses = get_losses_functions( **kwargs)
        losses.append("total")

        self.losses = losses
        # Instantiate loss functions
        self._losses_func = {loss: hydra.utils.instantiate(kwargs[loss + "_func"])
                             for loss in losses if loss != "total"}

        self.lossArguments = LossArguments(self._losses_func)

        # Save the lambda parameters
        self._params = {loss: kwargs[loss] for loss in losses if loss != "total"}


    def forward(self, **kwargs):
        total = 0.0
        losses_results = {}

        for loss in self.losses:
            if loss == "total":
                continue

            # compute the loss
            val = getattr(self.lossArguments, loss)(**kwargs)
            # compute the weighted value
            weighted_val = self._params[loss] * val

            # gradients attached to total
            total += weighted_val

            # keep only for logging
            losses_results[loss] = val.detach()

        losses_results["total"] = total.detach()
        return total, losses_results
