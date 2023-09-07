import numpy as np
from pytorch_lightning import LightningModule
from hydra.utils import instantiate
from src.model.metrics import ComputeMetrics
from torchmetrics import MetricCollection
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class BaseModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False, 
                                  ignore=['eval_model','renderer']) # ignore TEMOS score

        # Save visuals, one validation step per validation epoch
        self.store_examples = {"train": None,
                               "val": None}
        self.loss_dict = {'train': None,
                          'val': None,}

        # Need to define:
        # forward
        # allsplit_step()
        # metrics()
        # losses()

    def __post_init__(self):
        trainable, nontrainable = 0, 0
        for p in self.parameters():
            if p.requires_grad:
                trainable += np.prod(p.size())
            else:
                nontrainable += np.prod(p.size())
        self.hparams.n_params_trainable = trainable
        self.hparams.n_params_nontrainable = nontrainable


    def training_step(self, batch, batch_idx):
        train_loss, step_loss_dict = self.allsplit_step("train", 
                                                         batch, batch_idx)
        if self.loss_dict['train'] is None:
            for k, v in step_loss_dict.items():
                step_loss_dict[k] = [v]
            self.loss_dict['train'] = step_loss_dict
        else:
            for k, v in step_loss_dict.items():
                self.loss_dict['train'][k].append(v)
        # for name, param in model.named_parameters():
        #    if param.grad is None:
        #         print(name)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        
        val_loss, step_val_loss_dict = self.allsplit_step("val",
                                                          batch, batch_idx)
        if self.loss_dict['val'] is None:
            for k, v in step_val_loss_dict.items():
                step_val_loss_dict[k] = [v]
            self.loss_dict['val'] = step_val_loss_dict
        else:
            for k, v in step_val_loss_dict.items():
                self.loss_dict['val'][k].append(v)
        # for name, param in model.named_parameters():
        #    if param.grad is None:
        #         print(name)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        return self.allsplit_step("test", batch, batch_idx)

    def loss2logname(self, loss: str, split: str):
        if loss == "total":
            log_name = f"{loss}/{split}"
        else:
            
            if '_multi' in loss:
                if 'bodypart' in loss:
                    loss_type, name, multi, _ = loss.split("_")                
                    name = f'{name}_multiple_bp'
                else:
                    loss_type, name, multi = loss.split("_")                
                    name = f'{name}_multiple'
            else:
                loss_type, name = loss.split("_")
            log_name = f"{loss_type}/{name}/{split}"
        return log_name

    def allsplit_epoch_end(self, split: str):
        # loss_tracker = self.tracker[split]
        # loss_dict = loss_tracker.compute()
        # loss_tracker.reset()
        dico = {self.loss2logname(loss_name, split): sum(loss_value) / len(loss_value)
                for loss_name, loss_value in self.loss_dict[split].items()}
        # workaround for LR, assuming 1 optimizer, 1 scheduler, very weak
        curr_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        dico.update({'Learning Rate': curr_lr})

        dico.update({"epoch": float(self.trainer.current_epoch),
                     "step": float(self.trainer.global_step)})

        if split == "val":
            pass
            # metrics_dict = self.metrics.compute()
            # dico.update({f"Metrics/{metric}": value for metric, value in metrics_dict.items() if '_mean_' in metric})
        self.log_dict(dico, on_epoch=True, sync_dist=True)
        return 

    def on_train_epoch_end(self):
        return self.allsplit_epoch_end("train")
    
    def on_validation_epoch_end(self):
        return self.allsplit_epoch_end("val")

    def on_test_epoch_end(self):
        return self.allsplit_epoch_end("test")

    def configure_optimizers(self):
        optim_dict = {}
        optimizer = torch.optim.AdamW(lr=self.hparams.optim.lr, params=self.parameters())
        
        optim_dict['optimizer'] = optimizer
        # if self.hparams.NAME 
        if self.hparams.lr_scheduler == 'reduceonplateau':
            optim_dict['lr_scheduler'] = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1e-3)
            optim_dict['monitor'] = 'losses/total/train'
        elif self.hparams.lr_scheduler == 'steplr':
            optim_dict['lr_scheduler'] = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200)

        return optim_dict 

    # might be needed not working in multi GPU --> all_split_end
    # Logging per joint things 
    
    # from sinc.info.joints import smplh_joints
    # smplh_joints = smplh_joints[:22]
    # columns = ['Method', 'Epoch']
    # columns.extend(smplh_joints[1:])
    # # self.wandb_table = self.logger.experiment.Table(columns=columns)
    # mnames = ['AVE_pose', 'AVE_joints', 'APE_pose', 'APE_joints']
    # table_joints = []
    # for metric in mnames:
    #     tabdata = metrics_dict[metric]
    #     if '_joints' in metric:
    #         tabdata = tabdata.detach().cpu()
    #         tabdata = tabdata[1:] # discard root for global errors
    #     else:
    #         tabdata = tabdata.detach().cpu()
    #     tabdata = torch.cat((torch.tensor([self.trainer.current_epoch]), tabdata))
    #     tabdata = list(tabdata.numpy())
    #     tabdata = [metric] + tabdata
    #     table_joints.append(tabdata)
    #     # self.logger.experiment.add_data(table_joints)
    #     self.logger.log_table(key="Joints Metrics", data=table_joints, columns=columns)
