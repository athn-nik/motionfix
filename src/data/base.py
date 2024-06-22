import pytorch_lightning as pl
from torch.utils.data import DataLoader

from src.data.tools.collate import collate_batch_last_padding, collate_datastruct_and_text
import torch
from typing import List

class BASEDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size: int,
                 num_workers: int,
                 load_feats: List[str],
                 batch_sampler: str | None = None,
                 dataset_percentages: dict[str, float] | None = None):
        super().__init__()

        collate_fn = lambda b: collate_batch_last_padding(b, load_feats)

        def set_worker_sharing_strategy(worker_id: int) -> None:
            sharing_strategy = "file_system"
            torch.multiprocessing.set_sharing_strategy(sharing_strategy)
        self.dataloader_options = {
            'batch_size': batch_size,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
            'drop_last': False,
            'worker_init_fn': set_worker_sharing_strategy
            # 'pin_memory': True,
            }
        self.batch_sampler = batch_sampler
        self.ds_perc = dataset_percentages
        self.batch_size = batch_size
        # need to be overloaded:
        # - self.Dataset
        # - self._sample_set => load only a small subset
        #   There is an helper below (get_sample_set)
        # - self.nfeats
        # - self.transforms
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

        # Optional
        self._subset_dataset = None
        
    def get_sample_set(self, overrides={}):
        sample_params = self.hparams.copy()
        sample_params.update(overrides)
        return self.Dataset(**sample_params)

    def train_dataloader(self):
        if self.batch_sampler is not None:
            from src.data.sampling.custom_batch_sampler import PercBatchSampler, CustomBatchSampler, CustomBatchSamplerV2, CustomBatchSamplerV4
            from src.data.sampling.custom_batch_sampler import mix_datasets_anysize
            # ratio_batch_sampler = CustomBatchSamplerV2(concat_dataset=self.dataset['train'],
            #                                          batch_size=self.batch_size)
            ratio_batch_sampler = CustomBatchSamplerV4(concat_dataset=self.dataset['train'],
                                                     batch_size=self.batch_size,
                                                     mix_percentages=self.ds_perc)

            # ratio_batch_sampler = PercBatchSampler(data_source=self.dataset['train'],
            #                                        baxtch_size=self.batch_size)
                                                #    dataset_percentages=self.ds_perc)
            del self.dataloader_options['batch_size']
            return DataLoader(self.dataset['train'],
                              batch_sampler=ratio_batch_sampler,
                              **self.dataloader_options)
        else:
            return DataLoader(self.dataset['train'],
                              shuffle=True,
                              **self.dataloader_options)

    def val_dataloader(self):
        if self.batch_sampler is not None:
            return DataLoader(self.dataset['test'],
                              #batch_sampler=ratio_batch_sampler,
                             shuffle=False,
                              **self.dataloader_options)
        else:
            return DataLoader(self.dataset['test'],
                             shuffle=False,
                              **self.dataloader_options)

    def test_dataloader(self):
        return DataLoader(self.dataset['test'],
                          shuffle=False,
                          **self.dataloader_options)

    # def train_dataloader(self):
    #     return DataLoader(self.train_dataset,
    #                       shuffle=True,
    #                       **self.dataloader_options)

    # def predict_dataloader(self):
    #     return DataLoader(self.train_dataset,
    #                       shuffle=False,
    #                       **self.dataloader_options)

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset,
    #                       shuffle=False,
    #                       **self.dataloader_options)

    # def test_dataloader(self):
    #     return DataLoader(self.test_dataset,
    #                       shuffle=False,
    #                       **self.dataloader_options)

    # def subset_dataloader(self):
    #     return DataLoader(self.subset_dataset,
    #                       shuffle=False,
    #                       **self.dataloader_options)
