from torch.utils.data import BatchSampler, Sampler
import numpy as np
import math
from itertools import cycle
import random

class PercBatchSampler(BatchSampler):
    def __init__(self, data_source, batch_size, dataset_percentages, num_gpus=1):
        self.data_source = data_source
        self.batch_size = batch_size
        self.dataset_percentages = dataset_percentages
        self.num_gpus = num_gpus
        self.adjusted_batch_size = self.batch_size * self.num_gpus  # Total batch size for all GPUs
        self.batches = self._precompute_batches()
        super().__init__(self.batches, batch_size, drop_last=False)  # Initialize the BatchSampler superclass

    def _group_indices_by_dataset(self):
        dataset_indices = {}
        for idx, item in enumerate(self.data_source):
            dataset_name = item['dataset_name']
            if dataset_name not in dataset_indices:
                dataset_indices[dataset_name] = []
            dataset_indices[dataset_name].append(idx)
        return dataset_indices

    def _precompute_batches(self):
        dataset_indices = self._group_indices_by_dataset()
        samples_per_dataset = {
            name: int(round(self.adjusted_batch_size * perc / self.num_gpus))
            for name, perc in self.dataset_percentages.items()
            if name in dataset_indices
        }

        # Adjust for proper distribution across GPUs
        for name, count in samples_per_dataset.items():
            if count % self.num_gpus != 0:
                samples_per_dataset[name] += self.num_gpus - (count % self.num_gpus)

        dataset_cycles = {
            name: cycle(indices)
            for name, indices in dataset_indices.items()
        }

        total_items = sum(len(indices) for indices in dataset_indices.values())
        total_batches = math.ceil(total_items / self.adjusted_batch_size)
        batches = []

        for _ in range(total_batches):
            batch = []
            for dataset_name, count in samples_per_dataset.items():
                batch.extend([next(dataset_cycles[dataset_name]) for _ in range(count)])
            random.shuffle(batch)
            batches.append(batch)
        return batches

    def __iter__(self):
        np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

# Using the sampler in DataLoader
# sampler = PercBatchSampler(data_source, batch_size, dataset_percentages, num_gpus=4)
# data_loader = DataLoader(dataset, batch_sampler=sampler)

