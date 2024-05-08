import numpy as np
import random
from torch.utils.data import Sampler, ConcatDataset

class PercBatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.datasets = data_source
        self.batch_size = batch_size
        self.weights = self._calculate_weights()
        self.total_size = sum(len(d) for d in data_source)

    def _calculate_weights(self):
        sizes = [len(d) for d in self.datasets.datasets]
        inv_prob = 1 / np.array(sizes, dtype=float)  # Ensure array of floats for division
        inv_prob /= inv_prob.sum()                   # Normalize
        weights = np.concatenate([np.full(size, fill_value=weight) for size, weight in zip(sizes, inv_prob)])
        return weights

    def __iter__(self):
        indices = list(range(self.total_size))
        random.shuffle(indices)  # Shuffle all indices
        chosen_indices = random.choices(indices, weights=self.weights, k=self.total_size)  # Weighted sampling without replacement
        batches = [chosen_indices[i:i + self.batch_size] for i in range(0, self.total_size, self.batch_size)]
        return iter(batches)

    def __len__(self):
        return (self.total_size + self.batch_size - 1) // self.batch_size

# Example usage assuming `datasets` is a list of PyTorch Dataset objects
# batch_sampler = PercBatchSampler(datasets, batch_size=10)
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
import numpy as np

class CustomBatchSampler:
    def __init__(self, data_source, batch_size):
        self.concat_dataset = data_source
        self.batch_size = batch_size
        self.sampler = self.create_weighted_sampler(data_source)

    def create_weighted_sampler(self, concat_dataset):
        # Extract individual dataset sizes from ConcatDataset
        dsize = [len(d) for d in concat_dataset.datasets]
        min_len = min(dsize)
        inv_prob = 1 / np.array(dsize, dtype=float)
        inv_prob /= sum(inv_prob)  # Normalize probabilities

        # Create weights for each sample in the ConcatDataset
        sampler_weights = np.array([])
        for ds, ip in zip(dsize, inv_prob):
            sampler_weights = np.append(sampler_weights, ip * np.ones(ds))

        # Compute epoch size based on the smallest dataset times number of datasets
        epoch_size = int(len(concat_dataset.datasets) * min_len)
        return WeightedRandomSampler(weights=sampler_weights, num_samples=epoch_size, replacement=False)


    def __iter__(self):
        indices = list(range(len(self.concat_dataset)))
        chosen_indices = np.random.choice(indices, size=len(indices), replace=False, p=self.weights)
        batch_indices = [chosen_indices[i:i + self.batch_size] for i in range(0, len(chosen_indices), self.batch_size)]
        for batch in batch_indices:
            yield batch
    # def __iter__(self):
    #     batch = []
    #     for idx in self.sampler:
    #         batch.append(idx)
    #         if len(batch) == self.batch_size:
    #             yield batch
    #             batch = []
    #     if batch:
    #         yield batch  # Yield any remaining items as the last batch

    def __len__(self):
        return (len(self.concat_dataset) + self.batch_size - 1) // self.batch_size

# Assuming your ConcatDataset is already defined and initialized
# concat_dataset = ConcatDataset([YourDataset1(), YourDataset2(), YourDataset3()])
# batch_size = 10

# # Create an instance of your custom batch sampler
# ratio_batch_sampler = CustomBatchSampler(concat_dataset, batch_size)

# # Create DataLoader with the custom batch sampler
# dataloader_options = {'num_workers': 4, 'pin_memory': True}
# train_dataloader = DataLoader(concat_dataset,
#                               batch_sampler=ratio_batch_sampler,
#                               **dataloader_options)

# # Example of iterating over the DataLoader
# for batch in train_dataloader:
#     # process_batch(batch)  # Implement your batch processing logic
#     print(batch)  # Just print batch indices for demonstration
import torch
import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Sampler

class CustomBatchSamplerV2(Sampler):
    def __init__(self, concat_dataset, batch_size):
        self.concat_dataset = concat_dataset
        self.batch_size = batch_size
        self.sizes = [len(d) for d in concat_dataset.datasets]
        self.weights = self._calculate_weights()
        self.epoch_size = len(concat_dataset.datasets) * min(self.sizes)  # Samples per epoch

    def _calculate_weights(self):
        inv_prob = np.array([1.0 / size for size in self.sizes])
        inv_prob /= np.sum(inv_prob)  # Normalize probabilities
        weights = []
        for size, weight in zip(self.sizes, inv_prob):
            weights.extend([weight] * size)
        weights = np.array(weights)
        weights /= np.sum(weights)  # Ensure weights sum to exactly 1
        return weights

    def __iter__(self):
        total_samples = self.epoch_size
        # Ensure weights sum to 1
        self.weights /= np.sum(self.weights)
        indices = np.random.choice(len(self.concat_dataset), total_samples, replace=True, p=self.weights)
        for i in range(0, total_samples, self.batch_size):
            yield indices[i:i + self.batch_size]

    def __len__(self):
        return (self.epoch_size + self.batch_size - 1) // self.batch_size


def mix_datasets_anysize(data_list):
    import torch
    # Size of each dataset in the list
    dsize = np.array([len(d) for d in data_list])
    # Determine the epoch size based on the smallest dataset
    min_len = min(dsize)
    inv_prob = 1 / dsize
    # Normalise to sum to 1
    inv_prob /= sum(inv_prob)
    sampler_weights = np.array([])
    for ds, ip in zip(dsize, inv_prob):
        sampler_weights = np.append(sampler_weights, ip * np.ones(ds))
    epoch_size = int(len(data_list) * min_len)
    
    sampler = torch.utils.data.WeightedRandomSampler(weights=sampler_weights,
                                                     num_samples=epoch_size,
                                                     replacement=False)
    return sampler