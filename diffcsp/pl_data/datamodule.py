import random
from typing import Optional, Sequence
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
# from torch.utils.data import RandomSampler, Sampler
from torch.utils.data.sampler import Sampler

from diffcsp.common.utils import PROJECT_ROOT
from diffcsp.common.data_utils import get_scaler_from_data_list
import math

import heapq
import torch
from collections import deque
from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
from typing import Callable, Optional

def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)
    
class OverfitSampler(Sampler):
    def __init__(self, num_samples: int, dataset_len: int):
        self.num_samples = num_samples
        self.dataset_len = dataset_len

    def __iter__(self):
        idx_list = list(range(self.dataset_len))
        num_repeats = math.ceil(self.num_samples / self.dataset_len)
        full_list = (idx_list * num_repeats)[:self.num_samples]
        return iter(random.sample(full_list, len(full_list)))

    def __len__(self):
        return self.num_samples
    
class DynamicBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset,
        max_batch_units: int = 1000,
        max_batch_size: Optional[int] = None,
        drop_last: bool = False,
        distributed: bool = False,
        sort_key: Callable = None,
        buffer_size_multiplier: int = 100,
        shuffle: bool = False,
        use_heap: bool = False,
    ):
        """
        Batch sampler that dynamically groups samples into batches based on a user-defined size metric (e.g., number of tokens, atoms, nodes).
        Each batch is constructed to stay within a maximum total unit count (`max_batch_units`), allowing for efficient batching of variable-sized data.

        Examples:
        - For tokenized text sequences: sort_key = lambda i: len(dataset[i][0])  # number of tokens
        - For molecular graphs: sort_key = lambda i: dataset[i].num_atoms        # number of atoms
        - For general graphs: sort_key = lambda i: dataset[i].num_nodes          # number of nodes
        """
        self.distributed = distributed
        if distributed:
            self.sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            self.sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

        super().__init__(self.sampler, batch_size=1, drop_last=drop_last)

        self.max_batch_units = max_batch_units
        self.max_batch_size = max_batch_size
        self.sort_key = sort_key
        self.max_buffer_size = max_batch_units * buffer_size_multiplier
        self._epoch = 0
        self.shuffle = shuffle
        self.use_heap = use_heap # Useful if padding is involved (e.g., for tokens)
        self.drop_last = drop_last

        self.bucket_batches = []

    def __len__(self):
        if not self.bucket_batches:
            self._build_batches()
        return len(self.bucket_batches)

    def __iter__(self):
        self._build_batches()
        for batch, _ in self.bucket_batches:
            yield batch

    def _build_batches(self):
        buffer = []
        buffer_deque = deque()  # Use deque for FIFO when use_heap=False
        buffer_size = 0

        batch = []
        batch_units = 0

        bucket_batches = []

        indices = list(self.sampler)
        for index in indices:
            # Add to buffer
            num_units = self.sort_key(index)
            if self.use_heap:
                # Store negative to simulate max-heap (largest first)
                heapq.heappush(buffer, (-num_units, index))
            else:
                buffer_deque.append((num_units, index))
            buffer_size += num_units

            # Flush buffer if exceeds max buffer size
            while buffer_size > self.max_buffer_size:
                if self.use_heap:
                    neg_units, index = heapq.heappop(buffer)
                    num_units = -neg_units
                else:
                    num_units, index = buffer_deque.popleft()
                buffer_size -= num_units

                # Check batch constraints
                if (batch_units + num_units > self.max_batch_units) or \
                   (self.max_batch_size and len(batch) >= self.max_batch_size):
                    bucket_batches.append((batch, batch_units))
                    batch, batch_units = [], 0
                batch.append(index)
                batch_units += num_units

        # Process remaining elements in buffer
        while buffer if self.use_heap else buffer_deque:
            if self.use_heap:
                neg_units, index = heapq.heappop(buffer)
                num_units = -neg_units
            else:
                num_units, index = buffer_deque.popleft()
            if (batch_units + num_units > self.max_batch_units) or \
               (self.max_batch_size and len(batch) >= self.max_batch_size):
                bucket_batches.append((batch, batch_units))
                batch, batch_units = [], 0

            batch.append(index)
            batch_units += num_units

        # Handle last batch
        if batch and not self.drop_last:
            bucket_batches.append((batch, batch_units))

        # Extra randomization for use_heap
        if self.shuffle and self.use_heap:
            np.random.shuffle(bucket_batches)

        # DDP synchronization
        if self.distributed:
            # Communicate the number of batches across processes
            num_batches = torch.tensor(len(bucket_batches), device='cuda')
            dist.all_reduce(num_batches, op=dist.ReduceOp.MIN)
            num_batches = num_batches.item()

            # Truncate to the minimum number of batches across all processes
            if len(bucket_batches) > num_batches:
                bucket_batches = bucket_batches[:num_batches]
        
        self.bucket_batches = bucket_batches

    def set_epoch(self, epoch):
        self._epoch = epoch
        if self.distributed:
            self.sampler.set_epoch(epoch)

class CrystDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        # download only
        pass

    def get_scaler(self, scaler_path):
        # Load once to compute property scaler
        if scaler_path is None:
            train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key='scaled_lattice')
            self.scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key=train_dataset.prop)
        else:
            try:
                self.lattice_scaler = torch.load(
                    Path(scaler_path) / 'lattice_scaler.pt')
                self.scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')
            except:
                train_dataset = hydra.utils.instantiate(self.datasets.train)
                self.lattice_scaler = get_scaler_from_data_list(
                    train_dataset.cached_data,
                    key='scaled_lattice')
                self.scaler = get_scaler_from_data_list(
                    train_dataset.cached_data,
                    key=train_dataset.prop)

    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            for val_dataset in self.val_datasets:
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.scaler = self.scaler

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]
            for test_dataset in self.test_datasets:
                test_dataset.lattice_scaler = self.lattice_scaler
                test_dataset.scaler = self.scaler

    def train_dataloader(self, shuffle = True) -> DataLoader:
    
    # def train_dataloader(self) -> DataLoader:
        
        is_train = True
        is_distributed = is_train and torch.distributed.is_initialized()
        
        return DataLoader(
            self.train_dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
        )
        
        # sampler = OverfitSampler(num_samples=154500, dataset_len=len(self.train_dataset))
        # # sampler = None
        # batch_sampler = DynamicBatchSampler(
        #     dataset=self.train_dataset,
        #     max_batch_units=2048,  # Total atoms per batch
        #     distributed=is_distributed,
        #     shuffle=is_train,
        #     sort_key=lambda i: self.train_dataset[i].num_atoms,
        #     use_heap=True
        # )
        
        # dataloader = DataLoader(
        #     dataset=self.train_dataset,
        #     # When using batch_sampler, these options are mutually exclusive:
        #     # shuffle, batch_size, sampler, drop_last
        #     batch_sampler=batch_sampler,
        #     num_workers=self.num_workers.train,
        #     worker_init_fn=worker_init_fn,
        # )
        
        # return dataloader

    def val_dataloader(self) -> Sequence[DataLoader]:
        
        is_train = False
        is_distributed = is_train and torch.distributed.is_initialized()

        # return [
        #     DataLoader(
        #         dataset,
        #         shuffle=False,
        #         batch_size=self.batch_size.val,
        #         num_workers=self.num_workers.val,
        #         sampler= OverfitSampler(num_samples=11000, dataset_len=len(dataset)),
        #         worker_init_fn=worker_init_fn,
        #     )
        #     for dataset in self.val_datasets
        # ]
        
        
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.val_datasets
        ]
        
        # sampler = None
        
        # return [
        #     DataLoader(
        #         dataset,
        #         # When using batch_sampler, these options are mutually exclusive:
        #         # shuffle, batch_size, sampler, drop_last
        #         batch_sampler=DynamicBatchSampler(
        #             dataset=dataset, 
        #             max_batch_units=2048,  # Total atoms per batch 
        #             distributed=is_distributed, 
        #             shuffle=is_train, 
        #             sort_key=lambda i: dataset[i].num_atoms, 
        #             use_heap=True
        #         ),
        #         num_workers=self.num_workers.val,
        #         worker_init_fn=worker_init_fn,
        #     )
        #     for dataset in self.val_datasets
        # ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        
        is_train = False
        is_distributed = is_train and torch.distributed.is_initialized()

        # return [
        #     DataLoader(
        #         dataset,
        #         shuffle=False,
        #         batch_size=self.batch_size.test,
        #         num_workers=self.num_workers.test,
        #         sampler=OverfitSampler(num_samples=11000, dataset_len=len(dataset)),
        #         worker_init_fn=worker_init_fn,
        #     )
        #     for dataset in self.test_datasets
        # ]

        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
            )
            for dataset in self.test_datasets
        ]
        
        # sampler = None
        
        # return [
        #     DataLoader(
        #         dataset,
        #         # When using batch_sampler, these options are mutually exclusive:
        #         # shuffle, batch_size, sampler, drop_last
        #         batch_sampler=DynamicBatchSampler(
        #             dataset=dataset, 
        #             max_batch_units=2048,  # Total atoms per batch 
        #             distributed=is_distributed, 
        #             shuffle=is_train, 
        #             sort_key=lambda i: dataset[i].num_atoms, 
        #             use_heap=True
        #         ),
        #         num_workers=self.num_workers.test,
        #         worker_init_fn=worker_init_fn,
        #     )
        #     for dataset in self.test_datasets
        # ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
