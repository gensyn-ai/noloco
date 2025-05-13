import torch
import torch.distributed as dist
from datasets.io.parquet import ParquetDatasetReader


class ShardedDataset(torch.utils.data.IterableDataset):
    def __init__(self, path: str, shard: bool = True, pg=None):
        self.data_reader = ParquetDatasetReader(path, streaming=True).read()
        # self.data_reader.set_format("torch")
        self.shard = shard
        self.pg = pg

    def __iter__(self):
        rank = dist.get_rank(group=self.pg)
        world_size = dist.get_world_size(group=self.pg)
        for index, sample in enumerate(self.data_reader):
            if not self.shard or index % world_size == rank:
                yield {
                    key: torch.tensor(value, dtype=torch.int64, device="cpu")
                    for key, value in sample.items()
                }
