import torch
import glob
import os
import torch.distributed as dist
from typing import List
from transformers import AutoTokenizer
from datasets import load_dataset, VerificationMode


def _get_rank(group=None):
    if dist.is_initialized():
        return dist.get_rank(group=group)
    else:
        return 0


def _get_world_size(group=None):
    if dist.is_initialized():
        return dist.get_world_size(group=group)
    else:
        return 1


def create_dataset(path, partition, shard, pg):
    rank = _get_rank(group=pg)
    world_size = _get_world_size(group=pg)
    all_files = [
        os.path.basename(f)
        for f in glob.glob(os.path.join(path, f"c4-{partition}.*.json.gz"))
    ]
    data_files = (
        [f for i, f in enumerate(all_files) if i % world_size == rank ]
        if shard
        else all_files
    )
    return load_dataset(
        path,
        data_files=data_files,
        streaming=True,
        split="train",
        verification_mode=VerificationMode.NO_CHECKS,
    )


class TokenizedHuggingFaceShardedDataset(torch.utils.data.IterableDataset):

    def __init__(
        self, 
        partition: str,
        tokenizer_path: str,
        sequence_length: int,
        path="/home/gensyn/shared/data/huggingface/c4/en",
        shard: bool = True,
        shard_on_samples: bool = False,
        access_token=None,
        pg=None
    ):
        self.partition = partition
        self.path = path
        self.sequence_length = sequence_length
        self.shard = shard
        self.shard_on_samples = shard_on_samples
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            token=access_token,
        )
        self.pg = pg

        # Overwrite file sharding
        if self.shard_on_samples:
            self.shard = False


    def __iter__(self):
        rank = _get_rank(group=self.pg)
        world_size = _get_world_size(group=self.pg)
        dataset = create_dataset(self.path, self.partition, self.shard, self.pg)
        buffer: List[int] = []
        target_length = self.sequence_length + 1
        for index, raw_data in enumerate(dataset):
            # Dataset is already sharded across workers if shard=True
            if self.shard_on_samples and index % world_size != rank:
                continue
            text = raw_data["text"]
            tokens = self.tokenizer.encode(text)
            buffer += tokens

            while len(buffer) > target_length:
                input_ids = buffer[:target_length]
                label = buffer[1 : (target_length + 1)]
                buffer = buffer[target_length:]
                yield {
                    "input_ids": torch.tensor(input_ids, dtype=torch.int64, device="cpu"),
                    "label": torch.tensor(label, dtype=torch.int64, device="cpu"),
                }
