# Run Instructions

## Installation
Run the following commands on a machine equipped with Nvidia GPUs.
```commandline
pip install .
```

## Running experiments

### Configs
Prepare the yaml configs (examples can be found under configs/no_reduce/C4) by modifying the c4_base.yaml to incorporate the correct HuggingFace access token and ensure that you have access (with the token) to download the Llama3 tokenizer. 
Add the correct path to the data loaders that point to the local copy of the raw HuggingFace C4 english partition.

```commandline
train_data_loader:
  _target_: dipaco.c4_data_loader.TokenizedHuggingFaceShardedDataset
  partition: train
  tokenizer_path: *tokenizer
  sequence_length: *sequence_length
  access_token: *access_token
  path: *path_local_c4_en_copy
```

### Running
Go to the script folder and modify run.sh to include the config you want to run (e.g. c4_diloco_100m_8.yaml):
```commandline
#!/bin/bash

source ~/.profile
ROOT=$PWD

torchrun \
    --nnodes=1 \
    --nproc-per-node=8 \
    --rdzv-id=dipaco.run \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$ROOT/src/dipaco/train.py" \
    --config-path "$ROOT/configs/C4" \
    --config-name c4_diloco_100m_8.yaml
```

Note that the total number of workers is dictated by nnodes and nproc-per-node (in this case 8).
Launch the training by running the script in every node that should participate.
The environment variables `MASTER_ADDR` and `MASTER_PORT` should be set to match one of the nodes with a port and IP address that is reachable from all nodes.
