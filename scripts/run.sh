#!/bin/bash

source ~/.profile
ROOT=$PWD

torchrun \
    --nnodes=8 \
    --nproc-per-node=8 \
    --rdzv-id=dipaco.run \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    "$ROOT/src/dipaco/train.py" \
    --config-path "$ROOT/configs/C4" \
    --config-name c4_dipaco_7b_64.yaml
