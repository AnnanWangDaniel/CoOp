#!/bin/bash

# custom config
DATA=/home/FYP/c190190/DATA
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only