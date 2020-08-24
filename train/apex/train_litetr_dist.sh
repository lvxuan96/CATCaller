#!/bin/bash
python -m torch.distributed.launch --nproc_per_node=4 \
path/to/train_litetr_dist.py \
-as path/to/signal_out_path/train \
-al path/to/label_out_path/train \
-es path/to/signal_out_path/valid \
-el path/to/label_out_path/valid  \
--save_model path/to/save/trained_model/model.2048 \
--store_model True > train.log 2>&1