#!/bin/bash
arr=("Location" "Service" "Cleanliness")
aspect=2
python run.py \
  --aspect $aspect \
  --dataset hotel \
  --train_path "data/hotel/hotel_${arr[aspect]}.train" \
  --dev_path "data/hotel/hotel_${arr[aspect]}.dev" \
  --test_path "data/hotel/annotations/hotel_${arr[aspect]}.train" \
  --save_path "results/hotel/" \
  --max_length 256 \
  --save_name st_model \
  --fix_embedding \
  --lr 0.0001  \
  --sparsity 0.1 \
  --sparsity_lambda 8 \
  --continuity_lambda 10 \
  --model share