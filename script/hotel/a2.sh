#!/usr/bin/env bash
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
  --num_epochs 100 \
  --save_name st_model \
  --fix_embedding \
  --lr 0.0001  \
  --sparsity 0.095 \
  --sparsity_lambda 10 \
  --continuity_lambda 12 \
  --model sep