#!/bin/bash
arr=("Location" "Service" "Cleanliness")
aspect=0
python run_guide.py \
  --aspect $aspect \
  --dataset hotel \
  --train_path "data/hotel/hotel_${arr[aspect]}.train" \
  --dev_path "data/hotel/hotel_${arr[aspect]}.dev" \
  --test_path "data/hotel/annotations/hotel_${arr[aspect]}.train" \
  --save_path "results/hotel/" \
  --max_length 256 \
  --save_name guide_st \
  --fix_embedding \
  --lr 0.0002  \
  --sparsity 0.08 \
  --sparsity_lambda 8 \
  --continuity_lambda 10 \
  --guide_lambda 2.5 \
  --guide_decay 1e-5 \
  --match_lambda 1.5 \
  --model sep