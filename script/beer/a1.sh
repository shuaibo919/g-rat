aspect=1
python run.py \
  --aspect $aspect \
  --train_path data/beer/reviews.aspect$aspect.train.txt \
  --dev_path data/beer/reviews.aspect$aspect.heldout.txt \
  --test_path data/beer/annotations.json \
  --max_length 256 \
  --save_name st_model \
  --fix_embedding \
  --num_epochs 500 \
  --lr 0.0001 \
  --sparsity 0.123 \
  --sparsity_lambda 12 \
  --continuity_lambda 10 \
  --model sep