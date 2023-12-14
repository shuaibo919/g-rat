aspect=2
python run.py \
  --aspect $aspect \
  --train_path data/beer/reviews.aspect$aspect.train.txt \
  --dev_path data/beer/reviews.aspect$aspect.heldout.txt \
  --test_path data/beer/annotations.json \
  --max_length 256 \
  --num_epochs 500 \
  --batch_size 64 \
  --save_name share_model \
  --fix_embedding \
  --lr 0.0001 \
  --sparsity 0.12 \
  --sparsity_lambda 10 \
  --continuity_lambda 10 \
  --model share
