aspect=2
python run_guide.py \
  --aspect $aspect \
  --train_path data/beer/reviews.aspect$aspect.train.txt \
  --dev_path data/beer/reviews.aspect$aspect.heldout.txt \
  --test_path data/beer/annotations.json \
  --max_length 256 \
  --num_epochs 100 \
  --save_name guide_st \
  --fix_embedding \
  --batch_size 64 \
  --lr 0.0001 \
  --sparsity 0.1 \
  --sparsity_lambda 10 \
  --continuity_lambda 10 \
  --guide_lambda 10.0 \
  --guide_decay 1e-5 \
  --match_lambda 0.2 \
  --model sep