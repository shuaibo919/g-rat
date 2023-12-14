aspect=0
python run_guide.py \
  --aspect $aspect \
  --train_path data/beer/reviews.aspect$aspect.train.txt \
  --dev_path data/beer/reviews.aspect$aspect.heldout.txt \
  --test_path data/beer/annotations.json \
  --max_length 256 \
  --save_name guide_st \
  --fix_embedding \
  --lr 0.0001 \
  --sparsity 0.15 \
  --sparsity_lambda 10 \
  --continuity_lambda 10 \
  --guide_lambda 10 \
  --guide_decay 1e-5 \
  --match_lambda 1.5 \
  --model sep
