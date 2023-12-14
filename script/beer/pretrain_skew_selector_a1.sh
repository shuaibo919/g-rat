aspect=1
python pretrain_skew.py \
  --aspect $aspect \
  --train_path data/beer/reviews.aspect$aspect.train.txt \
  --dev_path data/beer/reviews.aspect$aspect.heldout.txt \
  --save_name skew_st \
  --fix_embedding \
  --sparsity 0.1