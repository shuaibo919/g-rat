aspect=1
python pretrain_skew.py \
  --aspect $aspect \
  --batch_size 500 \
  --max_length 128 \
  --lr 0.0001 \
  --train_path data/beer/reviews.aspect$aspect.train.txt \
  --dev_path data/beer/reviews.aspect$aspect.heldout.txt \
  --save_name skew_st \
  --fix_embedding \
  --skew_type predictor
