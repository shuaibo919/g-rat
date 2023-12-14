import argparse
import os

"""
Parameters lambda(guide and lambda(match) sensitivity analysis
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameter analysis")
    parser.add_argument('--aspect', type=int, default=2)
    parser.add_argument('--mode', type=str, default="guide", help="guide/match")
    args = vars(parser.parse_args())
    aspect = args['aspect']
    mode = args['mode']
    lambda_range = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20]
    aspect_to_sparsity = [0.15, 0.1, 0.1]
    sparsity = aspect_to_sparsity[aspect]
    if mode == 'guide':
        lambda_range = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10, 20, 30]
    for i, item in enumerate(lambda_range):
        skew_intensity = 70
        if mode == 'guide':
            cmd = f"python run_guide.py \
                      --aspect {aspect} \
                      --train_path data/beer/reviews.aspect{aspect}.train.txt \
                      --dev_path data/beer/reviews.aspect{aspect}.heldout.txt \
                      --test_path data/beer/annotations.json \
                      --max_length 256 \
                      --save_name guide_st_analysis_{mode} \
                      --fix_embedding \
                      --lr 0.0002 \
                      --sparsity {sparsity} \
                      --sparsity_lambda 10 \
                      --continuity_lambda 10 \
                      --match_lambda 0.0 \
                      --guide_decay 1e-4  \
                      --init_with_bias \
                      --skew_intensity {skew_intensity} \
                      --guide_lambda {item} \
                      --model sep > results/result_{mode}_{aspect}_{i}.txt"
        else:
            cmd = f"python run_guide.py \
                      --aspect {aspect} \
                      --train_path data/beer/reviews.aspect{aspect}.train.txt \
                      --dev_path data/beer/reviews.aspect{aspect}.heldout.txt \
                      --test_path data/beer/annotations.json \
                      --max_length 256 \
                      --save_name guide_st_analysis_{mode} \
                      --fix_embedding \
                      --lr 0.0002 \
                      --sparsity {sparsity} \
                      --sparsity_lambda 10 \
                      --continuity_lambda 10 \
                      --guide_lambda 0.0 \
                      --match_lambda {item} \
                      --guide_decay 1e-4  \
                      --model sep > results/result_{mode}_{aspect}_{i}.txt"

        os.system(cmd)
        print(f'group-{i} finished.')
