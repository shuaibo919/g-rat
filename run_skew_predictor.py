import argparse
import os

"""
Experiment of Skew-Predictor
python run_skew_predictor.py --aspect 1
python run_skew_predictor.py --aspect 2
"""

aspect_to_sparsity = [None, 0.09, 0.09]
lambda_range = [10, 15, 20]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameter analysis")
    parser.add_argument('--aspect', type=int, default=1)
    args = vars(parser.parse_args())
    aspect = args['aspect']
    if aspect == 0:
        raise ValueError('see A2R(https://arxiv.org/abs/2110.13880)')

    model = 'guide'
    sparsity = aspect_to_sparsity[aspect]
    for i, item in enumerate(lambda_range):
        cmd = f"python run_guide.py \
                --aspect {aspect} \
                --num_epochs 300 \
                --train_path data/beer/reviews.aspect{aspect}.train.txt \
                --dev_path data/beer/reviews.aspect{aspect}.heldout.txt \
                --test_path data/beer/annotations.json \
                --max_length 256 \
                --save_name {model}_model_bad_init_{item} \
                --fix_embedding \
                --lr 0.0002 \
                --sparsity {sparsity} \
                --sparsity_lambda 10 \
                --continuity_lambda 10 \
                --guide_lambda 5. \
                --match_lambda 0.9 \
                --guide_decay 1e-4 \
                --init_with_bias \
                --skew_intensity {item} \
                --skew_name skew_st \
                --skew_type predictor \
                --model sep > results/result_{model}_predictor{aspect}_bad_init_{item}.txt"

        os.system(cmd)
        print(f'group-{i} finished.')
