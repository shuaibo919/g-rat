import argparse
import os

"""
Experiment of Skew-Selector
python run_skew_selector.py --model sep --aspect 1
python run_skew_selector.py --model share --aspect 1
python run_skew_selector.py --model guide --aspect 1
"""

aspect_to_sparsity = [0.15, 0.1, 0.1]
lambda_range = [55, 60, 65, 70, 75, 80]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parameter analysis")
    parser.add_argument('--aspect', type=int, default=0)
    parser.add_argument('--model', type=str, default="sep", help="sep/share/guide")
    args = vars(parser.parse_args())
    aspect = args['aspect']
    model = args['model']
    sparsity = aspect_to_sparsity[aspect]
    for i, item in enumerate(lambda_range):
        if model == 'share' or model == 'sep':
            cmd = f"python run.py \
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
                      --init_with_bias \
                      --skew_intensity {item} \
                      --model {model} > results/result_{model}_{aspect}_bad_init_{item}.txt"
        elif model == 'guide':
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
                    --model sep > results/result_{model}_{aspect}_bad_init_{item}.txt"

        os.system(cmd)
        print(f'group-{i} finished.')
