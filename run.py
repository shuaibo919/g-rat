import os
import torch
import argparse

from model import RNPSTModel, RNPSTShareModel
from train_utils import prepare_dataset, show_config, try_gpu, run_on_annotations, set_seed, train, valid

MODEL_TYPE = {
    'sep': RNPSTModel,  # Re-RNP
    'share': RNPSTShareModel,  # FR
}


def get_share_args():
    parser = argparse.ArgumentParser(description="Straight Through Training")

    # Beer specific arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--aspect', type=int, default=0)
    parser.add_argument('--embeddings', type=str, default="data/glove.6B.100d.txt",
                        help="path to external embeddings")
    parser.add_argument('--dataset', type=str, default="beer")
    parser.add_argument('--train_path', type=str, default="data/beer/reviews.aspect0.train.txt")
    parser.add_argument('--dev_path', type=str, default="data/beer/reviews.aspect0.heldout.txt")
    parser.add_argument('--test_path', type=str, default="data/beer/annotations.json")
    parser.add_argument('--max_length', type=int, default=256, help="maximum input length (skip)")
    parser.add_argument('--save_path', type=str, default='results/beer/')
    parser.add_argument('--save_name', type=str, default='model_beer')
    parser.add_argument('--fix_embedding', action='store_true', default=False)
    parser.add_argument('--print_rationale', action='store_true', default=False, help="print rationales in testing")
    parser.add_argument('--print_rationale_nums', type=int, default=2)
    parser.add_argument('--task_type', type=str, default='classification')
    # general arguments
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--embedding_size', type=int, default=100)
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--output_size', type=int, default=2)
    parser.add_argument('--cell_type', type=str, default='gru', help="rcnn/lstm/gru")
    # optimization
    parser.add_argument('--weight_decay', type=float, default=2e-6)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.0004)
    # regularization for nums of selection
    parser.add_argument('--sparsity', type=float, default=None,
                        help="select this ratio of input words (e.g. 0.13 for 13%)")
    # regularization for Bernoulli/gumbel-softmax model
    parser.add_argument('--sparsity_lambda', type=float, default=0.0003)
    parser.add_argument('--continuity_lambda', type=float, default=2.)
    # share setting
    parser.add_argument('--model', type=str, default='share', help='sep/share/rl')
    # gumbel-softmax temperature
    parser.add_argument('--tau', type=float, default=1.)
    parser.add_argument('--tau_decay', type=float, default=1.)
    # save model
    parser.add_argument('--history_performance', type=float, default=0.8)
    parser.add_argument('--init_with_bias', action='store_true', default=False)
    parser.add_argument('--skew_name', type=str, default='skew_st')
    parser.add_argument('--skew_type', type=str, default='selector', help='predictor/selector')
    parser.add_argument('--skew_intensity', type=int, default=60)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    configs = get_share_args()
    configs = vars(configs)

    print('CONFIG DETAIL:')
    configs = show_config(configs)

    print('DEVICE:')
    train_device = try_gpu(configs['gpu_id'])
    print(f'device:{train_device}')

    print('LOADING DATA:')
    vocab, train_dataloader, dev_dataloader, test_dataloader = prepare_dataset(configs)
    set_seed(configs['seed'])

    print('LOADING MODEL:')
    instantiating_class = MODEL_TYPE[configs['model']]
    model = instantiating_class(
        vocab_size=vocab.vocab_size,
        emb_size=configs['embedding_size'],
        hidden_size=configs['hidden_size'],
        output_size=configs['output_size'],
        dropout=configs['dropout'],
        sparsity=configs['sparsity'],
        continuity_lambda=configs['continuity_lambda'],
        sparsity_lambda=configs['sparsity_lambda'],
        pretrained_embedding=vocab.embedding_matrix,
        cell_type=configs['cell_type'],
        fix_embedding=configs['fix_embedding'],
    )

    if configs['init_with_bias']:
        model_path = f"results/beer/{configs['skew_name']}_{configs['cell_type']}_{configs['aspect']}.{configs['skew_intensity']}"
        if configs['skew_type'] == 'selector':
            print(f'load skew selector:{model_path}')
            cell_state_dict = torch.load(model_path + '.selector_cell.pt', map_location=torch.device('cpu'))
            if configs['model'] != 'share':
                model.selector_cell.load_state_dict(cell_state_dict)
            else:
                model.share_cell.load_state_dict(cell_state_dict)
            model.selector_head.load_state_dict(
                torch.load(model_path + '.selector_head.pt', map_location=torch.device('cpu'))
            )
        if configs['skew_type'] == 'predictor':
            print(f'load skew predictor:{model_path}')
            cell_state_dict = torch.load(model_path + '.predictor_cell.pt', map_location=torch.device('cpu'))
            if configs['model'] != 'share':
                model.selector_cell.load_state_dict(cell_state_dict)
            else:
                model.share_cell.load_state_dict(cell_state_dict)
            model.selector_head.load_state_dict(
                torch.load(model_path + '.predictor_head.pt', map_location=torch.device('cpu'))
            )

    print('TRAINING START:')
    save_name = f"{configs['save_name']}_{configs['cell_type']}_{configs['task_type']}_{configs['aspect']}.pt"
    save_name = configs['save_path'] + save_name
    model.to(train_device)
    optimizer = torch.optim.Adam(model.parameters(), configs['lr'], weight_decay=configs['weight_decay'])
    for epoch_id in range(configs['num_epochs']):
        print(f'EPOCH {epoch_id}:')
        # train
        train(model, optimizer, train_dataloader, train_device, configs)
        # validation
        performance, sparsity_rate = valid(model, dev_dataloader, train_device, configs)
        if configs['task_type'] == 'regression':
            saving_condition = (performance < configs['history_performance'] and epoch_id > 10)
        else:
            saving_condition = (performance > configs['history_performance'] and epoch_id > 10)

        temp = (configs['sparsity'] - 0.035) < sparsity_rate < (configs['sparsity'] + 0.035)
        saving_condition = saving_condition and temp

        if saving_condition:
            print(f'New High Performance!{performance}')
            configs['history_performance'] = performance
            print(f'saving model')
            torch.save(model.state_dict(), save_name)
        # test
        run_on_annotations(model, test_dataloader, vocab, train_device, configs, save_name)
        configs['history_performance'] = performance

    print('<-------------:Loading the Best:------------->')
    if os.path.exists(save_name):
        model.load_state_dict(
            torch.load(save_name, map_location=train_device)
        )
        run_on_annotations(model, test_dataloader, vocab, train_device, configs, save_name)
    else:
        print(f'no model saved in {save_name}')
        print(f'so save last epoch model in {save_name} and evaluating')
        torch.save(model.state_dict(), save_name)
        model.load_state_dict(torch.load(save_name, map_location=train_device))
        run_on_annotations(model, test_dataloader, vocab, train_device, configs, save_name)
