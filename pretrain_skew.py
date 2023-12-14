import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
import torchmetrics.classification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from model import RNPSTModel
from train_utils import show_config, try_gpu
from preprocessing.preprocessing_beer import GloveVocabulary, BeerDataset

SAVE_POINTS = {
    # skew-selector
    'hard': [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    'simple': None,  # Deprecated
    # skew-predictor
    'predictor': [10, 15, 20]
}


def _get_induce_dev_accuracy(instance, dev_iter, device):
    induce_dev_pre = torchmetrics.classification.BinaryAccuracy()
    induce_dev_pre.to(device)
    instance.eval()
    for dev_batch in dev_iter:
        dev_batch = [item.to(device=device) for item in dev_batch]
        x, m, y = dev_batch
        l = m.float().sum(dim=-1)
        _, rationales = instance.induce_skew_selector(x, m)
        induce_dev_pre(rationales[:, 0], y.view(-1))
    return induce_dev_pre.compute()


def _get_induce_args():
    parser = argparse.ArgumentParser(description="Beer multi-aspect sentiment analysis")

    # Beer specific arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--aspect', type=int, default=0)
    parser.add_argument('--embeddings', type=str, default="data/glove.6B.100d.txt",
                        help="path to external embeddings")
    parser.add_argument('--train_path', type=str, default="data/beer/reviews.aspect0.train.txt")
    parser.add_argument('--dev_path', type=str, default="data/beer/reviews.aspect0.heldout.txt")
    parser.add_argument('--max_length', type=int, default=256, help="maximum input length (skip)")
    parser.add_argument('--save_path', type=str, default='results/beer/')
    parser.add_argument('--save_name', type=str, default='model_beer')
    parser.add_argument('--fix_embedding', action='store_true', default=False)
    parser.add_argument('--print_rationale', action='store_true', default=False, help="print rationales in testing")
    parser.add_argument('--print_rationale_nums', type=int, default=2)
    parser.add_argument('--print_model_params', action='store_true', default=False,
                        help="print model parameters in program starting.")
    parser.add_argument('--task_type', type=str, default='classification')
    # general arguments
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
    # skew setting
    parser.add_argument('--skew_type', type=str, default='hard', help='hard or simple or predictor')
    args = parser.parse_args()
    return args


def prepare():
    """
    Return: model, optimizer, train_dataloader, dev_dataloader, train_device, save_name, configs
    """
    _configs = _get_induce_args()
    _configs = vars(_configs)
    torch.manual_seed(_configs['seed'])
    np.random.seed(_configs['seed'])
    random.seed(_configs['seed'])
    _configs['save_points'] = SAVE_POINTS[_configs['skew_type']]
    print('-CONFIG DETAIL:')
    _configs = show_config(_configs)

    print('-DEVICE:')
    _train_device = try_gpu(_configs['gpu_id'])
    print(f'device:{_train_device}')

    print('-LOADING DATA:')
    vocab = GloveVocabulary(_configs['embeddings'], _configs['embedding_size'])

    _dev_dataloader = None
    if _configs['skew_type'] == 'predictor':
        train_dataset = BeerDataset(_configs['train_path'], vocab, _configs["aspect"], _configs["max_length"],
                                    convert_label_to_binary_sentiment=True, balanced=True,
                                    only_load_first_sentence=True)
    else:
        train_dataset = BeerDataset(_configs['train_path'], vocab, _configs["aspect"], _configs["max_length"],
                                    convert_label_to_binary_sentiment=True, balanced=True)
        dev_dataset = BeerDataset(_configs['dev_path'], vocab, _configs["aspect"], _configs["max_length"],
                                  convert_label_to_binary_sentiment=True, balanced=True)
        _dev_dataloader = DataLoader(
            dev_dataset, sampler=SequentialSampler(dev_dataset),
            collate_fn=dev_dataset.collate_fn, batch_size=_configs['batch_size']
        )
    _train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset),
        collate_fn=train_dataset.collate_fn, batch_size=_configs['batch_size'],
    )

    print('-LOADING MODEL:')
    _model = RNPSTModel(
        vocab_size=vocab.vocab_size,
        emb_size=_configs['embedding_size'],
        hidden_size=_configs['hidden_size'],
        output_size=2,
        dropout=_configs['dropout'],
        sparsity=_configs['sparsity'],
        continuity_lambda=0.5,
        sparsity_lambda=0.5,
        cell_type=_configs['cell_type'],
        pretrained_embedding=vocab.embedding_matrix,
        fix_embedding=_configs['fix_embedding']
    )

    if _configs['print_model_params']:
        _model.report_parameters()

    print('-INDUCING START:')
    _save_name = _configs['save_path'] + _configs['save_name'] + f"_{_configs['cell_type']}_{_configs['aspect']}.pt"

    _model.to(_train_device)
    _optimizer = torch.optim.AdamW(_model.parameters(), _configs['lr'], weight_decay=_configs['weight_decay'])

    return _model, _optimizer, _train_dataloader, _dev_dataloader, _train_device, _save_name, _configs


if __name__ == '__main__':

    model, optimizer, train_dataloader, dev_dataloader, train_device, save_name, configs = prepare()
    # binary_accuracy
    induce_acc = torchmetrics.classification.BinaryAccuracy()
    induce_pre = torchmetrics.classification.BinaryPrecision()
    induce_recall = torchmetrics.classification.BinaryRecall()
    induce_acc.to(train_device)
    induce_pre.to(train_device)
    induce_recall.to(train_device)

    save_points: list = configs['save_points']

    # train skew
    if configs['skew_type'] != 'predictor':
        while True:
            # train
            tbar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch_idx, batch in enumerate(tbar):
                model.train()
                batch = [item.to(device=train_device) for item in batch]
                input_ids, masks, labels = batch
                selector_logits, rationale_masks = model.induce_skew_selector(input_ids, masks)
                lengths = masks.float().sum(dim=-1)
                loss, loss_info = model.calculate_skew_loss(selector_logits, labels, masks, configs['skew_type'],
                                                            rationale_masks)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                induce_acc(selector_logits[:, 0, :].softmax(dim=-1)[:, 1], labels.view(-1))
                induce_pre(selector_logits[:, 0, :].softmax(dim=-1)[:, 1], labels.view(-1))
                induce_recall(selector_logits[:, 0, :].softmax(dim=-1)[:, 1], labels.view(-1))
                # According to FR: Since the accuracy increases rapidly in the first a few epochs,
                # obtaining a model that precisely achieves the pre-defined accuracy is almost impossible.
                # So we check the precision on the dev dataset every two batches to save it.
                if batch_idx % 2 == 0:
                    dev_acc = _get_induce_dev_accuracy(model, dev_dataloader, train_device)
                    acc = induce_acc.compute()
                    recall = induce_recall.compute()
                    pre = induce_pre.compute()
                    tbar.set_postfix_str(
                        ' '.join([f'{key}:{value:.6f}' for key, value in loss_info.items()]) +
                        f' acc:{acc:.6f} recall:{recall:.6f} pre:{pre:.6f}  dev_acc:{dev_acc:.6f} '
                    )

                    if dev_acc > save_points[0]:
                        induce_intensity = int(save_points[0] * 100)
                        print(f'saving biased-{induce_intensity} model')
                        torch.save(model.selector_cell.state_dict(),
                                   f'.{induce_intensity}.selector_cell.'.join(save_name.split('.')))
                        torch.save(model.selector_head.state_dict(),
                                   f'.{induce_intensity}.selector_head.'.join(save_name.split('.')))
                        save_points.pop(0)
                        if len(save_points) == 0:
                            print('finished!')
                            exit()

            induce_acc.reset()
            induce_pre.reset()
            induce_recall.reset()
    else:
        max_epochs = save_points[-1]
        for idx in range(max_epochs):
            # train
            tbar = tqdm(train_dataloader, total=len(train_dataloader))
            for batch_idx, batch in enumerate(tbar):
                model.train()
                batch = [item.to(device=train_device) for item in batch]
                input_ids, masks, labels = batch
                predictor_logits = model.induce_skew_predictor(input_ids, masks)
                lengths = masks.float().sum(dim=-1)
                loss, loss_info = model.calculate_skew_loss(predictor_logits, labels, masks, configs['skew_type'], None)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tbar.set_postfix_str(' '.join([f'{key}:{value:.6f}' for key, value in loss_info.items()]))

            if idx == save_points[0] - 1:
                induce_intensity = save_points[0]
                print(f'saving biased-{induce_intensity} model')
                torch.save(model.selector_cell.state_dict(),
                           f'.{induce_intensity}.predictor_cell.'.join(save_name.split('.')))
                torch.save(model.selector_head.state_dict(),
                           f'.{induce_intensity}.predictor_head.'.join(save_name.split('.')))
                save_points.pop(0)
                if len(save_points) == 0:
                    print('finished!')
                    exit()
