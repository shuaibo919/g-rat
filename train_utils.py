import json
import random

import numpy as np
import torch
from tqdm import tqdm

from preprocessing.vocabulary import GloveVocabulary
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from metrics import PerformanceEvaler, RationaleStatistic, RationaleEvaler
from preprocessing.preprocessing_beer import BeerDataset, BeerAnnotationDataset
from preprocessing.preprocessing_hotel import HotelDataset, HotelAnnotationDataset
# from preprocessing.preprocessing_movie import MovieDataset


def train(model, model_optimizer: torch.optim.Optimizer,
          dataloader: DataLoader, device, configs: dict):
    model.train()
    tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    train_performance = PerformanceEvaler(configs['output_size'], device=device, task_type=configs['task_type'])
    train_performance.to(device)
    for batch_idx, batch in tqdm_bar:
        batch = [item.to(device=device) for item in batch]
        input_ids, masks, labels = batch
        model_output = model.forward(input_ids, masks)
        # calculate loss
        loss, log_info = model.calculate_loss(model_output, labels, masks)
        train_performance.step(model_output.predictions, labels)
        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()
        tqdm_bar.set_postfix_str(' '.join([f'{key}:{value:.6f}' for key, value in log_info.items()]))

    if configs['task_type'] == 'regression':
        performance = train_performance.compute()
        print(f"train mse:{performance}")
    else:
        performance, task_pre, task_recall, task_f1 = train_performance.compute()
        print(
            f"train acc:{performance} train task_precision:{task_pre}"
            f"train task_recall:{task_recall} train task_f1:{task_f1}"
        )


def train_guide(model, model_optimizer: torch.optim.Optimizer,
                guider, guider_optimizer: torch.optim.Optimizer,
                dataloader: DataLoader, device, configs: dict, train_model=False):
    # prepare progress bar
    tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    # prepare metric calculator
    train_performance = PerformanceEvaler(configs['output_size'], device=device, task_type=configs['task_type'])
    train_performance.to(device)
    for batch_idx, batch in tqdm_bar:
        model.train()
        guider.train()
        # prepare data
        batch = [item.to(device=device) for item in batch]
        input_ids, masks, labels = batch
        # guider forward
        guider_optimizer.zero_grad()
        guider_output = guider.forward(input_ids, masks)
        # update guider
        guider_loss, log_info = guider.calculate_loss(guider_output, labels)
        guider_loss.backward()
        guider_optimizer.step()
        guider.eval()
        # calculate model loss
        if train_model:
            model_output = model.forward(input_ids, masks)
            loss, loss_info = model.calculate_loss(model_output, labels, masks, guider, input_ids)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            # update log info
            log_info.update(loss_info)
            train_performance.step(model_output.predictions, labels)
        # show log info
        if batch_idx % 20 == 0:
            tqdm_bar.set_postfix_str(' '.join([f'{key}:{value:.6f}' for key, value in log_info.items()]))

    # print metrics
    if train_model:
        if configs['task_type'] == 'regression':
            performance = train_performance.compute()
            print(f"train mse:{performance}")
        else:
            performance, task_pre, task_recall, task_f1 = train_performance.compute()
            print(
                f"train acc:{performance} train task_precision:{task_pre}"
                f"train task_recall:{task_recall} train task_f1:{task_f1}"
            )


def valid(model, dataloader: DataLoader, device, configs: dict):
    dev_performance = PerformanceEvaler(configs['output_size'], device=device, task_type=configs['task_type'])
    rationale_stat = RationaleStatistic()
    dev_performance.to(device)
    with torch.no_grad():
        model.eval()
        tbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, batch in tbar:
            batch = [item.to(device=device) for item in batch]
            input_ids, masks, labels = batch
            model_output = model.forward(input_ids, masks)
            # calculate loss
            loss, log_info = model.calculate_loss(model_output, labels, masks)
            rationale_stat.step(model_output.rationales, masks)
            dev_performance.step(model_output.predictions, labels)
            if batch_idx % 10 == 0:
                tbar.set_postfix_str(' '.join([f'{key}:{value:.5f}' for key, value in log_info.items()]))

    if configs['task_type'] == 'regression':
        performance = dev_performance.compute()
        print(f"dev mse:{performance}")
    else:
        performance, task_pre, task_recall, task_f1 = dev_performance.compute()
        print(
            f"dev acc:{performance} dev task_precision:{task_pre} "
            f"dev task_recall:{task_recall} dev task_f1:{task_f1} "
        )

    dev_sparsity_rate = rationale_stat.compute()
    print(f"history:{configs['history_performance']}")
    print(f'dev selection:{dev_sparsity_rate}')
    return performance, dev_sparsity_rate


def valid_guide(model, guider, dataloader: DataLoader, device, configs: dict):
    dev_performance = PerformanceEvaler(configs['output_size'], device=device, task_type=configs['task_type'])
    rationale_stat = RationaleStatistic()
    dev_performance.to(device)
    with torch.no_grad():
        model.eval()
        guider.eval()
        tqdm_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, batch in tqdm_bar:
            batch = [item.to(device=device) for item in batch]
            input_ids, masks, labels = batch
            model_output = model.forward(input_ids, masks)
            # calculate loss
            loss, log_info = model.calculate_loss(model_output, labels, masks, guider, input_ids)
            rationale_stat.step(model_output.rationales, masks)
            dev_performance.step(model_output.predictions, labels)
            if batch_idx % 10 == 0:
                tqdm_bar.set_postfix_str(' '.join([f'{key}:{value:.5f}' for key, value in log_info.items()]))

    if configs['task_type'] == 'regression':
        performance = dev_performance.compute()
        print(f"dev mse:{performance}")
    else:
        performance, task_pre, task_recall, task_f1 = dev_performance.compute()
        print(
            f"dev acc:{performance} dev task_precision:{task_pre} "
            f"dev task_recall:{task_recall} dev task_f1:{task_f1} "
        )

    dev_sparsity_rate = rationale_stat.compute()
    print(f"history:{configs['history_performance']}")
    print(f'dev selection:{dev_sparsity_rate}')
    return performance, dev_sparsity_rate


def run_on_annotations(model, dataloader: DataLoader, vocab, device, configs: dict, save_name: str, llm=False):
    # token_acc,token_recall,token_precision,token_f1
    rationale_metric = RationaleEvaler(configs['output_size'], device=device, task_type=configs['task_type'])

    # sparsity statistic
    rationale_stat = RationaleStatistic()

    # saving rationale
    rationale_dump_list = []

    # testing
    with torch.no_grad():
        model.eval()
        tbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, batch in tbar:
            batch = [item.to(device=device) for item in batch]
            input_ids, masks, labels, human_rationale = batch
            model_output = model(input_ids, masks)
            rationale_metric.step(model_output.predictions, labels, model_output.rationales, human_rationale)
            rationale_stat.step(model_output.rationales, masks.to(dtype=torch.long))
            if llm:
                model_rationale_list = vocab.batch_decode(
                    mask_non_rationale(input_ids, vocab, model_output.rationales), skip_special_token=True
                )
                annotation_list = vocab.batch_decode(
                    mask_non_rationale(input_ids, vocab, human_rationale), skip_special_token=True
                )
            else:
                model_rationale_list = vocab.batch_decode(input_ids, model_output.rationales)
                annotation_list = vocab.batch_decode(input_ids, human_rationale)
            prediction_list = model_output.predictions.cpu().numpy().tolist()
            label_list = labels.cpu().numpy().tolist()
            # save results
            for four_var in zip(model_rationale_list, annotation_list, prediction_list, label_list):
                rationale_dump_list.append(
                    {
                        'model_answer': four_var[0],
                        'annotation': four_var[1],
                        'model_prediction': four_var[2],
                        'true_label': four_var[3],
                    }
                )

        print('Results:')
        print(f'{rationale_metric.compute()}')
        print(f'model selection rate:{rationale_stat.compute()}')
        if configs['print_rationale']:
            for print_idx in range(configs['print_rationale_nums']):
                print(f'example {print_idx + 1}:')
                print(
                    f"  model_prediction:{' '.join(rationale_dump_list[print_idx]['model_prediction'])}\n"
                    f"  model_answer:{' '.join(rationale_dump_list[print_idx]['model_answer'])}\n"
                    f"  true_label:{rationale_dump_list[print_idx]['true_label']}\n"
                    f"  annotation:{rationale_dump_list[print_idx]['annotation']}\n"
                )

        output_path = save_name.split('.')[0] + '.json'
        with open(output_path, 'w+') as fp:
            for dump_line in rationale_dump_list:
                json.dump(dump_line, fp)
                fp.write('\r\n')


def prepare_dataset(configs: dict):
    """
    loading the dataset and embeddings
    return vocab,train_dataloader,dev_dataloader,test_dataset
    """
    vocab = GloveVocabulary(configs['embeddings'], configs['embedding_size'])
    convert = (configs['task_type'] == 'classification')
    if configs['dataset'] == 'beer':
        train_dataset = BeerDataset(configs['train_path'], vocab, configs["aspect"], configs["max_length"],
                                    convert_label_to_binary_sentiment=convert, balanced=convert)
        dev_dataset = BeerDataset(configs['dev_path'], vocab, configs["aspect"], configs["max_length"],
                                  convert_label_to_binary_sentiment=convert)
        # test_dataset = BeerAnnotationDataset(configs['test_path'], vocab, configs["aspect"],True,convert)
        # keep same setting with FR.
        test_dataset = BeerAnnotationDataset(configs['test_path'], vocab, configs["aspect"],
                                             True, convert, configs["max_length"])
    elif configs['dataset'] == 'hotel':
        train_dataset = HotelDataset(configs['train_path'], vocab, configs["aspect"], configs["max_length"],
                                     balanced=True)
        dev_dataset = HotelDataset(configs['dev_path'], vocab, configs["aspect"], configs["max_length"])
        test_dataset = HotelAnnotationDataset(configs['test_path'], vocab, configs["aspect"])
    elif configs['dataset'] == 'movie':
        raise DeprecationWarning
        # train_dataset = MovieDataset(configs['train_path'], vocab, 'train', configs["max_length"])
        # dev_dataset = MovieDataset(configs['dev_path'], vocab, 'val', configs["max_length"])
        # test_dataset = MovieDataset(configs['test_path'], vocab, 'test')
    else:
        raise NotImplementedError

    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset),
        collate_fn=train_dataset.collate_fn, batch_size=configs['batch_size'],
    )

    dev_dataloader = DataLoader(
        dev_dataset, sampler=SequentialSampler(dev_dataset),
        collate_fn=dev_dataset.collate_fn, batch_size=configs['batch_size']
    )

    test_dataloader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset),
        collate_fn=test_dataset.collate_fn, batch_size=64
    )

    return vocab, train_dataloader, dev_dataloader, test_dataloader


def mask_non_rationale(input_ids, tokenizer, mask):
    mask = mask.long()
    pad_id = tokenizer.pad_token_id
    input_ids = input_ids * mask + input_ids * (1 - mask) * pad_id
    return input_ids


def show_config(configs: dict, show=True):
    if show:
        for k, v in configs.items():
            print(f'{k} : {v}')
    return configs


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
