import csv
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.vocabulary import GloveVocabulary


class HotelDataset(Dataset):
    def __init__(self, path: str, vocabulary: GloveVocabulary, aspect: int = -1,
                 max_length: int = 256, skip_long_sentence: bool = False,
                 sort_in_mini_batch: bool = False,
                 balanced: bool = False):
        """
        Loading Hotel Review dataset:
        :param path: dataset path
        :param vocabulary: vocabulary of pretrained embedding
        :param aspect: specify an aspect in beer reviews (0-Appearance,1-Smell,2-Palate)
        :param max_length: pad to a length specified by the max_length
        :param skip_long_sentence: whether to skip long sentences(>max_length), the default is truncation.
        :param sort_in_mini_batch: whether to sort by sentence length in mini_batch
        """
        self.num_to_aspect = {0: 'Location', 1: 'Service', 2: 'Cleanliness'}
        self.path = path
        self.aspect = aspect
        self.sort_in_mini_batch = sort_in_mini_batch
        self.vocabulary = vocabulary
        # load data
        self.reviews = []
        self.scores = []
        self.positive_indices = []
        self.negative_indices = []

        with open(self.path, "rt") as f:
            reader = csv.reader(f, delimiter="\t")
            for idx, line in enumerate(reader):
                if idx == 0:
                    continue
                review_text = line[2].split(' ')
                if len(review_text) > max_length:
                    if skip_long_sentence:
                        continue
                    else:
                        review_text = review_text[:max_length]
                self.scores.append(int(line[1]))
                if self.scores[-1] == 0:
                    self.negative_indices.append(len(self.reviews))
                else:
                    self.positive_indices.append(len(self.reviews))
                self.reviews.append(review_text)

        print(f'\nHotelDataset aspect{aspect} load to cpu memory finished.\n'
              f'HotelDataset.len:{len(self.reviews)}')
        print(f'\n positive aspect{aspect} instance{len(self.positive_indices)}.\n'
              f'\n negative aspect{aspect} instance{len(self.negative_indices)}.\n')

        if balanced:
            # random.seed(20230815)
            # print(f'Make the Training dataset class balanced. The Sample seed is 20230815(paper deadline)!')
            min_examples = min(len(self.positive_indices), len(self.negative_indices))
            if len(self.positive_indices) > min_examples:
                samples_to_pop = random.sample(self.positive_indices, len(self.positive_indices) - min_examples)
                print(f'Drop {len(samples_to_pop)} positive reviews!')
            else:
                samples_to_pop = random.sample(self.negative_indices, len(self.negative_indices) - min_examples)
                print(f'Drop {len(samples_to_pop)} negative reviews!')

            samples_to_pop = sorted(samples_to_pop, reverse=True)
            for sample_idx in samples_to_pop:
                self.reviews.pop(sample_idx)
                self.scores.pop(sample_idx)
            print(f'balance finished.\n'
                  f'HotelDataset.len:{len(self.reviews)}')

    def __getitem__(self, item):
        return self.reviews[item], self.scores[item]

    def collate_fn(self, batch):
        """
        this function is for torch.utils.Dataloader.
        """
        lengths = np.array([len(item[0]) for item in batch])
        max_length = lengths.max()
        input_ids, labels = [], []
        for item in batch:
            input_ids.append(self.vocabulary.encode(item[0], max_length.item(), return_mask=False))
            labels.append([item[1]])

        input_ids = np.array(input_ids)
        labels = np.array(labels)

        if self.sort_in_mini_batch:  # required for LSTM
            sort_idx = np.argsort(lengths)[::-1]
            input_ids = input_ids[sort_idx]
            labels = labels[sort_idx]

        # input_ids, mask, labels
        return torch.from_numpy(input_ids), \
            torch.from_numpy(input_ids != self.vocabulary.pad_token_id), \
            torch.from_numpy(labels)

    def __len__(self):
        return len(self.reviews)


class HotelAnnotationDataset(Dataset):
    def __init__(self, path: str, vocabulary, aspect: int = -1, sort_in_mini_batch: bool = False):
        """
        Loading Hotel Review dataset with human-annotations
        """
        self.vocabulary = vocabulary
        self.path = path
        self.aspect = aspect
        # load data
        self.reviews = []
        self.scores = []
        self.rationales = []
        self.sort_in_mini_batch = sort_in_mini_batch
        self.positive_nums = 0
        self.negative_nums = 0

        with open(self.path, "rt") as f:
            reader = csv.reader(f, delimiter="\t")
            lines = []
            for idx, line in enumerate(reader):
                if idx == 0:
                    continue
                lines.append(line)
                review_text = line[2].split(' ')
                self.scores.append(int(line[1]))
                if self.scores[-1] == 0:
                    self.negative_nums += 1
                else:
                    self.positive_nums += 1
                self.reviews.append(review_text)
                self.rationales.append([int(x) for x in line[3].split(' ')])

        print(f'\nHotelAnnotationDataset aspect{aspect} load finished.\n'
              f'HotelAnnotationDataset.len:{len(self.reviews)}')
        print(f'\n positive aspect{aspect} instance{self.positive_nums}.\n'
              f'\n negative aspect{aspect} instance{self.negative_nums}.\n')

    def __getitem__(self, item):
        return self.reviews[item], self.scores[item], self.rationales[item]

    @staticmethod
    def _pad_list(input_list, pad_lens):
        max_len = pad_lens
        output_list = []
        # iterate over the input list
        for sublist in input_list:
            # copy the sublist
            padded_sublist = sublist[:]
            # append zeros to the end of the sublist until it reaches the maximum length
            while len(padded_sublist) < max_len:
                padded_sublist.append(0)
            # add the padded sublist to the output list
            output_list.append(padded_sublist)
        # return the output list
        return output_list

    def collate_fn(self, batch):
        """
        this function is for torch.utils.Dataloader.
        """
        lengths = np.array([len(item[0]) for item in batch])
        max_length = lengths.max()
        input_ids, labels, rationales = [], [], []
        for idx, item in enumerate(batch):
            input_ids.append(self.vocabulary.encode(item[0], max_length.item(), return_mask=False))
            labels.append([item[1]])
            rationales.append(item[2])

        rationales = self._pad_list(rationales, max_length.item())
        input_ids = np.array(input_ids)
        rationales = np.array(rationales)
        labels = np.array(labels)

        if self.sort_in_mini_batch:  # required for LSTM
            sort_idx = np.argsort(lengths)[::-1]
            input_ids = input_ids[sort_idx]
            labels = labels[sort_idx]
            rationales = rationales[sort_idx]

        # input_ids, mask, labels, rationales
        return torch.from_numpy(input_ids), \
            torch.from_numpy(input_ids != self.vocabulary.pad_token_id), \
            torch.from_numpy(labels), torch.from_numpy(rationales)

    def __len__(self):
        return len(self.reviews)




