import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset
# from transformers import PreTrainedTokenizer
from preprocessing.vocabulary import GloveVocabulary


class BeerDataset(Dataset):
    def __init__(self, path: str, vocabulary: GloveVocabulary, aspect: int = 0,
                 max_length: int = 256, skip_long_sentence: bool = False,
                 sort_in_mini_batch: bool = True,
                 convert_label_to_binary_sentiment: bool = False,
                 balanced: bool = False,
                 only_load_first_sentence: bool = False
                 ):
        """
        Loading Beer Advocate data set, refer to the implementation:
        "Interpretable Neural Predictions with Differentiable Binary Variables"
        (https://aclanthology.org/P19-1284/)
        :param path: dataset path
        :param vocabulary: vocabulary of pretrained embedding
        :param aspect: specify an aspect in beer reviews (0-Appearance,1-Smell,2-Palate)
        :param max_length: pad to a length specified by the max_length
        :param skip_long_sentence: whether to skip long sentences(>max_length), the default is truncation.
        :param sort_in_mini_batch: whether to sort by sentence length in mini_batch
        :param convert_label_to_binary_sentiment: whether to convert scores into binary label
        :param only_load_first_sentence: whether to load only the first sentence, used for Skew experiment
        """
        self.convert_label_to_binary_sentiment = convert_label_to_binary_sentiment
        self.path = path
        self.aspect = aspect
        self.sort_in_mini_batch = sort_in_mini_batch
        self.vocabulary = vocabulary
        # load data
        self.reviews = []
        self.scores = []
        self.data_max = 0
        if self.convert_label_to_binary_sentiment:
            self.positive_indices = []
            self.negative_indices = []
        with open(self.path, 'rt', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                scores = list(map(float, parts[:5]))
                # load a review
                if len(parts[5:]) > max_length:
                    if skip_long_sentence:
                        continue
                    else:
                        temp_review_string = parts[5:5 + max_length]
                else:
                    temp_review_string = parts[5:]

                if only_load_first_sentence is True:
                    # Take the first sentence
                    temp_review_string = ' '.join(temp_review_string).split('.')[0].split(' ')

                self.data_max = max(self.data_max, len(temp_review_string) - 5)
                # load a score
                if self.aspect > -1:
                    score = scores[self.aspect]
                    if convert_label_to_binary_sentiment:
                        if score <= 0.4:
                            score = 0
                            self.negative_indices.append(len(self.reviews))
                        elif score >= 0.6:
                            score = 1
                            self.positive_indices.append(len(self.reviews))
                        else:
                            continue

                self.reviews.append(temp_review_string)
                self.scores.append(score)

        print(f'\nBeerDataset aspect{aspect} load to cpu memory finished.\n'
              f'BeerDataset.data_max:{self.data_max}\n'
              f'BeerDataset.len:{len(self.reviews)}')

        if balanced and convert_label_to_binary_sentiment:
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
                  f'BeerDataset.len:{len(self.reviews)}')

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
        if self.convert_label_to_binary_sentiment:
            labels = np.array(labels)
        else:
            labels = np.array(labels, dtype=np.float32)

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


class BeerAnnotationDataset(Dataset):
    def __init__(self, path: str, vocabulary, aspect: int = -1, sort_in_mini_batch: bool = True,
                 convert_label_to_binary_sentiment: bool = False, max_length: int = None):
        """
        Loading Beer Advocate dataset with human-annotations, refer to the implementation:
        "Interpretable Neural Predictions with Differentiable Binary Variables"
        (https://aclanthology.org/P19-1284/)
        :param path: dataset path
        :param vocabulary: vocabulary of pretrained embedding
        :param aspect: specify an aspect in beer reviews (0-Appearance,1-Smell,2-Palate)
        :param max_length: maximum sentence length, which is not limited by default.
        :param sort_in_mini_batch: whether to sort by sentence length in mini_batch
        :param convert_label_to_binary_sentiment: whether to convert scores into binary label
        """
        self.vocabulary = vocabulary
        self.path = path
        self.aspect = aspect
        # load data
        self.reviews = []
        self.scores = []
        self.rationales = []
        self.convert_label_to_binary_sentiment = convert_label_to_binary_sentiment
        self.sort_in_mini_batch = sort_in_mini_batch
        self.max_length = max_length
        if convert_label_to_binary_sentiment:
            self.positive_nums = 0
            self.negative_nums = 0
        with open(self.path, 'rt', encoding='utf-8') as f:
            for line in f:
                data_description = json.loads(line)
                review = data_description['x']

                if max_length is not None:
                    if len(review) > max_length:
                        review = review[:max_length]

                scores = list(map(float, data_description['y']))
                annotations = data_description[f'{aspect}']
                if len(annotations) == 0:
                    continue

                if self.aspect > -1:
                    score = scores[self.aspect]
                    if convert_label_to_binary_sentiment:
                        if score <= 0.4:
                            score = 0
                            self.negative_nums += 1
                        elif score >= 0.6:
                            score = 1
                            self.positive_nums += 1
                        else:
                            continue

                self.reviews.append(review)
                self.scores.append(score)
                self.rationales.append(data_description[f'{aspect}'])
        print(f'\nBeerAnnotationDataset aspect{aspect} load finished.\n'
              f'BeerAnnotationDataset.len:{len(self.reviews)}')
        if convert_label_to_binary_sentiment:
            print(f'\n positive aspect{aspect} instance{self.positive_nums}.\n'
                  f'\n negative aspect{aspect} instance{self.negative_nums}.\n')

    def __getitem__(self, item):
        # input_ids, mask = self.vocabulary.encode(self.reviews[item], max_length=len(self.reviews[item]))
        # input_ids = torch.tensor(input_ids)
        # mask = torch.tensor(mask, dtype=torch.bool)
        # label = torch.tensor([self.scores[item]])
        return self.reviews[item], self.scores[item], self.rationales[item]

    def collate_fn(self, batch):
        """
        this function is for torch.utils.Dataloader.
        """
        lengths = np.array([len(item[0]) for item in batch])
        max_length = lengths.max()
        input_ids, labels = [], []
        rationales = torch.zeros((len(batch), max_length.item()), dtype=torch.long)
        for idx, item in enumerate(batch):
            input_ids.append(self.vocabulary.encode(item[0], max_length.item(), return_mask=False))
            labels.append([item[1]])
            for rationale_interval in item[2]:
                if max_length is not None:
                    if rationale_interval[0] >= max_length:
                        continue
                rationales[idx, rationale_interval[0]:rationale_interval[1]] = 1

        input_ids = np.array(input_ids)
        rationales = np.array(rationales)
        if self.convert_label_to_binary_sentiment:
            labels = np.array(labels)
        else:
            labels = np.array(labels, dtype=np.float32)

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

    def get_rationale_average_ratio(self):
        """
        :return:  the average proportion of all rationales in corresponding sentences.
        """
        rationale_lens = []
        for idx, rationale_intervals in enumerate(self.rationales):
            dist = 0.
            for interval in rationale_intervals:
                dist += (interval[1] - interval[0])
            print(f'{idx}: dist is {dist},total len is {len(self.reviews[idx].split())}')
            rationale_lens.append(dist / len(self.reviews[idx].split()))
        return np.mean(rationale_lens)

# class BeerDatasetForLLM(BeerDataset):
#     """
#     encode_sent['input_ids'].squeeze(), encode_sent['attention_mask'].squeeze(), label
#     """
#     def __init__(self, path: str, aspect: int = -1, max_length=100,
#                  tokenizer: PreTrainedTokenizer = None, skip_long_sentence=False,
#                  convert_label_to_binary_sentiment: bool = False,
#                  balanced:bool = False):
#         super(BeerDatasetForLLM,self).__init__(
#             path, None, aspect, max_length, skip_long_sentence, False,convert_label_to_binary_sentiment,balanced
#         )
#         self.tokenizer = tokenizer
#
#     def __getitem__(self, item):
#         return self.reviews[item], self.scores[item]
#
#     def collate_fn(self, batch):
#         """
#         this function is for torch.utils.Dataloader.
#         """
#         lengths = np.array([len(item[0]) for item in batch])
#         tokenize_max_length = lengths.max().item() * 3 // 2
#         input_ids, labels, masks = [], [], []
#         for item in batch:
#             encode_sent = self.tokenizer(item[0], add_special_tokens=True,
#                                      max_length=tokenize_max_length,
#                                      truncation=True,
#                                      padding='max_length',
#                                      return_length=True,
#                                      is_split_into_words=True,
#                                      return_tensors='pt')
#             input_ids.append(encode_sent['input_ids'].squeeze())
#             masks.append(encode_sent['attention_mask'].squeeze())
#             labels.append([item[1]])
#
#         input_ids = torch.stack(input_ids,dim=0)
#         masks = torch.stack(masks, dim=0)
#         if self.convert_label_to_binary_sentiment:
#             labels = torch.tensor(labels)
#         else:
#             labels = torch.tensor(labels, dtype=torch.float)
#
#         return input_ids, masks, labels
#     def __len__(self):
#         return len(self.reviews)
#
#
# class BeerAnnotationDatasetForLLM(BeerAnnotationDataset):
#     def __init__(self, path: str, aspect: int = -1, tokenizer: PreTrainedTokenizer = None,
#                  convert_label_to_binary_sentiment:bool=True):
#         super(BeerAnnotationDatasetForLLM,self).__init__(
#             path, None, aspect, False, convert_label_to_binary_sentiment
#         )
#         self.tokenizer = tokenizer
#
#     def __getitem__(self, item):
#         return self.reviews[item],self.scores[item],self.rationales[item]
#
#     def collate_fn(self, batch):
#         """
#         this function is for torch.utils.Dataloader.
#         """
#         lengths = np.array([len(item[0]) for item in batch])
#         tokenize_max_length = lengths.max().item() * 3 // 2
#         input_ids, labels, masks,rationales = [], [], [], []
#         for item in batch:
#             encode_sent = self.tokenizer(
#                 item[0], add_special_tokens=True,
#                 max_length=tokenize_max_length,
#                 truncation=True, padding='max_length',
#                 return_length=True, is_split_into_words=True,
#                 return_offsets_mapping=True, return_tensors='pt'
#             )
#             input_ids.append(encode_sent['input_ids'].squeeze())
#             masks.append(encode_sent['attention_mask'].squeeze())
#             rationales.append(
#                 self.align_rationale_with_tokenized(tokenize_max_length,item[2],encode_sent['offset_mapping']).squeeze()
#             )
#             labels.append([item[1]])
#
#         input_ids = torch.stack(input_ids,dim=0)
#         masks = torch.stack(masks, dim=0)
#         rationales = torch.stack(rationales, dim=0)
#         if self.convert_label_to_binary_sentiment:
#             labels = torch.tensor(labels)
#         else:
#             labels = torch.tensor(labels, dtype=torch.float)
#
#         return input_ids, masks, labels, rationales
#
#     def __len__(self):
#         return len(self.reviews)
#
#     @staticmethod
#     def align_rationale_with_tokenized(sentence_size,human_rationales: list, offsets_mapping: torch.Tensor):
#         rationale = torch.zeros((offsets_mapping.size(0),sentence_size), dtype=torch.long)
#         for rationale_interval in human_rationales:
#             old_start,old_end = rationale_interval[0] + 1,rationale_interval[1] + 1
#             for list_idx,offset_mapping in enumerate(offsets_mapping):
#                 loop_flag, new_start, new_end = 0, 0, 0
#                 for idx,offset_char_start in enumerate(offset_mapping[:,0]):
#                     if loop_flag == old_start:
#                         new_start = idx
#                     if loop_flag > old_end:
#                         new_end = idx - 1
#                         break
#                     if offset_char_start.item() == 0 :
#                         loop_flag +=1
#                 rationale[list_idx,new_start:new_end] = 1
#         return rationale

# simple test
# vocab = GloveVocabulary('../data/glove.6B.100d.txt', 100)
# bd = BeerDataset('../data/beer/reviews.aspect2.train.txt',vocab,aspect=2,only_load_first_sentence=True)
# print(bd[0])
# print(bd[1])
# print(bd[2])
# print(bd[3])
# (['coppery', 'brown', ',', 'with', 'suspended', 'sediment', 'throughout', ''], 0.6)
# (['all', 'im', 'going', 'to', 'say', 'is', 'that', 'this', 'is', 'one', 'of', 'the', 'most', 'enjoyable', 'beers', 'i', "'ve", 'ever', 'had', ''], 0.6)
# (['pours', 'a', 'very', 'hazy', 'blonde', 'body', 'with', 'a', 'great', 'foamy', 'white', 'head', 'that', 'dies', 'down', 'with', 'time', 'but', 'a', 'good', 'amout', 'is', 'retained', 'for', 'much', 'of', 'the', 'drink', ''], 0.4)
# (['this', 'is', 'a', '12oz', ''], 0.5)
