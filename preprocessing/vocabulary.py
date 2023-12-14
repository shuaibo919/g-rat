import gzip
import torch
import numpy as np


class GloveVocabulary:
    def __init__(self, embedding_file_path: str, embedding_dim: int = 100,
                 unk_token: str = '<unk>',
                 pad_token: str = '<pad>'):
        """
        Load the vocabulary from word vectors
        :param embedding_file_path: Pre-trained word vector file path
        :param embedding_dim: default to 200
        :param unk_token: unk
        :param pad_token: pad
        """
        vectors = []
        self.w2i = {}
        self.i2w = []
        self._embedding_matrix = None

        self.unk_token_id = 0
        self.pad_token_id = 1
        # Random embedding vector for unknown words
        vectors.append(np.random.uniform(-0.05, 0.05, embedding_dim).astype(np.float32))
        self.w2i[unk_token] = self.unk_token_id
        self.i2w.append(unk_token)

        # Zero vector for padding
        vectors.append(np.zeros(embedding_dim).astype(np.float32))
        self.w2i[pad_token] = self.pad_token_id
        self.i2w.append(pad_token)

        if embedding_file_path.endswith('.gz'):
            file_source = gzip.open(embedding_file_path, 'rt', encoding='utf-8')
        elif embedding_file_path.endswith('.txt'):
            file_source = open(embedding_file_path, 'rt', encoding='utf-8')
        else:
            raise NotImplementedError

        with file_source as f:
            for line in f:
                word, vec = line.split(u' ', 1)
                self.w2i[word] = len(vectors)
                self.i2w.append(word)
                v = np.array(vec.split(), dtype=np.float32)
                assert len(v) == embedding_dim, "dim mismatch"
                vectors.append(v)

        self._embedding_matrix = np.stack(vectors)

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def embedding_matrix(self):
        return self._embedding_matrix

    def encode(self, input_text: list, max_length: int = None, return_mask=True):
        """
        :return: output_id or output_id, mask
        """
        if max_length is None:
            output_id = [self.w2i.get(token, self.unk_token_id) for token in input_text]
        else:
            output_id = [self.w2i.get(token, self.unk_token_id) for token in input_text] + \
                        [self.pad_token_id] * (max_length - len(input_text))
            mask = [1] * len(input_text) + [0] * (max_length - len(input_text))
            if return_mask:
                return output_id, mask
        return output_id

    def batch_encode(self, input_texts: str, max_length: int = None):
        raise NotImplementedError

    def batch_decode(self, input_ids: torch.Tensor, sequence_mask: torch.Tensor, skip_special_token: bool = False):
        input_ids = input_ids * sequence_mask + self.pad_token_id * (1 - sequence_mask)
        input_ids = input_ids.cpu().numpy().tolist()
        outputs = []
        for i in range(len(input_ids)):
            output = []
            for j in range(len(input_ids[i])):
                token_id = int(input_ids[i][j])
                if skip_special_token and (token_id == 0 or token_id == 1):
                    continue
                output.append(self.i2w[token_id])
            outputs.append(output)
        return outputs
