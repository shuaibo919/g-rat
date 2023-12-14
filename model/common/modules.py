import torch
from typing import Literal, Any, Optional


class RNNModule(torch.nn.Module):

    def __init__(self, cell_type, embedding_dim, hidden_dim, num_layers):
        super(RNNModule, self).__init__()
        if cell_type == 'lstm':
            self.rnn = torch.nn.LSTM(input_size=embedding_dim,
                                     hidden_size=hidden_dim // 2,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     bidirectional=True)
        elif cell_type == 'gru':
            self.rnn = torch.nn.GRU(input_size=embedding_dim,
                                    hidden_size=hidden_dim // 2,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    bidirectional=True)
        else:
            raise NotImplementedError(f'cell_type {cell_type} is not implemented')

    def forward(self, x):
        """
        :param x: Inputs(batch_size, seq_length, input_dim)
        :return: h: bidirectional(batch_size, seq_length, hidden_dim)
        """
        h = self.rnn(x)
        return h


class GuidanceAttention(torch.nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim, bias=bias)

    def shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        inverted_mask = 1.0 - expanded_mask

        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

    def forward(
            self,
            hidden_states: torch.Tensor,
            mask: torch.Tensor,
    ):
        """Input shape: Batch x Time x Channel"""
        mask = self.expand_mask(mask, hidden_states.dtype)

        bsz, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states) * self.scaling
        key_states = self.shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self.shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self.shape(query_states, seq_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, seq_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, seq_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if mask is not None:
            if mask.size() != (bsz, 1, seq_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, seq_len, src_len)}, but is {mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, seq_len, src_len) + mask
            attn_weights = attn_weights.view(bsz * self.num_heads, seq_len, src_len)

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, seq_len, src_len)
        attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, seq_len, src_len)
        attn_probs = torch.nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, seq_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, seq_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class GuidanceModule(torch.nn.Module):
    """
    RNN - Weighted Score - MLP.
    """

    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, num_classes: int, num_layers: int = 1,
                 dropout: float = 0.2, cell_type: Literal['rcnn', 'gru'] = 'gru', noise_sigma: float = 1.0,
                 pretrained_embedding=None, fix_embedding=True):
        super(GuidanceModule, self).__init__()
        self.noise_sigma = noise_sigma
        self.embedding_layer = torch.nn.Embedding(vocab_size, emb_size, padding_idx=1)
        if pretrained_embedding is not None:
            self.embedding_layer.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding_layer.weight.requires_grad = not fix_embedding
        self.activation = torch.nn.GELU()
        self.convert_layer = torch.nn.Linear(emb_size, hidden_size, bias=True)
        self.in_cell = RNNModule(cell_type, emb_size, hidden_size, num_layers)
        self.att_fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1),
        )
        self.prj_fc = torch.nn.Linear(hidden_size, hidden_size)
        self.pooling = AttentionPooling(hidden_size, hidden_size)
        self.out_head = torch.nn.Linear(hidden_size, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, masks: torch.Tensor, return_final_states: bool = False):
        """
        :param input_ids: (batch, seq_len) — Indices of input sequence tokens in the vocabulary.
        :param masks: (batch, seq_len) — Mask to avoid performing calculation on padding token indices.
        :param return_final_states: whether to return the hidden states of the final layer.
        :return: attn_weights, final_logits, final_states(if return)
        """
        extend_mask = masks.unsqueeze(-1)
        extend_float_mask = extend_mask.float()
        # in cell
        embedding = extend_float_mask * self.embedding_layer(input_ids)
        input_states = self.convert_layer(embedding)
        hidden_states, _ = self.in_cell(embedding)
        # add & norm
        hidden_states = self.layer_norm(hidden_states + input_states)
        attn_weights = self.att_fc(self.dropout(hidden_states))
        attn_weights = attn_weights.masked_fill_(~extend_mask, torch.finfo(torch.float).min)
        # add noise
        if self.training:
            attn_noises = torch.normal(0, self.noise_sigma, attn_weights.size(), device=attn_weights.device,
                                       dtype=attn_weights.dtype)
            attn_noises = torch.abs(attn_noises).masked_fill_(~extend_mask, torch.finfo(torch.float).min)
            attn_weights = attn_weights + attn_noises
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=1)
        hidden_states = self.activation(self.prj_fc(attn_weights * hidden_states))
        hidden_states = self.layer_norm(hidden_states)
        final_states = self.pooling.forward(hidden_states, masks)
        final_logits = self.out_head(self.dropout(final_states))

        if return_final_states:
            return attn_weights, final_logits, final_states

        return attn_weights, final_logits

    @staticmethod
    def get_expand_token_scores(attn_weights: torch.tensor, mask: torch.tensor, sparsity: float):
        soft_weights = attn_weights[:, :, 0].detach()
        scaling = torch.mean(soft_weights, dim=-1) + (1. / (1. + mask.float().sum(dim=-1)))
        scaling_soft_weights = soft_weights / (1e-8 + scaling.unsqueeze(dim=1))
        scaling_soft_probs = torch.clamp_max(scaling_soft_weights, 1.0) * mask.float()
        return scaling_soft_probs

    @staticmethod
    def calculate_loss(model_output: tuple, labels: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        :param model_output: forward result
        :param labels: [batch,1]
        :return: loss, loss_info
        """
        soft_weights, final_logits = model_output[0], model_output[1]
        loss = torch.nn.functional.cross_entropy(final_logits, labels.squeeze())
        loss_info = {
            'guider': loss.item(),
        }

        return loss, loss_info


class AttentionPooling(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_size: int):
        """
        AttentionPoolingBlock
        """
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_size),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1)
        )
        self._weights = None

    @property
    def attn_weights(self):
        return self._weights

    def forward(self, hidden_states: torch.Tensor, masks: torch.Tensor = None):
        """
        :param hidden_states: [batch,seq_len,dim]
        :param masks: bool tensor [batch,seq_len]
        :return: converge_representations [batch,dim]
        """
        weight = self.attention(hidden_states).squeeze()
        if masks is not None:
            masks = masks.bool()
            weight.masked_fill_(~masks, torch.finfo(torch.float).min)
        weight = torch.nn.functional.softmax(weight, dim=-1)
        self._weights = weight
        weight = weight.unsqueeze(dim=1)
        converge_representations = torch.bmm(weight, hidden_states).squeeze()
        return converge_representations


class JensenShannonDivergence:

    def __init__(self):
        super(JensenShannonDivergence, self).__init__()

    def __call__(self, net_1_logits, net_2_logits):
        net_1_probs = torch.nn.functional.softmax(net_1_logits, dim=1)
        net_2_probs = torch.nn.functional.softmax(net_2_logits, dim=1)

        total_m = 0.5 * (net_1_probs + net_2_probs)
        loss = 0.0
        loss += torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(net_1_logits, dim=1), total_m, reduction="batchmean"
        )
        loss += torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(net_2_logits, dim=1), total_m, reduction="batchmean"
        )

        return 0.5 * loss


class FactorAnnealer:
    def __init__(self, factor: float, decay_callback):
        self.factor = factor
        self.decay_callback = decay_callback
        self.current_step = 0
        self._current_factor = factor

    def step(self):
        decay_factor = self.decay_callback(self.current_step)
        self.current_step += 1
        self._current_factor = self.factor * decay_factor
        return self._current_factor

    @property
    def current_factor(self):
        return self._current_factor
