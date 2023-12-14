import torch
from typing import Literal, Optional
from model.common import FactorAnnealer, JensenShannonDivergence, RNNModule, GuidanceModule

from dataclasses import dataclass
from model.model_utils import ModelOutput


@dataclass
class STModelOutput(ModelOutput):
    final_states: Optional[torch.FloatTensor] = None


# Our re-implementations:
# RNPSTModel(Re-RNP) - Tao Lei's RNP(2016) method but using the Gumbel Softmax instead of using REINFORCE.
# RNPSTShareModel(FR) - Liu wei's FR(2022) named 'Folded Rationalization with a Unified Encoder'.

class RNPSTModel(torch.nn.Module):
    """
    Rationale model using the Straight-Through trick.
    Refer to 'yala/text_nn'(https://github.com/yala/text_nn).
    """

    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, output_size: int, num_layers: int = 1,
                 dropout: float = 0.2, sparsity: float = None, continuity_lambda: float = None,
                 sparsity_lambda: float = None, cell_type: Literal['rcnn', 'gru'] = 'gru',
                 pretrained_embedding=None, fix_embedding=True):
        super(RNPSTModel, self).__init__()
        self.task_lambda = 1.0
        self.sparsity_lambda = sparsity_lambda
        self.continuity_lambda = continuity_lambda
        self.sparsity = sparsity
        self.embedding_layer = torch.nn.Embedding(vocab_size, emb_size, padding_idx=1)
        if pretrained_embedding is not None:
            self.embedding_layer.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding_layer.weight.requires_grad = not fix_embedding
        self.tau = 1.0

        self.selector_cell = RNNModule(cell_type, emb_size, hidden_size, num_layers)
        self.predictor_cell = RNNModule(cell_type, emb_size, hidden_size, num_layers)
        self.selector_head = torch.nn.Linear(hidden_size, 2)
        self.predictor_head = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.select_logits = None

    def forward(self, input_ids: torch.Tensor, masks: torch.Tensor, return_final_states: bool = False):
        """
        :param input_ids: (batch, seq_len) — Indices of input sequence tokens in the vocabulary.
        :param masks: (batch, seq_len) — Mask to avoid performing calculation on padding token indices.
        :param return_final_states: whether to return the hidden states of the final layer.
        :return: rationale_masks, final_logits, final_states(if return)
        """
        extend_mask = masks.float().unsqueeze(-1)
        # select
        embedding = extend_mask * self.embedding_layer(input_ids)
        select_output, _ = self.selector_cell(embedding)
        select_output = self.layer_norm(select_output)
        select_logits = self.selector_head(self.dropout(select_output))
        self.select_logits = select_logits
        # sample
        if self.training:
            rationale_masks = torch.nn.functional.gumbel_softmax(select_logits, tau=self.tau, hard=True, dim=-1)
            rationale_masks = rationale_masks[:, :, 1]
        else:
            rationale_masks = (torch.nn.functional.softmax(select_logits, dim=-1)[:, :, 1] >= 0.5).float()
        # predict
        predictor_embedding = embedding * (rationale_masks.unsqueeze(-1))
        predictor_cell_outputs, _ = self.predictor_cell(predictor_embedding)
        predictor_cell_outputs = self.layer_norm(predictor_cell_outputs)
        predictor_cell_outputs = predictor_cell_outputs * extend_mask + (1. - extend_mask) * torch.finfo(
            torch.float).min
        final_states = torch.transpose(predictor_cell_outputs, 1, 2)
        final_states, _ = torch.max(final_states, dim=2)
        final_logits = self.predictor_head(self.dropout(final_states))

        if return_final_states:
            return STModelOutput(rationales=rationale_masks, predictions=final_logits, final_states=final_states)

        return STModelOutput(rationales=rationale_masks, predictions=final_logits)

    def induce_skew_selector(self, input_ids: torch.Tensor, masks: torch.Tensor):
        """
        used to train a skewed selector.
        :param input_ids: (batch, seq_len) — Indices of input sequence tokens in the vocabulary.
        :param masks: (batch, seq_len) — Mask to avoid performing calculation on padding token indices.
        :return: selector_logits, rationale_masks[:,:,1]
        """
        #  masks_ (batch_size, seq_length, 1)
        extend_mask = masks.float().unsqueeze(-1)
        # select
        embedding = extend_mask * self.embedding_layer(input_ids)  # (batch, seq_len, embedding_dim)
        selector_last_hidden_states, _ = self.selector_cell(embedding)  # (batch, seq_len, hidden_dim)
        selector_last_hidden_states = self.layer_norm(selector_last_hidden_states)  # (batch, seq_len, hidden_dim)
        selector_logits = self.selector_head(self.dropout(selector_last_hidden_states))  # (batch_size, seq_len, 2)
        # sample
        if self.training:
            rationale_masks = torch.nn.functional.gumbel_softmax(selector_logits, self.tau, hard=True, dim=-1)
        else:
            rationale_masks = (torch.nn.functional.softmax(selector_logits, dim=-1) >= 0.5).float()

        # finish
        return selector_logits, rationale_masks[:, :, 1]

    def induce_skew_predictor(self, input_ids: torch.Tensor, masks: torch.Tensor):
        """
        used to train a skewed predictor.
        :param input_ids: (batch, seq_len) — Indices of input sequence tokens in the vocabulary.
        :param masks: (batch, seq_len) — Mask to avoid performing calculation on padding token indices.
        :return: selector_logits, rationale_masks[:,:,1]
        """
        #  masks_ (batch_size, seq_length, 1)
        extend_mask = masks.float().unsqueeze(-1)
        # predict
        embedding = self.embedding_layer(input_ids)
        predictor_embedding = embedding * extend_mask
        predictor_cell_outputs, _ = self.predictor_cell(predictor_embedding)
        predictor_cell_outputs = self.layer_norm(predictor_cell_outputs)
        predictor_cell_outputs = predictor_cell_outputs * extend_mask + (1. - extend_mask) * torch.finfo(
            torch.float).min
        final_states = torch.transpose(predictor_cell_outputs, 1, 2)
        final_states, _ = torch.max(final_states, dim=2)
        final_logits = self.predictor_head(self.dropout(final_states))

        return final_logits

    def calculate_skew_loss(self, skew_logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor,
                            skew_type: Literal['simple', 'hard', 'predictor'] = 'hard', rationale_mask=None):
        """
        used to train a skewed selector or predictor.
        :param skew_logits: [batch,seq_len,2] - The selector/predictor scores (before SoftMax).
        :param targets: (batch, 1) - Labels for computing the skew loss.
        :param masks: (batch, seq_len) - Mask to avoid performing calculation on padding token indices.
        :param skew_type: Choose a way(simple or hard) to induce the selector degradation.
        :param rationale_mask: (batch, seq_len) - Model selections.
        :return: total_loss, loss_info_optional
        """
        loss_info_optional = {}

        # induce degeneration
        if skew_type == 'predictor':
            loss_predictor = torch.nn.functional.cross_entropy(skew_logits.view(-1, 2),
                                                               targets.view(-1))
            loss_info_optional["loss_p"] = loss_predictor.item()
            total_loss = loss_predictor
        elif skew_type == 'simple':
            raise DeprecationWarning
            # sentence_first_token_logits = skew_logits[:, 0, :]
            # loss_generator = torch.nn.functional.cross_entropy(
            #       sentence_first_token_logits.view(-1, 2),targets.view(-1)
            # )
        elif skew_type == 'hard':
            sentence_first_token_logits = skew_logits[:, 0, :]
            model_sparsity = torch.sum(rationale_mask) / torch.sum(masks.float())
            sparsity_loss = self.sparsity_lambda * torch.abs(model_sparsity - self.sparsity)
            model_continuity = torch.mean(torch.abs(rationale_mask[:, 1:] - rationale_mask[:, :-1]))
            continuity_loss = self.continuity_lambda * model_continuity
            loss_generator = torch.nn.functional.cross_entropy(sentence_first_token_logits.view(-1, 2),
                                                               targets.view(-1))
            loss_info_optional["sparsity"] = model_sparsity.item()
            loss_info_optional["continuity"] = continuity_loss.item()
            loss_info_optional["loss_g"] = loss_generator.item()
            total_loss = loss_generator + sparsity_loss + continuity_loss
        else:
            raise NotImplementedError(f'skew_type:{skew_type} is not implemented yet.')

        return total_loss, loss_info_optional

    def calculate_loss_(self, logits: torch.Tensor, targets: torch.Tensor, rationale_mask: torch.Tensor,
                        masks: torch.Tensor):
        """
        The inward loss interface.
        :param logits: [batch,seq_len,2] - The predictor scores (before SoftMax).
        :param targets: (batch, 1) - Labels for computing the task loss.
        :param rationale_mask: (batch, seq_len) - Model selections.
        :param masks: Mask to avoid performing calculation on padding token indices.
        :return: loss, loss_optional
        """
        loss_optional = {}
        task_loss = torch.nn.functional.cross_entropy(logits, targets.squeeze())
        model_sparsity = torch.sum(rationale_mask) / torch.sum(masks.float())
        sparsity_loss = self.sparsity_lambda * torch.abs(model_sparsity - self.sparsity)
        model_continuity = torch.mean(torch.abs(rationale_mask[:, 1:] - rationale_mask[:, :-1]))
        continuity_loss = self.continuity_lambda * model_continuity
        loss = self.task_lambda * task_loss + sparsity_loss + continuity_loss

        loss_optional['task'] = task_loss.item()
        loss_optional['sparsity'] = model_sparsity.item()
        loss_optional['continuity'] = model_continuity.item()
        loss_optional['loss_s'] = sparsity_loss.item()
        loss_optional['loss_c'] = continuity_loss.item()
        loss_optional['loss'] = loss.item()

        return loss, loss_optional

    def calculate_loss(self, model_output: STModelOutput, targets: torch.Tensor, masks: torch.Tensor):
        """
        :param model_output: The result of model forward(...).
        :param targets: (batch, 1) - Labels for computing the task loss.
        :param masks: Mask to avoid performing calculation on padding token indices.
        :return: loss, loss_optional
        """
        return self.calculate_loss_(model_output.predictions, targets, model_output.rationales, masks)


class RNPSTShareModel(torch.nn.Module):
    """
    Rationale model using Straight-through trick, and sharing the cell for its predictor and selector.
    Refer to 'FR: Folded Rationalization with a Unified Encoder'(https://arxiv.org/abs/2209.08285).
    """

    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, output_size: int, num_layers: int = 1,
                 dropout: float = 0.2, sparsity: float = None, continuity_lambda: float = None,
                 sparsity_lambda: float = None, cell_type: Literal['rcnn', 'gru'] = 'gru', pretrained_embedding=None,
                 fix_embedding=True):
        super(RNPSTShareModel, self).__init__()
        self.task_lambda = 1.0
        self.sparsity_lambda = sparsity_lambda
        self.continuity_lambda = continuity_lambda
        self.sparsity = sparsity
        self.embedding_layer = torch.nn.Embedding(vocab_size, emb_size, padding_idx=1)
        if pretrained_embedding is not None:
            self.embedding_layer.weight.data = torch.from_numpy(pretrained_embedding)
        self.embedding_layer.weight.requires_grad = not fix_embedding
        self.tau = 1.0
        self.share_cell = RNNModule(cell_type, emb_size, hidden_size, num_layers)
        self.selector_head = torch.nn.Linear(hidden_size, 2)
        self.predictor_head = torch.nn.Linear(hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, masks: torch.Tensor):
        """
        :param input_ids: (batch, seq_len) — Indices of input sequence tokens in the vocabulary.
        :param masks: (batch, seq_len) — Mask to avoid performing calculation on padding token indices.
        :return: rationale_masks, final_logits
        """
        extend_mask = masks.float().unsqueeze(-1)
        # select
        embedding = extend_mask * self.embedding_layer(input_ids)
        select_output, _ = self.share_cell(embedding)
        select_output = self.layer_norm(select_output)
        select_logits = self.selector_head(self.dropout(select_output))
        # sample
        if self.training:
            rationale_masks = torch.nn.functional.gumbel_softmax(select_logits, self.tau, hard=True, dim=-1)
            rationale_masks = rationale_masks[:, :, 1]
        else:
            rationale_masks = (torch.nn.functional.softmax(select_logits, dim=-1) >= 0.5).float()
            rationale_masks = rationale_masks[:, :, 1]
        # predict
        rationale_masks = rationale_masks * masks.float()
        predictor_embedding = embedding * (rationale_masks.unsqueeze(-1))
        predictor_cell_outputs, _ = self.share_cell(predictor_embedding)
        predictor_cell_outputs = self.layer_norm(predictor_cell_outputs)
        predictor_cell_outputs = predictor_cell_outputs * extend_mask + (1. - extend_mask) * torch.finfo(
            torch.float).min
        predictor_cell_outputs = torch.transpose(predictor_cell_outputs, 1, 2)
        predictor_cell_outputs, _ = torch.max(predictor_cell_outputs, dim=2)
        final_logits = self.predictor_head(self.dropout(predictor_cell_outputs))

        return STModelOutput(rationales=rationale_masks, predictions=final_logits)

    def calculate_loss(self, model_output: STModelOutput, targets: torch.Tensor, masks: torch.Tensor):
        """
        :param model_output: The result of model forward(...).
        :param targets: (batch, 1) - Labels for computing the skew loss.
        :param masks: Mask to avoid performing calculation on padding token indices.
        :return: loss, loss_optional
        """
        loss_optional = {}
        rationale_masks, final_logits = model_output.rationales, model_output.predictions
        task_loss = torch.nn.functional.cross_entropy(final_logits, targets.squeeze())
        model_sparsity = torch.sum(rationale_masks) / torch.sum(masks.float())
        sparsity_loss = self.sparsity_lambda * torch.abs(model_sparsity - self.sparsity)
        model_continuity = torch.mean(torch.abs(rationale_masks[:, 1:] - rationale_masks[:, :-1]))
        continuity_loss = self.continuity_lambda * model_continuity
        loss = self.task_lambda * task_loss + sparsity_loss + continuity_loss

        loss_optional['task'] = task_loss.item()
        loss_optional['sparsity'] = model_sparsity.item()
        loss_optional['continuity'] = model_continuity.item()
        loss_optional['l_spar'] = sparsity_loss.item()
        loss_optional['l_con'] = continuity_loss.item()
        loss_optional['loss'] = loss.item()

        return loss, loss_optional


# Our methods:
# GuidanceModule - The teacher model is used to guide the rationale model during training.
# GuidanceBasedRationaleModule - Inherited from RNPSTModel.

class GuidanceBasedRationaleModule(RNPSTModel):
    """
    RNN - HardSelection - RNN.
    """

    def __init__(self, vocab_size: int, emb_size: int, hidden_size: int, output_size: int, num_layers: int = 1,
                 dropout: float = 0.2, sparsity: float = None, continuity_lambda: float = None,
                 sparsity_lambda: float = None, cell_type: Literal['rcnn', 'gru'] = 'gru',
                 pretrained_embedding=None, fix_embedding=True,
                 guide_lambda: float = None, match_lambda: float = None, guide_decay: float = None):
        super(GuidanceBasedRationaleModule, self).__init__(vocab_size, emb_size, hidden_size, output_size, num_layers,
                                                           dropout, sparsity, continuity_lambda, sparsity_lambda,
                                                           cell_type,
                                                           pretrained_embedding, fix_embedding)
        self.guide_lambda = guide_lambda
        self.match_lambda = match_lambda
        self.guide_decay = guide_decay

        self.match_criterion = JensenShannonDivergence()

        def decay_func(current_step: int):
            return max(-float(current_step) * self.guide_decay + 1.0, 0.0)

        self.guide_annealer: FactorAnnealer = FactorAnnealer(1.0, decay_func)

    def calculate_loss(self, model_output: STModelOutput, targets: torch.Tensor, masks: torch.Tensor,
                       guider: GuidanceModule = None, input_ids: torch.Tensor = None):
        """
        :param model_output: The result of model forward(...).
        :param targets: (batch, 1) - Labels for computing the skew loss.
        :param masks: Mask to avoid performing calculation on padding token indices.
        :param guider: the Guidance Module.
        :param input_ids: (batch, seq_len) — Indices of input sequence tokens in the vocabulary.
        :return: loss, loss_optional
        """
        loss, loss_info = self.calculate_loss_(model_output.predictions, targets, model_output.rationales, masks)
        guide_factor = self.guide_annealer.current_factor
        guide_dynamic_lambda = self.guide_lambda * guide_factor
        # guide
        guider.eval()
        soft_weights, guider_logits = guider.forward(input_ids, masks)
        scaling_soft_probs = guider.get_expand_token_scores(soft_weights, masks, self.sparsity)
        guide_bce = torch.nn.functional.binary_cross_entropy_with_logits(self.select_logits[:, :, 1],
                                                                         scaling_soft_probs)
        guide_loss = guide_bce * guide_dynamic_lambda
        # match
        js_div = self.match_criterion(model_output.predictions, guider_logits.detach())
        js_loss = js_div * self.match_lambda * (1. - guide_factor)
        # total loss
        loss = loss + guide_loss + js_loss
        # save loss info
        loss_info['cur_lambda'] = guide_dynamic_lambda
        loss_info['bce'] = guide_bce.item()
        loss_info['js'] = js_div.item()
        loss_info['js_loss'] = js_loss.item()
        loss_info['loss'] = loss.item()
        loss_info['anneal'] = guide_factor
        if guide_dynamic_lambda > 0.001:
            loss_info['guide_loss'] = guide_loss.item()

        # step guide_annealer
        if self.training:
            self.guide_annealer.step()

        return loss, loss_info
