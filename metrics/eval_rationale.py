import numpy
import torch
import torchmetrics


class RationaleEvaler:
    def __init__(self, task_classes: int, device: torch.device, task_type: str = 'regression', ):
        """
        RationaleEvaler-Rationale
        :param task_classes: The number of categories in the classification task
        :param device: gpu or cpu
        :param task_type: 'regression' or 'classification'
        """
        # regression or classification
        self.task_type = task_type
        if task_type == 'classification':
            self.pEvaler = PerformanceEvaler(task_classes, device, task_type)
        else:
            self.task_acc = torchmetrics.MeanSquaredError(True)
        # Rationale has only two states that are selected or unselected.
        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.recall = torchmetrics.classification.BinaryRecall()
        self.precision = torchmetrics.classification.BinaryPrecision()
        self.f1 = torchmetrics.classification.BinaryF1Score()
        self.to(device)

    def step(self, task_prediction: torch.Tensor, task_label: torch.Tensor, rationale: torch.Tensor,
             human_rationale: torch.Tensor):
        """
        Update the evaluation metrics once.
        :param task_prediction: Results of task prediction (logits)
        :param task_label: Task label (un-squeezed)
        :param rationale: the tensor of binary rationale mask
        :param human_rationale: human gold rationale (binary) with same shape as the param rationale.
        :return: None
        """
        acc = self.acc(rationale, human_rationale)
        recall = self.recall(rationale, human_rationale)
        precision = self.precision(rationale, human_rationale)
        f1score = self.f1(rationale, human_rationale)
        if self.task_type == 'classification':
            task_acc, task_pre, task_recall, task_f1 = self.pEvaler.step(task_prediction, task_label)
            return f'task_acc:{task_acc}\n' \
                   f'task_precision:{task_pre}\n' \
                   f'task_recall:{task_recall}\n' \
                   f'task_f1:{task_f1}\n' \
                   f'token_acc:{acc}\n' \
                   f'token_recall:{recall}\n' \
                   f'token_precision:{precision}\n' \
                   f'token_f1:{f1score}\n'
        elif self.task_type == 'regression':
            task_acc = self.task_acc(task_prediction, task_label)
            return f'mse:{task_acc}\n' \
                   f'token_acc:{acc}\n' \
                   f'token_recall:{recall}\n' \
                   f'token_precision:{precision}\n' \
                   f'token_f1:{f1score}\n'
        else:
            raise NotImplementedError

    def compute(self):
        """report final results"""
        acc = self.acc.compute()
        recall = self.recall.compute()
        precision = self.precision.compute()
        f1score = self.f1.compute()
        if self.task_type == 'regression':
            task_acc = self.task_acc.compute()
            return f'mse:{task_acc}\n' \
                   f'token_acc:{acc}\n' \
                   f'token_recall:{recall}\n' \
                   f'token_precision:{precision}\n' \
                   f'token_f1:{f1score}\n'
        elif self.task_type == 'classification':
            task_acc, task_pre, task_recall, task_f1 = self.pEvaler.compute()
            return f'task_acc:{task_acc} task_precision:{task_pre} ' \
                   f'task_recall:{task_recall} task_f1:{task_f1}\n\n' \
                   f'token_acc:{acc}\n' \
                   f'token_recall:{recall}\n' \
                   f'token_precision:{precision}\n' \
                   f'token_f1:{f1score}\n'

    def reset(self):
        self.acc.reset()
        self.recall.reset()
        self.precision.reset()
        self.f1.reset()
        if self.task_type == 'classification':
            self.pEvaler.reset()
        elif self.task_type == 'regression':
            self.task_acc.reset()
        else:
            raise NotImplementedError

    def to(self, device):
        self.acc.to(device)
        self.precision.to(device)
        self.recall.to(device)
        self.f1.to(device)
        if self.task_type == 'classification':
            self.pEvaler.to(device)
        elif self.task_type == 'regression':
            self.task_acc.to(device)
        else:
            raise NotImplementedError


class PerformanceEvaler:
    def __init__(self, task_classes: int, device: torch.device, task_type: str = 'regression', ):
        """
        RationaleEvaler-Rationale
        :param task_classes: The number of categories in the classification task
        :param device: gpu or cpu
        :param task_type: 'regression' or 'classification'
        """
        # regression or classification
        self.task_type = task_type
        if task_type == 'classification':
            self.task_acc = torchmetrics.Accuracy(task="multiclass", average='macro', num_classes=task_classes)
            self.task_pre = torchmetrics.Precision(task="multiclass", average='macro', num_classes=task_classes)
            self.task_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=task_classes)
            self.task_f1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=task_classes)
        else:
            self.task_acc = torchmetrics.MeanSquaredError(True)
        self.to(device)

    def step(self, task_prediction: torch.Tensor, task_label: torch.Tensor):
        """
        Update the performance metrics once.
        :param task_prediction: Results of task prediction (logits)
        :param task_label: Task label (un-squeezed)
        :return: None
        """
        if self.task_type == 'classification':
            task_prediction = torch.nn.functional.softmax(task_prediction, dim=-1)
            task_label = task_label.squeeze(dim=-1)
            task_acc = self.task_acc(task_prediction, task_label)
            task_pre = self.task_pre(task_prediction, task_label)
            task_recall = self.task_recall(task_prediction, task_label)
            task_f1 = self.task_f1(task_prediction, task_label)
            return task_acc, task_pre, task_recall, task_f1
        task_acc = self.task_acc(task_prediction, task_label)
        return task_acc

    def compute(self):
        """
        report final results
        task_acc,task_pre,task_recall,task_f1 or mse
        """
        task_acc = self.task_acc.compute()
        if self.task_type == 'classification':
            task_acc = self.task_acc.compute()
            task_pre = self.task_pre.compute()
            task_recall = self.task_recall.compute()
            task_f1 = self.task_f1.compute()
            return task_acc, task_pre, task_recall, task_f1

        return task_acc

    def reset(self):
        self.task_acc.reset()
        if self.task_type == 'classification':
            self.task_pre.reset()
            self.task_recall.reset()
            self.task_f1.reset()

    def to(self, device):
        self.task_acc.to(device)
        if self.task_type == 'classification':
            self.task_pre.to(device)
            self.task_recall.to(device)
            self.task_f1.to(device)


class RationaleStatistic:
    """
    Count the proportion of words selected by rationale.
    """

    def __init__(self):
        self.sparsity = []

    def step(self, rationale: torch.Tensor, mask: torch.Tensor):
        """update once."""
        true_sentence_length = torch.sum(mask, dim=-1)
        selection_sparsity = torch.sum(rationale, dim=-1) / true_sentence_length
        self.sparsity.append(selection_sparsity.detach().mean().item())

    def compute(self):
        return numpy.mean(self.sparsity)
