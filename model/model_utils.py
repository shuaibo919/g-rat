from dataclasses import dataclass

import torch


@dataclass
class ModelOutput:
    predictions: torch.FloatTensor = None
    rationales: torch.Tensor = None



