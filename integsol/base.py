from abc import ABC
import torch

class BaseClass(ABC):
    torch.set_default_dtype(torch.float64)
    pass

