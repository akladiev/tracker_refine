import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.neck.neck import AdjustLayer, AdjustAllLayer

NECKS = {
         'AdjustLayer': AdjustLayer,
         'AdjustAllLayer': AdjustAllLayer
        }


def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)
