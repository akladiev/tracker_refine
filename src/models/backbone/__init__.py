from src.models.backbone.alexnet import alexnet
from src.models.backbone.resnet_atrous import resnet50

BACKBONES = {
              'resnet50': resnet50,
              'alexnet': alexnet,
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
