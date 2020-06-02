from src.models.head.mask import MaskCorr, Refine
from src.models.head.rpn import DepthwiseRPN

RPNS = {
        'DepthwiseRPN': DepthwiseRPN,
       }

MASKS = {
         'MaskCorr': MaskCorr,
        }

REFINE = {
          'Refine': Refine,
         }


def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)


def get_mask_head(name, **kwargs):
    return MASKS[name](**kwargs)


def get_refine_head(name):
    return REFINE[name]()
