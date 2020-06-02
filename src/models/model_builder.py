import torch.nn.functional as F

from src.config import cfg
from src.models.base import Model
from src.models.backbone import get_backbone
from src.models.head import get_rpn_head, get_mask_head, get_refine_head
from src.models.neck import get_neck


class ModelBuilder(Model):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def extract_features(self, img):
        img_f = self.backbone(img)
        return img_f

    def track(self, xf, zf):
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        cls, loc = self.rpn_head(zf, xf)

        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
        return {
            'cls': cls,
            'loc': loc,
            'mask': mask if cfg.MASK.MASK else None
        }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls
