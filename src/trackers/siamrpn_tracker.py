import numpy as np
import torch
import torch.nn.functional as F

from src.config import cfg
from src.models.refine import get_refine_net
from src.utils.anchor import Anchors
from src.trackers.base_tracker import Tracker


class SiamRPNTracker(Tracker):
    def __init__(self, net):
        super(SiamRPNTracker, self).__init__()
        self.net = net
        self.net.eval()

        instance_size = cfg.TRACK.INSTANCE_SIZE
        exemplar_size = cfg.TRACK.EXEMPLAR_SIZE
        stride = cfg.ANCHOR.STRIDE
        base_size = cfg.TRACK.BASE_SIZE

        self.score_size = \
            (instance_size - exemplar_size) // stride + 1 + base_size

        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)

        if cfg.REFINE_TEMPLATE.METHOD == 'NETWORK':
            self.refine = get_refine_net(cfg.REFINE_TEMPLATE.NETWORK.MODEL)
            checkpoint = torch.load(cfg.REFINE_TEMPLATE.NETWORK.CHECKPOINT)
            self.refine.load_state_dict(checkpoint['state_dict'])
            self.refine.eval().to(cfg.DEVICE)

    def init(self, img, bbox):
        """
        :param img:
        :param bbox: (x, y, w, h)
        """
        bbox = np.array([bbox[0]+(bbox[2]-1)/2, bbox[1]+(bbox[3]-1)/2, bbox[2], bbox[3]])
        self.center_pos = bbox[:2]
        self.size = bbox[2:]
        self.channel_average = np.mean(img, axis=(0, 1))
        self.zf = self.extract_template(img, bbox)

        if cfg.REFINE_TEMPLATE.METHOD != 'OFF':
            # As self.zf will be updated after each frame,
            # save actual initial template features
            self.zf_init = self.zf

    def track(self, img):
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = Tracker.get_subwindow(
            img, self.center_pos, cfg.TRACK.INSTANCE_SIZE,
            round(s_x), self.channel_average
        )
        xf = self.net.extract_features(x_crop)
        infer_res = self.net.track(xf, self.zf)
        score = self.convert_score(infer_res['cls'])
        pred_bbox = self.convert_bbox(infer_res['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(
            sz(pred_bbox[2, :], pred_bbox[3, :]) /
            (sz(self.size[0] * scale_z, self.size[1] * scale_z))
        )
        # aspect ratio penalty
        r_c = change(
            (self.size[0]/self.size[1]) /
            (pred_bbox[2, :]/pred_bbox[3, :])
        )
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        win_influence = cfg.TRACK.WINDOW_INFLUENCE
        pscore = pscore * (1 - win_influence) + self.window * win_influence

        # find largest activaton index
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self.bbox_clip(
            cx, cy, width, height, img.shape[:2]
        )
        # update tracker state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [
            cx - width / 2,
            cy - height / 2,
            width,
            height
        ]
        best_score = score[best_idx]

        outputs = {
            'bbox': bbox,
            'best_score': best_score,
        }

        # --------------------  REFINE  --------------------
        if cfg.REFINE_TEMPLATE.METHOD != 'OFF':
            refined = self.zf  # refined template (for stage > 1)
            # extract features from new detection
            current = self.extract_template(img, np.array([cx, cy, width, height]))

            if cfg.REFINE_TEMPLATE.METHOD == 'LINEAR':
                lr = cfg.REFINE_TEMPLATE.LINEAR.RATE
                self.zf = (1 - lr) * refined + lr * current

            elif cfg.REFINE_TEMPLATE.METHOD == 'NETWORK':
                net_input = torch.cat((self.zf_init, refined, current), 1)
                self.zf = self.refine(net_input)

            outputs['zf'] = current.cpu().data  # for dataset collection

        return outputs

    def extract_template(self, img, bbox):
        pos, size = bbox[:2], bbox[2:]
        w_z = size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        h_z = size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(size)
        s_z = round(np.sqrt(w_z * h_z))
        z_crop = self.get_subwindow(
            img, pos, cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average
        )
        zf = self.net.extract_features(z_crop)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.net.neck(zf)
        return zf

    @staticmethod
    def bbox_clip(cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    @staticmethod
    def convert_bbox(delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    @staticmethod
    def convert_score(score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    @staticmethod
    def generate_anchor(score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1+x2)*0.5, (y1+y2)*0.5, x2-x1, y2-y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor
