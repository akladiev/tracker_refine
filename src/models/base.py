import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def extract_features(self, img):
        """ Extract deep features from the given image
        :param img: BGR image (np.ndarray)
        :returns: np array/torch tensor with extracted deep features
        """
        raise NotImplementedError

    def track(self, xf, zf):
        """
        :param xf: deep features of image to track template on
        :param zf: deep features of template to track
        :returns: dict with tracking results
        """
        raise NotImplementedError
