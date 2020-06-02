import torch.nn as nn


class Refine2L(nn.Module):
    def __init__(self):
        super(Refine2L, self).__init__()

        self.net = nn.Sequential(
           nn.Conv2d(768, 192, 1),
           nn.ReLU(inplace=True),
           nn.Conv2d(192, 256, 1),
        )

    def forward(self, x):
        return self.net(x)


class Refine3L(nn.Module):
    def __init__(self):
        super(Refine3L, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(768, 192, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, 1),
        )

    def forward(self, x):
        return self.net(x)


REFINE = {
    'Refine2L': Refine2L,
    'Refine3L': Refine3L,
}


def get_refine_net(name, **kwargs):
    return REFINE[name](**kwargs)
