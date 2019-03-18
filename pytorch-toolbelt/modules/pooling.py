"""Implementation of different pooling modules

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return F.avg_pool2d(inputs, kernel_size=in_size[2:])


class GlobalMaxPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalMaxPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return F.max_pool2d(inputs, kernel_size=in_size[2:])


class GWAP(nn.Module):
    """
    Global Weighted Average Pooling from paper "Global Weighted Average Pooling Bridges Pixel-level Localization and Image-level Classification"
    """

    def __init__(self, features, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(features, 1, kernel_size=1, bias=True)
        self.logits = nn.Linear(features, num_classes)

    def fscore(self, x):
        m = self.conv(x)
        m = m.sigmoid().exp()
        return m

    def norm(self, x: torch.Tensor):
        ndim = len(x.size())
        return x / x.sum(dim=list(range(ndim - 2, ndim)), keepdim=True)

    def forward(self, x):
        ndim = len(x.size())
        dims = list(range(ndim - 2, ndim))
        input_x = x
        self.conv(x)
        x = x.sigmoid()
        x = x - torch.logsumexp(x, dim=dims, keepdim=True)
        x = x * input_x
        ndim = len(x.size())
        x = x.sum(dim=dims)
        logits = self.logits(x)
        return logits


class RMSPool(nn.Module):
    """
    Root mean square pooling
    """

    def __init__(self):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d()

    def forward(self, x):
        ndim = len(x.size())
        x_mean = x.mean(dim=list(range(ndim-2, ndim)))
        avg_pool = self.avg_pool((x - x_mean) ** 2)
        return avg_pool.sqrt()


class MILCustomPoolingModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        self.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.weight_generator = nn.Sequential(nn.BatchNorm2d(in_channels),
                                              nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
                                              nn.ReLU(True),
                                              nn.Conv2d(in_channels // reduction, out_channels, kernel_size=1),
                                              nn.Sigmoid())

    def forward(self, x):
        w = self.weight_generator(x)
        l = self.classifier(x)
        ndim = len(x.size())
        dim = list(range(ndim - 2, ndim))
        logits = torch.sum(w * l, dim=dim) / (torch.sum(w, dim=dim) + 1e-6)
        return logits
