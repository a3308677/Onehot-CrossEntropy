import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"

criterion1 = nn.CrossEntropyLoss()


class OnehotCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, size_average=False, ignore_index=-1):
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        :param input: NxC
        :param target: NxC
        :return:
        """
        _assert_no_grad(target)

        # NxC
        log_p = F.log_softmax(input)

        nll = (log_p * target).sum(dim=1)
        if self.weight:
            nll *= self.weight

        if self.ignore_index:
            # TODO: implement ignore index
            fixing = 100

        if self.size_average:
            # TODO: fix total
            nll /= 100

        return nll

