import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


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


def onehot_cross_entropy(input, target, weight=None, size_average=True, ignore_index=-1):
    """
    :param input: NxC
    :param target: NxC
    :return:
    """

    # NxC
    log_p = F.log_softmax(input)

    nll = (log_p * target.float()).sum(dim=1)

    if weight:
        nll *= weight

    if ignore_index:
        # TODO: implement ignore index
        fixing = 100

    if size_average:
        nll /= nll.size(0)

    return -nll.sum()


if __name__ == "__main__":
    from onehot import make_onehot, check1d

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = OnehotCrossEntropyLoss()

    batchSz = 10
    nClasses = 3
    random_input = torch.randn(batchSz, nClasses)
    random_target = torch.LongTensor(batchSz).random_(nClasses)
    random_onehot = make_onehot(random_target, nClasses)

    random_input = Variable(random_input)
    random_target = Variable(random_target)
    random_onehot = Variable(random_onehot)

    res1 = F.cross_entropy(random_input, random_target)
    res2 = onehot_cross_entropy(random_input, random_onehot)
    assert res1.data.equal(res2.data), "error"