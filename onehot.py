from __future__ import absolute_import, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn

import numpy as np

from crossentropy import OnehotCrossEntropyLoss

onehot_cross_entropy = OnehotCrossEntropyLoss()

def onehot(input, n_classes):
    """
    :param input: Nx1, where all numbers are from 0 ~ C-1
    :param n_classes: C
    :return:
    """
    batch_size = input.size(0)
    y_onehot = torch.LongTensor(batch_size, n_classes)
    y_onehot.zero_()
    return y_onehot.scatter_(1, input, 1)


def onehot2d(input, batch_size, n_classes):
    assert input.dim() == 3, "input must be a NxHxW tensor"
    assert input.type() == 'torch.LongTensor', "The data type of input must be torch.LongTensor"

    input_shape = input.size()
    batch_size = input_shape[0]
    # target: N x H x W  ->  (NxHxW) x 1
    onehot_input = input.clone().view(-1, 1)
    onehot_batch_size = onehot_input.size()[0]

    # target: (NxHxW) x 1 -> (NxHxW) x N_class
    # expand to onehot form
    onehot_target = onehot(onehot_input, batch_size=onehot_batch_size, n_classes=n_classes)

    # target: (NxHxW) x N_class -> N x H x W x N_class -> N x N_class x W x H
    onehot_target = onehot_target.view(input_shape[0], input_shape[1], input_shape[2], n_classes).transpose(3, 2).transpose(2, 1).contiguous()

    return onehot_target


def check1d(oneVec, oneHot):
    """
    :param oneVec: Nx1
    :param oneHot: NxC
    :return:
        True/False
    """
    batchSz = oneHot.size(0)
    data1 = torch.ones(batchSz).long()
    data2 = oneHot.sum(dim=1)
    assert oneHot.sum(dim=1).equal(torch.ones(batchSz).long()), "not all dimensions sum to 1"
    for _ in range(batchSz):
        index = oneVec[_, 0]
        assert oneHot[_, index] == 1, "onehot mismatch!"

def test1d(batchRange=[1, 100], classRange=[1, 100]):
    """
    :param batchRange:
    :param classRange:
    :return:
    """
    batchSz = np.random.randint(batchRange[0], batchRange[1])
    nClasses = np.random.randint(classRange[0], classRange[1])
    oneVec = torch.LongTensor(batchSz, 1).random_(nClasses)
    oneHot = onehot(oneVec, nClasses)
    check1d(oneVec, oneHot)


def checkOnehotCE(input, target):
    """
    :param input: NxC softmax out
    :param target: Nx1 ground truth label
    :return: 
    """
    normalCE = F.cross_entropy(input, target)

    nClasses = input.size(1)
    onehotTarget = onehot(target, nClasses)
    onehotCE = onehot_cross_entropy(input, onehotTarget)

    assert onehotCE.data.equal(normalCE.data), "onehot CE is not equal to normal CE, check fail"


if __name__ == "__main__":
    test1d()
