import torch
import torch.nn as nn
from torch.autograd import Variable

class Scalar(nn.Module):
    def __init__(self, name, scale=1, trainable=True,
                    constraint=None, initializer=None):
        super().__init__()

        self.scale = scale  # to potentially accelerate training!
        self.weight_constraint = constraint  # not possible in pytorch
        self.weight_initializer = initializer  # TODO: call respective init, for now: init value
        self.weight_trainable = trainable
        self.weight_name = name

        self._weight = Variable(torch.rand(int(initializer)) * self.scale, requires_grad=trainable)

    def forward(self, x):
        return x * self._weight.expand_as(x).cuda()