import torch
import unittest
import numpy as np
import fastmri.utils

class Real2chCNN(torch.nn.Module):
    def __init__(self, dim='2D', input_dim=1, filters=64, kernel_size=3, num_layer=5,
                       activation='relu', use_bias=True, **kwargs):
        super().__init__()
        # get correct conv operator
        if dim == '2D':
            conv_layer = torch.nn.Conv2d
        elif dim == '3D':
            conv_layer = torch.nn.Conv3d
        else:
            raise RuntimeError(f"Convlutions for dim={dim} not implemented!")

        if activation == 'relu':
            act_layer = torch.nn.ReLU

        padding = kernel_size // 2
        # create layers
        self.ops = []
        self.ops.append(conv_layer(input_dim * 2, filters, kernel_size, padding=padding,
                                        bias=use_bias,**kwargs))
        self.ops.append(act_layer(inplace=True))
            
        for _ in range(num_layer-2):
            self.ops.append(conv_layer(filters, filters, kernel_size, padding=padding,
                                        bias=use_bias,**kwargs))
            self.ops.append(act_layer(inplace=True))

        self.ops.append(conv_layer(filters, input_dim * 2, kernel_size,
                                    bias=False,
                                    padding=padding, **kwargs))
        self.ops = torch.nn.Sequential(*self.ops)
        self.apply(self.weight_initializer)

    def weight_initializer(self, module):
        if isinstance(module, torch.nn.Conv2d) \
           or isinstance(module, torch.nn.Linear):
            # equivalent to tf.layers.xavier_initalization()
            torch.nn.init.xavier_uniform_(module.weight, gain=1)
            if module.bias is not None:
                module.bias.data.fill_(0)

    def forward(self, inputs):
        x = fastmri.utils.complex2real(inputs)
        x = self.ops(x)
        return fastmri.utils.real2complex(x)

class TestCNN(unittest.TestCase):
    def testCnn(self):
        input_dim = 2
        x = np.random.randn(5, input_dim, 11, 11) + 1j * np.random.randn(5, input_dim, 11, 11)
        op = Real2chCNN(input_dim=input_dim).double()
        y = op(torch.from_numpy(x))
        print(op)