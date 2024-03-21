import merlinth
import torch.nn as nn


class MoDL_2D(nn.Module):
    def __init__(self, dim='2D', filters=24, kernel_size=(3, 3, 3), pool_size=2, num_layer_per_block=4,
                       activation='ReLU', normalization='BN', use_bias=True, n_channels=[2, 2], name='MoDL_2D',  **kwargs):
        # 2-channel real-valued input (as original paper)

        super(MoDL_2D, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_layer_per_block = num_layer_per_block
        self.filters = filters
        self.use_bias = use_bias
        self.activation = activation
        self.conv_layer = nn.Conv2d  # merlinth.layers.ComplexConv2d

        # get normalization operator
        normalizations = {
            'bn': nn.BatchNorm2d,
            'in': nn.InstanceNorm2d,
            'none': None
        }
        self.norm_layer = normalizations[normalization.lower()]

        # get activation operator
        activations = {
            'relu': nn.ReLU,
            'leakyrelu': nn.LeakyReLU,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'linear': nn.Linear,
            #'modrelu': merlinth.layers.ModReLU,
            'none': None
        }
        activations_arg = {
            'relu': {'inplace': True},
            'leakyrelu': {'negative_slope': 0.1, 'inplace': True},
            'sigmoid': {'inplace': True},
            'tanh': {'inplace': True},
            'linear': {'in_features': 2, 'out_features': 2, 'bias': True},
            'modrelu': {'num_parameters': 32, 'bias_init': 0.1, 'requires_grad': True}
        }

        self.activation_layer = activations[activation.lower()]
        self.activation_layer_args = activations_arg[activation.lower()]

        self.Nw = []
        in_channels = list(n_channels)[0]
        for ilayer in range(self.num_layer_per_block - 1):
            self.Nw.append(self.conv_layer(in_channels, self.filters, kernel_size=self.kernel_size, padding='same', bias=self.use_bias))
            in_channels = self.filters
            self.Nw.append(self.norm_layer(in_channels))
            self.Nw.append(self.activation_layer())
        self.Nw.append(self.conv_layer(in_channels, list(n_channels)[1], kernel_size=self.kernel_size, padding='same', bias=self.use_bias))
        self.Nw.append(self.norm_layer(list(n_channels)[1]))
        self.Nw = nn.ModuleList(self.Nw)

    def forward(self, x):
        for op in self.Nw:
            x = op(x)
        return x
