import torch
import numpy as np

import merlinth.layers

class SingleCoilProxLayer(torch.nn.Module):
    """
        Data Consistency layer from DC-CNN, apply for single coil mainly
    """
    def __init__(self, weight_init=0.1, weight_scale=1.0, center=True, trainable=True):
        """
        Args:
            lambda_init (float): Init value of data consistency block (DCB)
        """
        super(SingleCoilProxLayer, self).__init__()
        self._weight = torch.nn.Parameter(torch.Tensor(1))
        self._weight.data = torch.tensor(weight_init, dtype=self._weight.dtype)
        self._weight.proj = lambda: self._weight.data.clamp_(0, 1 / weight_scale)
        self.weight_scale = weight_scale

        self.trainable = trainable
        self.set_trainable(trainable)

        self.fft = merlinth.layers.fft2c if center else  merlinth.layers.fft2
        self.ifft =  merlinth.layers.ifft2c if center else  merlinth.layers.ifft2

    @property
    def weight(self):
        return self._weight * self.weight_scale

    def forward(self, inputs):
        x = inputs[0]
        y = inputs[1]
        mask = inputs[2]
        A_x = self.fft(x)
        k_dc = (1 - mask) * A_x + mask * (
            self.weight * A_x + (1 - self.weight) * y)
        x_dc = self.ifft(k_dc)
        return x_dc

    def extra_repr(self):
        return f"weight={self.weight.item():.4g},trainable={self.trainable}"

    def set_trainable(self, flag):
        self.trainable = flag
        self._weight.requires_grad = self.trainable


def get_denoiser(config):
    R_type = config['R_type']

    if R_type == 'real2ch_cnn':
        return merlinth.models.Real2chCNN(**config['R_config'])
    elif R_type == 'complex_cnn':
        return merlinth.models.ComplexCNN(**config['R_config'])
    elif R_type == 'foe':
        return merlinth.models.Real2chFoE(config['R_config'])
    elif R_type == 'magnitude_foe':
        return merlinth.models.MagnitudeFoE(config['R_config'])
    else:
        raise ValueError(f'R_type {R_type} not defined!')

def get_dc(config):
    D_type = config['D_type']

    A = merlinth.layers.MulticoilForwardOp(center=False, coil_axis=-3, channel_dim_defined=True)
    AH = merlinth.layers.MulticoilAdjointOp(center=False, coil_axis=-3, channel_dim_defined=True)

    if D_type == 'gd':
        return merlinth.keras.layers.DCGD(A, AH, **config['D_config'])
    elif D_type == 'pm':
        return SingleCoilProxLayer(**config['A_config'], **config['D_config'])  # TODO: check if centered FFT
    elif D_type == 'none':
        return None
    else:
        raise ValueError(f'D_type {D_type} not defined!')

class VN(torch.nn.Module):
    def __init__(
        self,
        config,
        is_training=True,
        multicoil=False,
        **kwargs
    ):
        super().__init__()
        self.is_training = is_training
        self.S = config['S'] if config['dynamic'] else 1
        self.S_end = config['S']
        self.R_type = config['R_type']
        self.D_type = config['D_type']
        self.multicoil = multicoil

        self.denoiser = torch.nn.ModuleList([get_denoiser(config) for _ in range(self.S)])
        self.dc = torch.nn.ModuleList([get_dc(config) for _ in range(self.S)])

    def forward(self, inputs):
        x = inputs[0]
        constants = list(inputs[1:])

        for i in range(self.S_end):
            # Denoising step
            ii = i % len(self.denoiser)
            den = self.denoiser[ii](x)
            x = x - merlinth.complex_scale(den, 1 / self.S_end)
            if self.dc[ii] is not None:  # and (self.is_training):  # or (not self.is_training and i < self.S_end)):  # get the output of the non-deterministic (trainable) part for uncertainty estimation, TODO: check if impact!
                x = self.dc[ii]([x,] + constants)

        return x

    def __str__(self):
        return f'VN-S{self.S}-{self.R_type}-{self.D_type}'