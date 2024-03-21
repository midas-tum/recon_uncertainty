import torch
import torch.nn as nn
import merlinth
from ensemble.models.architecture.unet import UNet
from ensemble.models.architecture.common import Scalar
#from merlinth.models.cnn import Real2chCNN
from ensemble.models.architecture.cnn import Real2chCNN
from ensemble.models.architecture.modl import MoDL_2D
import random
import numpy as np
import copy

def get_denoiser(config):
    R_type = config['R_type']

    if R_type == 'real2ch_unet':
        return UNet(**config['R_config'])
    elif R_type == 'mag_unet':
        return UNet(**config['R_config'])
    elif R_type == 'real2ch_foe':
        return merlinth.models.foe.Real2chFoE(config['R_config'])
    elif R_type == 'mag_foe':
        return merlinth.models.foe.MagnitudeFoE(config['R_config'])
    elif R_type == 'real2ch_cnn':
        return Real2chCNN(**config['R_config'])
    elif R_type == 'real2ch_modl':
        return MoDL_2D(**config['R_config'])
    else:
        raise ValueError(f'R_type {R_type} not defined!')

def get_dc(config, multicoil=False):
    D_type = config['D_type']
    if D_type == 'none':   # denoising only network
        return None

    if multicoil:
        A = merlinth.layers.mri.MulticoilForwardOp(**config['A_config'])
        AH = merlinth.layers.mri.MulticoilAdjointOp(**config['A_config'])
    else:  # single-coil
        A = merlinth.layers.mri.ForwardOp(**config['A_config'])
        AH = merlinth.layers.mri.AdjointOp(**config['A_config'])

    if D_type == 'gd':
        return merlinth.layers.data_consistency.DCGD(A, AH, **config['D_config'])
    elif D_type == 'pm':
        return merlinth.layers.data_consistency.DCPM(A, AH, **config['D_config'])
    elif D_type == 'none':
        return None
    else:
        raise ValueError(f'D_type {D_type} not defined!')

class UnrolledNetwork(nn.Module):
    def __init__(self, config, is_training=True, multicoil=False):
        super().__init__()

        self.S = config['S'] if config['dynamic'] else 1
        self.S_end = config['S']
        self.is_training = is_training
        self.parameter_estimation = config['parameter_estimation']
        self.multicoil = multicoil

        self.tau = Scalar(name='tau',
                        scale=config['tau']['scale'],
                        trainable=config['tau']['trainable'],
                        constraint=(config['tau']['min'], config['tau']['max']),
                        initializer=config['tau']['init'])

        self.softplus = nn.Softplus()
        if self.parameter_estimation:
            if self.S > 1:
                config_firsts = copy.deepcopy(config)
                if config['R_type'] == 'real2ch_unet' or config['R_type'] == 'mag_unet' or config['R_type'] == 'real2ch_modl':
                    n_channels = list(config['R_config']['n_channels'])
                    config_firsts['R_config']['n_channels'] = (n_channels[0], int(n_channels[1]/2))
                elif config['R_type'] == 'real2ch_cnn':
                    config_firsts['R_config']['output_dim'] = int(config['R_config']['output_dim']/2)
                self.denoiser = nn.ModuleList([get_denoiser(config_firsts) for _ in range(self.S - 1)] + [get_denoiser(config)])
            else:
                self.denoiser = nn.ModuleList([get_denoiser(config) for _ in range(self.S)])
        else:
            self.denoiser = nn.ModuleList([get_denoiser(config) for _ in range(self.S)])

        if config['D_type'] == 'none':
            self.dc = [None for _ in range(self.S_end)]
        else:
            self.dc = nn.ModuleList([get_dc(config, multicoil) for _ in range(self.S_end)])

        self.stop_gradient = config['stop_gradient'] if 'stop_gradient' in config else 0

        if config['R_type'] == 'mag_unet' or config['R_type'] == 'mag_foe':
            self.type = 'mag'
        elif config['R_type'] == 'real2ch_unet' or config['R_type'] == 'real2ch_foe' or config['R_type'] == 'real2ch_cnn' or config['R_type'] == 'real2ch_modl':
            self.type = 'real2ch'
        
    def forward(self, inputs):  # inputs: input_subsampled, input_kspace, mask, smaps, input_reference
        x = inputs[0]
        constants = list(inputs[1:])

        # reset states for new inputs
        for i in range(self.S_end):
            ii = i % len(self.denoiser)  # for weight sharing
            if hasattr(self.denoiser[ii], 'reset_states'):
                self.denoiser[ii].reset_states()

        for i in range(self.S_end):
            ii = i % len(self.denoiser)
            den = self.denoiser[ii](x)

            if self.parameter_estimation and ii == self.S_end-1:
                if self.type == 'mag':
                    sigma = den[:, [1], ...]  # preserve dimension
                    den = den[:, [0], ...]
                else:
                    sigma = den[:, 2:4, ...]
                    den = den[:, 0:2, ...]

            #if i < self.stop_gradient and not self.magnitude:
            #    den = tf.stop_gradient(den)
            #    self.denoiser[ii].stop_gradients()

            if self.type == 'mag':
                x = x - self.tau(den)
                x = torch.maximum(x,  torch.zeros_like(den))
            else:
                x = x - (self.tau(den) * 1 / self.S_end)
                x = merlinth.real2complex(x)

            if self.dc[ii] is not None:  # and (self.is_training):  # or (not self.is_training and ii < self.S_end-1)):  # get the output of the non-deterministic (trainable) part for uncertainty estimation, TODO: check if impact!
                x = self.dc[ii]([torch.squeeze(x, -1)] + constants, scale=1/self.S_end)
            if self.type == 'real2ch':
                x = merlinth.complex2real(x)

            #if i < self.stop_gradient and not self.magnitude:
            #    x = tf.stop_gradient(x)
        if self.parameter_estimation:
            sigma = self.softplus(sigma)
            return x, sigma
        else:
            return x

if __name__ == '__main__':
    # debug
    # python3 recon_model.py --config ../../config/unet.yml --experiment unet_mag_denoiser

    import merlinpy
    args = merlinpy.Experiment()
    args.parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = f'UnrolledNet-{args.experiment}'
    model_recon = UnrolledNetwork(args.config['model']).cuda()

    nBatch = 1
    M = 112
    N = 96
    C = 1
    imgsub = torch.randn(nBatch, C, M, N).cuda()
    kspace = torch.randn(nBatch, C, M, N).cuda()
    mask = torch.randint(0, 1, (nBatch, C, M, N)).cuda()
    smaps = torch.randn(nBatch, C, M, N).cuda()
    img_ref = torch.randn(nBatch, C, M, N).cuda()

    y = model_recon([imgsub, kspace, mask, smaps, img_ref])
