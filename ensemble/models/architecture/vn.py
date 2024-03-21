import torch
import numpy as np

import optoth.activations

import ensemble.models.architecture.vis

from ensemble.models.architecture.model_data import *
from ensemble.models.architecture.model_reg import *
import torch.utils.checkpoint as cp

class VNet(torch.nn.Module):
    """
    Variational Network
    """

    def __init__(self, config, efficient=False, dynamic=False):
        super(VNet, self).__init__()

        self.efficient = efficient
        self.dynamic = dynamic
       
        if self.dynamic:
            self.S = config['S']
        else:
            self.S = 1
        self.S_end = config['S']

        # setup the stopping time
        self.T = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.reset_scalar(self.T, **config["T"])

        # setup the modules
        self.steps = torch.nn.ModuleList([VNCell(config) for _ in range(self.S)])

        self.use_prox = config['D_prox']

        # setup the network parameter visualization
        self.vis = False
    
    def reset_scalar(self, scalar, init=1., min=0, max=1000, requires_grad=True):
        scalar.data = torch.tensor(init, dtype=scalar.dtype)
        # add a positivity constraint
        scalar.proj = lambda: scalar.data.clamp_(min, max)
        scalar.requires_grad = requires_grad

    def forward(self, x, y, smaps, mask):
        x_all = [x]

        #print(self.steps[0].lambdaa.detach().cpu().numpy(), self.T.detach().cpu().numpy())
        for s in range(self.S_end):
            # compute a single step
            tau = self.T / self.S_end
            step = self.steps[s % len(self.steps)]

            def current_step(x):
                if self.efficient and (x.requires_grad or y.requires_grad):
                    x_dot = cp.checkpoint(step, x, y, smaps, mask)
                else:
                    x_dot = step(x, y, smaps, mask)
                return x_dot

            x = x - tau * current_step(x)

            if self.use_prox:
                x = step.prox(x, y, smaps, mask, 1 / self.S_end)

            x_all.append(x)
        
        return x_all

    def set_end(self, s):
        assert 0 < s
        if self.dynamic:
            assert s <= self.S
        self.S_end = s

    def extra_repr(self):
        s = "S={S} prox={use_prox}"
        return s.format(**self.__dict__)

    def clear_hist(self):
        for s in self.steps:
            s.R.clear_hist()

    def compute_hist(self, v):
        assert isinstance(v, bool)
        for s in self.steps:
            s.R.compute_hist = v

    def setup_vis(self, vis_path):
        self.vis_f1 = vis.KernelFunctionSequence(vis_path, 'f1')
        self.vis_T = vis.LinePlot(vis_path, 'T', ['0'])

        #if self.use_prox:
        self.vis_lambda = vis.LinePlot(vis_path, 'lambda', ['{:d}'.format(i) for i in range(self.S)])

        self.vis_hists = vis.HistSequence(vis_path, 'hist_weights')
        self.vis_hist_act = vis.HistSequence(vis_path, 'activation')

        self.vis = True

    def add_scalar_to_save(self, epoch):
        assert self.vis
        self.vis_T.add(epoch, self.T.detach().cpu().numpy(), '0')

        #if self.use_prox:
        for s in range(self.S):
            self.vis_lambda.add(epoch, self.steps[s].get_lambda().data.detach().cpu().numpy(), '{}'.format(s))


    def save(self, epoch):
        assert self.vis
        pairs = []

        # plot the kernels and the functions
        for s in range(self.S):
            pairs.append(self.steps[s].get_vis())
            
        self.vis_f1.save(epoch, pairs)
        self.vis_T.save()
        #if self.use_prox:
        self.vis_lambda.save()

        # plot the weight histograms 
        histograms = []
        for s in range(self.S):
            theta = self.steps[s].get_theta()
            hist_theta = {k: np.histogram(v.detach().cpu().numpy().ravel(), bins=101)
                          for k, v in theta}
            histograms.append(hist_theta)

        self.vis_hists.save(epoch, histograms)


class VNCell(torch.nn.Module):
    """
    Basic cell of a Variational Network
    """

    def __init__(self, config):
        super(VNCell, self).__init__()

        self.config = config
        self.use_prox = config['D_prox']

        # setup the regularization module
        if config['R_type'] == 'foe':
            self.R = FoERegularization(config['R_config'])
        elif config['R_type'] == 'idub':
            self.R = IDUB(config['R_config'])
        else:
            raise RuntimeError('Invalid regularization type!')

        # setup the data term operator
        if config['D_type'] == 'mri':
            self.D = DataCellMRI()
        elif config['D_type'] == 'complex_denoise':
            self.D = DataCellComplexDenoise()
        else:
            print('Invalid data consistency type!')

        self.lambdaa = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.reset_scalar(self.lambdaa, **config["lambda"])
            
    def forward(self, x, y, smaps, mask):
        if self.use_prox:
            return  self.R(x)
        else:
            lambdaa = self.lambdaa * 10
            return (self.R(x) + lambdaa * self.D(x, y, smaps, mask))

    def prox(self, x, y, smaps, mask, tau):
        if self.use_prox:
            lambdaa = self.lambdaa * 100
            return self.D.prox(x, y, smaps, mask, tau = lambdaa * tau)
        else:
            return x

    def reset_scalar(self, scalar, init=1., min=0, max=1000, requires_grad=True):
        scalar.data = torch.tensor(init, dtype=scalar.dtype)
        # add a positivity constraint
        scalar.proj = lambda: scalar.data.clamp_(min, max)
        scalar.requires_grad = requires_grad

    def get_theta(self):
        return self.R.get_theta()
    
    def get_lambda(self):
        return self.lambdaa

    def get_vis(self):
        return self.R.get_vis()
