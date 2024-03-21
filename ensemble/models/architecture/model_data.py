import torch
import numpy as np

import optoth
import optoth.activations

import ensemble.models.architecture.vis

import merlinth.mytorch as mytorch

class DataCellMRI(torch.nn.Module):
    def __init__(self):
        super(DataCellMRI, self).__init__()

        self.dataOpForward = mytorch.mri.forwardSoftSenseOpNoShift
        self.dataOpAdjoint = mytorch.mri.adjointSoftSenseOpNoShift
        
        self.tol  = 1e-6
        self.iter = 10
        
        self.op = MyCG

    def forward(self, x, y, smaps, mask):
        # compute the data term
        diff  =       self.dataOpForward(x   , smaps, mask) - y
        nabla_D_x   = self.dataOpAdjoint(diff, smaps, mask)
        return nabla_D_x

    def prox(self, x, y, smaps, mask, tau):
        return self.op.apply(x, 1 / tau, y, smaps, mask, self.tol, self.iter)

    def energy(self, x, y, smaps, mask, grad=False):
        # compute the data term
        diff  = self.dataOpForward(x, smaps, mask) - y
        D_x = 0.5 * (diff ** 2).sum()

        if grad:
            nabla_D_x = self.dataOpAdjoint(diff, smaps, mask)
            return D_x, nabla_D_x
        else:
            return D_x

class DataCellComplexDenoise(torch.nn.Module):
    def forward(self, x, y, smaps, mask, get_energy = False, get_grad = True):
        diff = x - y
        if get_energy:
            D_x = 0.5 * (diff ** 2).sum()

        if get_energy and get_grad:
            return D_x, diff
        elif get_energy:
            return D_x
        elif get_grad:
            return diff
        else:
            raise ValueError('Requires to return either energy or grad! (default: grad)')
    
    def prox(self, x, y, smaps, mask, tau):
        return (x + tau * y) / (1 + tau) 

class MyCG(torch.autograd.Function):
    @staticmethod
    def complexDot(data1, data2):
        nBatch = data1.shape[0]
        mult = mytorch.complex.complex_mult_conj(data1, data2)
        re, im = torch.unbind(mult, dim=-1)
        return torch.stack([torch.sum(re.view(nBatch, -1), dim=-1),
                            torch.sum(im.view(nBatch, -1), dim=-1)], -1)

    @staticmethod
    def solve(x0, M, tol, max_iter):
        nBatch = x0.shape[0]
        x = torch.zeros(x0.shape).to(x0.device)
        r = x0.clone()
        p = x0.clone()
        x0x0 = (x0.pow(2)).view(nBatch, -1).sum(-1)
        rr = torch.stack([
            (r.pow(2)).view(nBatch, -1).sum(-1),
            torch.zeros(nBatch).to(x0.device)
        ], dim=-1)

        it = 0
        while torch.min(rr[..., 0] / x0x0) > tol and it < max_iter:
            it += 1
            q = M(p)
            alpha = mytorch.complex.complex_div(rr, MyCG.complexDot(p, q))
            x += mytorch.complex.complex_mult(
                alpha.reshape(nBatch, 1, 1, 1, -1), p.clone())

            r -= mytorch.complex.complex_mult(
                alpha.reshape(nBatch, 1, 1, 1, -1), q.clone())

            rr_new = torch.stack([
                (r.pow(2)).view(nBatch, -1).sum(-1),
                torch.zeros(nBatch).to(x0.device)
            ], dim=-1)

            beta = torch.stack([
                rr_new[..., 0] / rr[..., 0],
                torch.zeros(nBatch).to(x0.device)
            ], dim=-1)

            p = r.clone() + mytorch.complex.complex_mult(
                beta.reshape(nBatch, 1, 1, 1, -1), p)
            rr = rr_new.clone()
        return x

    @staticmethod
    def forward(ctx, z, lambdaa, y, smaps, mask, tol, max_iter):
        ctx.tol = tol
        ctx.max_iter = max_iter

        def A(x):
            return mytorch.mri.forwardSoftSenseOpNoShift(x, smaps, mask)

        def AT(y):
            return mytorch.mri.adjointSoftSenseOpNoShift(y, smaps, mask)

        def M(p):
            return AT(A(p)) + lambdaa * p

        x0 = AT(y) + lambdaa * z
        ctx.save_for_backward(z, x0, smaps, mask, lambdaa)

        return MyCG.solve(x0, M, ctx.tol, ctx.max_iter)

    @staticmethod
    def backward(ctx, grad_x):
        z, rhs, smaps, mask, lambdaa = ctx.saved_tensors

        def A(x):
            return mytorch.mri.forwardSoftSenseOpNoShift(x, smaps, mask)

        def AT(y):
            return mytorch.mri.adjointSoftSenseOpNoShift(y, smaps, mask)

        def M(p):
            return AT(A(p)) + lambdaa * p

        Qe  = MyCG.solve(grad_x, M, ctx.tol, ctx.max_iter)
        QQe = MyCG.solve(Qe,     M, ctx.tol, ctx.max_iter)

        grad_z = lambdaa * Qe

        grad_lambdaa = mytorch.complex.complex_dotp(Qe, z).sum() - \
                       mytorch.complex.complex_dotp(QQe, rhs).sum()

        return grad_z, grad_lambdaa, None, None, None, None, None
