import torch
#from merlinth.losses.loss_template import define_losses
from merlinth.losses.ssim import SSIM
import numpy as np

def get_loss(loss):
    return eval(loss)

def complex_abs(z, eps=1e-9):
    if torch.is_complex(z):
        return torch.sqrt(torch.real(torch.conj(z) * z) + eps)
    else:
        return torch.abs(z)

# assuming: batch x channel x X x Y

def loss_complex_mse(y_true, y_pred):
    diff = (y_true - y_pred)
    if torch.is_complex(diff):
        return torch.mean(torch.sum(torch.real(torch.conj(diff) * diff), axis=(2,3)), axis=(0,1))
    else:
        return torch.mean(torch.sum(torch.square(diff), axis=(2, 3)), axis=(0, 1))

def loss_abs_mse(y_true, y_pred):
    diff = (complex_abs(y_true) - complex_abs(y_pred))
    if torch.is_complex(diff):
        return torch.mean(torch.sum(torch.real(torch.conj(diff) * diff), axis=(2,3)), axis=(0,1))
    else:
        return torch.mean(torch.sum(torch.square(diff), axis=(2,3)), axis=(0,1))

def loss_complex_rmse(y_true, y_pred):
    diff = (y_true - y_pred)
    if torch.is_complex(diff):
        return torch.sqrt(torch.mean(torch.sum(torch.real(torch.conj(diff) * diff), axis=(2,3)), axis=(0,1)))
    else:
        return torch.sqrt(torch.mean(torch.sum(torch.square(diff), axis=(2, 3)), axis=(0, 1)))

def loss_complex_mae(y_true, y_pred):
    diff = (y_true - y_pred)
    if torch.is_complex(diff):
        return torch.mean(torch.sum(torch.sqrt(torch.real(torch.conj(diff) * diff) + 1e-9), axis=(2,3)), axis=(0,1))
    else:
        return torch.mean(torch.sum(torch.sqrt(torch.real(torch.square(diff)) + 1e-9), axis=(2, 3)), axis=(0, 1))

def loss_abs_mae(y_true, y_pred):
    diff = (complex_abs(y_true) - complex_abs(y_pred))
    if torch.is_complex(diff):
        return torch.mean(torch.sum(torch.sqrt(torch.real(torch.conj(diff) * diff) + 1e-9), axis=(2,3)), axis=(0,1))
    else:
        return torch.mean(torch.sum(torch.sqrt(torch.real(torch.square(diff)) + 1e-9), axis=(2, 3)), axis=(0, 1))

def loss_l2_heteroscedastic(y_true, mu, log_sigma):
    #mu = y_pred[0]
    #log_sigma = y_pred[1]
    var_eps = 1e-5
    bias_eps = 100
    if torch.is_complex(mu):
        log_sigma = complex_abs(log_sigma)
    sigma = torch.exp(log_sigma) + var_eps

    diff = complex_abs(y_true - mu)
    factor = 1 / sigma
    diffq = factor * torch.pow(diff, 2)
    loss1 = torch.mean(diffq)
    loss2_old = torch.mean(torch.log(sigma))
    #loss2 = torch.logsumexp(log_sigma, tuple(range(log_sigma.ndim)))/log_sigma.size(dim=0)
    loss2 = torch.mean(log_sigma)
    loss = 0.5 * (loss1 + loss2)

    '''
    if torch.is_complex(mu):
        diff = (y_true - mu)
        complex_nll = 0.5 * (torch.real(((torch.exp(torch.conj(log_sigma) * log_sigma) + var_eps) ** (-1)) * (torch.conj(diff) * diff)) + torch.real(torch.conj(torch.log(sigma)) * torch.log(sigma)))
    else:
        diff = complex_abs(y_true - mu)
        complex_nll = 0.5 * (torch.div(torch.square(diff), torch.square(torch.exp(log_sigma) + var_eps)) + torch.log(sigma))
        #complex_nll = 0.5 * ((torch.exp(log_sigma) ** (-2)) * torch.square(diff) + log_sigma)
    loss = torch.mean(torch.sum(complex_nll, axis=(2,3)), axis=(0,1))
    '''
    return loss

def loss_sure(y_true, y_pred, y_predptb, b_prime, eps, sigma_in):
    batch = y_true.size()[0]
    var_w = torch.square(sigma_in)[:, None, None, None].to('cuda')
    if torch.is_complex(y_pred):
        divergence = torch.multiply((1.0 / eps), torch.multiply(b_prime, torch.abs(y_predptb - y_pred)))
    else:
        divergence = torch.multiply((1.0 / eps), torch.multiply(b_prime, (y_predptb - y_pred)))
    divergence_sum = 2.0/batch * torch.sum(torch.multiply(var_w, divergence))
    var_w_sum = torch.sum(var_w)/2.0 * 3 * np.prod(y_true.size()[1:])

    loss = (1.0 / batch) * loss_abs_mse(y_true, y_pred) - var_w_sum + divergence_sum

    return loss

def loss_vn(target, output, **kwargs):
    _ssim = SSIM(device='cuda')
    def l1(x, xgt):
        tmp = abs(x - xgt)
        loss = tmp.sum()
        return loss / xgt.size()[0]

    def ssim(x, xgt):
        SSIM_SCALE = 1  # 100
        batchsize, nCha, nFE, nPE = xgt.size()
        dynamic_range = 1 # sample['attrs']['ref_max'].cuda()
        _, ssimmap = _ssim(
            xgt.view(batchsize, 1, nFE, nPE),
            x.view(batchsize, 1, nFE, nPE),
            data_range=dynamic_range, full=True,
        )

        # only take the mean over the foreground
        ssimmap = ssimmap.view(batchsize, -1)
        #ssim_val = (ssimmap).sum(-1, keepdim=True) # / mask_norm
        ssim_val = (ssimmap).mean(-1)
        return (1 - ssim_val.mean()) * SSIM_SCALE

    scale = kwargs.pop('scale', 1.)
    loss_l1 = l1(output, target)
    loss_ssim = ssim(output, target)

    loss = loss_ssim + loss_l1 * 1e-3
    loss /= scale
    return loss


def build_loss_vn(args):
    _ssim = SSIM(device='cuda')

    def l1(x, xgt):
        tmp = abs(x - xgt)
        loss = tmp.sum()
        return loss / xgt.size()[0]

    def ssim(x, xgt):
        SSIM_SCALE = 1  # 100
        batchsize, nFE, nPE = xgt.size()
        dynamic_range = 1 # sample['attrs']['ref_max'].cuda()
        _, ssimmap = _ssim(
            xgt.view(batchsize, 1, nFE, nPE),
            x.view(batchsize, 1, nFE, nPE),
            data_range=dynamic_range, full=True,
        )

        # only take the mean over the foreground
        ssimmap = ssimmap.view(batchsize, -1)
        #mask = mask.contiguous().view(batchsize, -1)
        #mask_norm = mask.sum(-1, keepdim=True)
        #mask_norm = torch.max(mask_norm, torch.ones_like(mask_norm))
        ssim_val = (ssimmap).sum(-1, keepdim=True) # / mask_norm
        return (1 - ssim_val.mean()) * SSIM_SCALE


    def criterion(output, target, **kwargs):
        scale = kwargs.pop('scale', 1.)
        loss_l1 = l1(output, target)
        loss_ssim = ssim(output, target)

        loss = loss_ssim + loss_l1 * 1e-3
        #loss = loss_l1
        loss /= scale

        return loss, loss_l1, loss_ssim

    return criterion