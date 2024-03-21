import torch
import numpy as np

import optoth
import optoth.activations

import ensemble.models.architecture.vis

# import common.mytorch as mytorch
# from common.mytorch.complex import ComplexInstanceNorm
#import myact

class DiederConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, invariant=False,
                 stride=1, dilation=1, groups=1, bias=False, 
                 padding_mode='reflect', zero_mean=False, bound_norm=False, complex=False):
        super(DiederConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.invariant = invariant
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.bias = torch.nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.padding_mode = padding_mode
        self.zero_mean = zero_mean
        self.bound_norm = bound_norm
        self.padding = 0
        self.complex = complex

        # add the parameter
        if self.invariant:
            assert self.kernel_size == 3
            self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, 1,  3))
            self.register_buffer('mask', torch.from_numpy(np.asarray([1,4,4], dtype=np.float32)[None, None, None, :]))
        else:
            self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size))
            self.register_buffer('mask', torch.from_numpy(np.ones((self.kernel_size, self.kernel_size), dtype=np.float32)[None, None, :, :]))
        # insert them using a normal distribution
        torch.nn.init.normal_(self.weight.data, 0.0, np.sqrt(1/np.prod(in_channels*kernel_size**2)))

        # specify reduction index
        if zero_mean or bound_norm:
            self.weight.reduction_dim = (1, 2, 3)
            self.weight.reduction_dim_mean = (2, 3) if self.complex else (1, 2, 3)

            # define a projection
            def l2_proj(surface=False):
                # reduce the mean
                if zero_mean:
                    mean = torch.sum(self.weight.data * self.mask, self.weight.reduction_dim_mean, True) / (self.in_channels*self.kernel_size**2)
                    self.weight.data.sub_(mean)
                # normalize by the l2-norm
                if bound_norm:
                    norm = torch.sum(self.weight.data**2 * self.mask, self.weight.reduction_dim, True).sqrt_()
                    if surface:
                        self.weight.data.div_(norm)
                    else:
                        self.weight.data.div_(
                            torch.max(norm, torch.ones_like(norm)))
            self.weight.proj = l2_proj

            # initially call the projection
            self.weight.proj(True)

    def get_weight(self):
        if self.invariant:
            weight = torch.empty(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, device=self.weight.device)
            weight[:,:,1,1] = self.weight[:,:,0,0]
            weight[:,:,::2,::2] = self.weight[:,:,0,2].view(self.out_channels,self.in_channels,1,1)
            weight[:,:,1::2,::2] = self.weight[:,:,0,1].view(self.out_channels,self.in_channels,1,1)
            weight[:,:,::2,1::2] = self.weight[:,:,0,1].view(self.out_channels,self.in_channels,1,1)
        else:
            weight = self.weight
        # print(self.weight)
        # print(weight)
        # print('mean:', torch.mean(weight, (1,2,3)))
        # print('norm:', torch.sum(weight**2, (1,2,3)).sqrt_())
        return weight

    def forward(self, x):
        # construct the kernel
        weight = self.get_weight()
        # then pad
        pad = weight.shape[-1]//2
        if pad > 0:
            x = torch.nn.functional.pad(x, (pad,pad,pad,pad), self.padding_mode)
        # compute the convolution
        return torch.nn.functional.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = "({out_channels}, {in_channels}, {kernel_size}), invariant={invariant}"
        if self.stride != 1:
            s += ", stride={stride}"
        if self.dilation != 1:
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if not self.bias is None:
            s += ", bias=True"
        if self.zero_mean:
            s += ", zero_mean={zero_mean}"
        if self.bound_norm:
            s += ", bound_norm={bound_norm}"
        return s.format(**self.__dict__)


class DiederConvScale2d(DiederConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, invariant,
                 groups=1, bias=False, 
                 padding_mode='reflect', zero_mean=False, bound_norm=False):
        super(DiederConvScale2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            invariant=invariant, stride=2, dilation=1, groups=groups, bias=bias, 
            padding_mode=padding_mode, zero_mean=zero_mean, bound_norm=bound_norm)

        # create the convolution kernel
        np_k = np.asarray([1, 4, 6, 4, 1], dtype=np.float32)[:, np.newaxis]
        np_k = np_k @ np_k.T
        np_k /= np_k.sum()
        np_k = np.reshape(np_k, (1, 1, 5, 5))
        self.register_buffer('blur', torch.from_numpy(np_k))

    def get_weight(self):
        weight = super().get_weight()
        weight = weight.reshape(-1, 1, self.kernel_size, self.kernel_size)
        weight = torch.nn.functional.conv2d(weight, self.blur, padding=4)
        weight = weight.reshape(self.out_channels, self.in_channels, self.kernel_size+4, self.kernel_size+4)
        return weight


class DiederConvScaleTranspose2d(DiederConvScale2d):
    
    def __init__(self, in_channels, out_channels, kernel_size, invariant,
                 groups=1, bias=False, padding=0, output_padding=0,
                 padding_mode='zeros', zero_mean=False, bound_norm=False):
        super(DiederConvScaleTranspose2d, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
            invariant=invariant, groups=groups, bias=bias, 
            padding_mode=padding_mode, zero_mean=zero_mean, bound_norm=bound_norm)

        self.padding = padding+2
        self.output_padding = output_padding

    def forward(self, x, output_shape):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        self.output_padding = (1 - output_shape[2]%2, 1 - output_shape[3]%2)

        # get the blurred weight
        weight = self.get_weight()
        # conv
        out = torch.nn.functional.conv_transpose2d(
            x, weight.permute(1,0,2,3), self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)
        return out

    def extra_repr(self):
        s = ""
        if self.padding != 0:
            s += ", padding={padding}"
        # if self.output_padding != 0:
        #     s += ", output_padding={output_padding}"
        return super().extra_repr() + s.format(**self.__dict__)


class Regularizer(torch.nn.Module):
    """
    Basic regularization function
    """

    def __init__(self):
        super(Regularizer, self).__init__()

        self.compute_hist = False
        self.clear_hist()

    def forward(self, x):
        # first compute the (non-)linear transformation
        with torch.enable_grad():
            x.requires_grad_(True)
            x_ = self.transformation(x)
        # apply the gradient of the potential
        if self.compute_hist:
            self.add_hist('activation', x_)
        f = self.activation(x_)
        # apply the gradient of the (non-)linear transformation
        grad = torch.autograd.grad(x_, x, f,
                                    create_graph=self.training, retain_graph=self.training, only_inputs=True)[0]
        return grad

    def transformation(self, x):
        """
        (non-)linear transformation of the input
        """
        raise NotImplementedError

    def activation(self, x):
        """
        activation function
        """
        raise NotImplementedError

    def clear_hist(self):
        self.hist = {'activation': []}

    def get_hist(self):
        return self.hist

    def add_hist(self, name, th_tensor):
        if self.compute_hist and self.training:
            np_tensor = th_tensor.detach().cpu().numpy().transpose(1, 0, 2, 3).reshape(th_tensor.shape[1], -1)
            hist = []
            for i in range(np_tensor.shape[0]):
                    hist.append(np.histogram(np_tensor[i], bins=101))
            self.hist['activation'].append(hist)

        return th_tensor

    def get_theta(self):
        """
        return all parameters of the regularization
        """
        return self.named_parameters()

    def get_vis(self):
        """
        return all parameters that are visualized
        """
        raise NotImplementedError

class ComplexRegularizer(Regularizer):
    """
    Basic regularization function
    """

    def __init__(self, reg_dtype):
        super(ComplexRegularizer, self).__init__()
        # todo define real, real2, complex_sum
        self.compute_hist = False
        self.clear_hist()
        self.reg_dtype = reg_dtype

        if reg_dtype == 'real2':
            self.forward = self._forward_real2
        elif reg_dtype == 'complex':
            self.forward = self._forward_complex

    def _forward_real2(self, x, get_transformation=False):
        x_r = x

        nBatch, nSmaps, nFE, nPE, nCh = x_r.shape
        x_r = x_r.view(nBatch * nSmaps, nFE, nPE, nCh).permute(0, 3, 1, 2)
        x_r = x_r.reshape(nBatch * nSmaps * nCh, 1, nFE, nPE)

        with torch.enable_grad():
            x_r.requires_grad_(True)
            #print('after', torch.max(x_r), torch.min(x_r))
            x_ = self.transformation(x_r)

            if get_transformation:
                grad_R = x_
            else:
                # apply the gradient of the potential
                if self.compute_hist:
                    self.add_hist('activation', x_)
                f = self.activation(x_)
                # apply the gradient of the (non-)linear transformation
                grad_R = torch.autograd.grad(x_, x_r, f,
                                            create_graph=self.training, retain_graph=self.training, only_inputs=True)[0]

        grad_R = grad_R.reshape(nBatch * nSmaps, nCh, nFE, nPE)
        grad_R = grad_R.permute(0, 2, 3, 1)
        #grad_R = mytorch.complex.complex_abs(grad_R, dim=-1, keepdim=True, eps=(1e-9)**2)
        grad_R = torch.sum(grad_R, dim=-1, keepdim=True)
        grad_R = grad_R.view(nBatch, nSmaps, nFE, nPE, 1)

        return grad_R

    def _forward_complex(self, x, get_transformation=False):
        x_r = x

        shape = x_r.shape
        x_r = x_r.view(shape[0]*shape[1], *shape[2:])
        if len(x_r.shape) == 3:  # single batch dim
            x_r = torch.unsqueeze(x_r, 0)
        x_r = x_r.permute((0, 3, 1, 2)) # [N, H, W, 2] -> [N, 2, H, W]

        with torch.enable_grad():
            x_r.requires_grad_(True)
            #print('after', torch.max(x_r), torch.min(x_r))
            x_ = self.transformation(x_r)

            if get_transformation:
                grad_R = x_
            else:
                # apply the gradient of the potential
                if self.compute_hist:
                    self.add_hist('activation', x_)
                f = self.activation(x_)
                # apply the gradient of the (non-)linear transformation
                grad_R = torch.autograd.grad(x_, x_r, f,
                                            create_graph=self.training, retain_graph=self.training, only_inputs=True)[0]

        grad_R = grad_R.permute(0, 2, 3, 1).view(*shape)
        return grad_R

    def energy(self, x, mask=None, grad=False):
        if grad:
            # regularization energy
            with torch.enable_grad():
                x_ = x.clone().requires_grad_(True)
                R_x = self.forward(x_, get_transformation=True)
                if mask is not None:
                    R_x = R_x * mask
                import medutils
                debug = R_x.detach()[0,0,...,0].cpu().numpy()
                medutils.visualization.imsave(debug, 'reg_energy.png')
                R_x = R_x.sum()
            # regularization gradient
            nabla_R_x = torch.autograd.grad(R_x, x_, create_graph=False, retain_graph=False, only_inputs=True)[0]
            return R_x, nabla_R_x
        else:
            # regularization energy
            R_x = self.forward(x, get_transformation=True)
            if mask is not None:
                R_x = R_x * mask
            R_x = R_x.sum()
            return R_x

    def transformation(self, x):
        """
        (non-)linear transformation of the input
        """
        raise NotImplementedError

    def activation(self, x):
        """
        activation function
        """
        raise NotImplementedError

    def clear_hist(self):
        self.hist = {'activation': []}

    def get_hist(self):
        return self.hist

    def add_hist(self, name, th_tensor):
        if self.compute_hist and self.training:
            np_tensor = th_tensor.detach().cpu().numpy().transpose(1, 0, 2, 3).reshape(th_tensor.shape[1], -1)
            hist = []
            for i in range(np_tensor.shape[0]):
                    hist.append(np.histogram(np_tensor[i], bins=101))
            self.hist['activation'].append(hist)

        return th_tensor


    def get_theta(self):
        """
        return all parameters of the regularization
        """
        return self.named_parameters()

    def get_vis(self):
        """
        return all parameters that are visualized
        """
        raise NotImplementedError

class FoERegularization(ComplexRegularizer):
    """
    Field of experts regularization functional
    """

    def __init__(self, config):
        super(FoERegularization, self).__init__(reg_dtype = config['dtype'])

        # setup the modules
        self.K1 = DiederConv2d(**config["K1"])
        self.f1 = optoth.activations.TrainableActivation(**config["f1"])

    def transformation(self, x):
        return self.K1(x)

    def activation(self, x):
        return self.f1(x) / x.shape[1]

    def get_vis(self):
        kernels = {k: v for k, v in self.named_parameters() if 'K1.weight' in k}
        return kernels, self.f1.draw()
