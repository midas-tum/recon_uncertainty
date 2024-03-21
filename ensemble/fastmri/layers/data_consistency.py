import torch
from fastmri.layers.fft import fft2, fft2c, ifft2, ifft2c

class SingleCoilProxLayer(torch.nn.Module):
    """
        Data Consistency layer from DC-CNN, apply for single coil mainly
    """
    def __init__(self, weight_init=0.1, weight_scale=1.0, center_fft=True, trainable=True):
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

        self.fft = fft2c if center_fft else fft2
        self.ifft = ifft2c if center_fft else ifft2

    @property
    def weight(self):
        return self._weight * self.weight_scale

    def forward(self, x, y, mask):
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
