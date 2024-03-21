import torch
DICOM_OFFSET=0

def root_sum_of_squares(data, dim=0, keepdim=False, eps=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim, keepdim=keepdim) + eps)

def removeFEOversampling(data, axes=(-2,-1), dicom_offset=DICOM_OFFSET):
    """ Remove Frequency Encoding (FE) oversampling.
        This is implemented such that they match with the DICOM images.
    """
    assert data.dim() >= 2

    nFE, nPE = data.shape[axes[0]:axes[1]+1]
    if nPE != nFE:
        indices = torch.arange(int(nFE*0.25)+dicom_offset, int(nFE*0.75)+dicom_offset)
#        print(indices[0],indices[-1])
        if data.device != torch.device("cpu"):
            indices = indices.cuda()
        return data.index_select(axes[0], indices)
    else:
        return data

def addFEOversampling(data, axes=(-2,-1), dicom_offset=DICOM_OFFSET):
    """ Add Frequency Encoding (FE) oversampling.
        This is implemented such that they match with the DICOM images.
    """
    assert data.dim() >= 2

    nFE = data.shape[axes[0]]*2
    pad_u = int(nFE * 0.25 + dicom_offset)
    pad_l = int(nFE * 0.25 - dicom_offset)
    cat_shape_u = list(data.shape)
    cat_shape_l = list(data.shape)
    cat_shape_u[axes[0]] = pad_u
    cat_shape_l[axes[0]] = pad_l

    cat_u = data.new_zeros(*cat_shape_u)
    cat_l = data.new_zeros(*cat_shape_l)

    return torch.cat([cat_u, data, cat_l], dim=axes[0])

def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)
    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero
    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)

def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.
        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero
        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std

def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]

def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.
    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]

def postprocess(tensor, shape):
    """Postprocess the tensor to be magnitude image and crop to the ROI,
    which is (min(nFE, shape[0]), min(nPE, shape[1]). The method expects either a
    tensor representing complex values
    (with shape [bs, nsmaps, nx, ny, 2])
    or a real-valued tensor
    (with shape [bs, nsmaps, nx, ny])
    """
    if tensor.shape[-1] == 2:
        tensor = root_sum_of_squares(tensor, dim=(1, -1), eps=1e-9)
    cropsize = (min(tensor.shape[-2], shape[0]), min(tensor.shape[-1], shape[1]))
    return center_crop(tensor, cropsize)