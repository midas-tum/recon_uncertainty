import numpy as np

def squeeze_batchdim(inputs):
    if isinstance(inputs, list):
        return [inp.squeeze(0) for inp in inputs]
    else:
        return inputs.squeeze(0)

def to_device(inputs, device):
    if isinstance(inputs, list):
        return [inp.to(device) for inp in inputs]
    else:
        return inputs.to(device)

def prepare_batch(batch, device):
    inputs, outputs = batch
    return to_device(squeeze_batchdim(inputs), device), to_device(squeeze_batchdim(outputs), device)

def np_ensure_complex64(x):
    """
    [Code from https://github.com/khammernik/sigmanet]
    This function is used to recover complex dtype if current dtype is float16,
    otherwise dtype of x is unchanged.

    Args:
        x: Input data of any datatype.
    """
    if x.dtype == np.float16:
        return np.ascontiguousarray(x.astype(np.float32)).view(dtype=np.complex64)
    else:
        return x

def np_ensure_float32(x):
    """
    [Code from https://github.com/khammernik/sigmanet]
    This function is used to recover complex dtype if current dtype is float16,
    otherwise dtype of x is unchanged.

    Args:
        x: Input data of any datatype.
    """
    if x.dtype == np.float16:
        return np.ascontiguousarray(x.astype(np.float32))
    else:
        return x

