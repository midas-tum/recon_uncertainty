import torch
import re
import yaml
import os

def center_crop(data, shape, channel_last=False):
    """
    [source] https://github.com/facebookresearch/fastMRI/blob/master/data/transforms.py
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (numpy.array): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        numpy.array: The center cropped image
    """
    if channel_last:
        dim0 = data.shape[-3]
        dim1 = data.shape[-2]
    else:
        dim0 = data.shape[-2]
        dim1 = data.shape[-1]

    assert 0 < shape[0] <= dim0
    assert 0 < shape[1] <= dim1
    w_from = (dim0 - shape[0]) // 2
    h_from = (dim1 - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    if channel_last:
        return data[..., w_from:w_to, h_from:h_to, :]
    else:
        return data[..., w_from:w_to, h_from:h_to]

def complex2real(z, channel_last=False):
    stack_dim = -1 if channel_last else 1
    return torch.cat([torch.real(z), torch.imag(z)], stack_dim)

def real2complex(z, channel_last=False):
    stack_dim = -1 if channel_last else 1
    (real, imag) = torch.chunk(z, 2, axis=stack_dim)
    return torch.complex(real, imag)

def loadYaml(cfile, experiment):
    """
    [SOURCE] https://medium.com/swlh/python-yaml-configuration-with-environment-variables-parsing-77930f4273ac
    Load a yaml configuration file and resolve any environment variables
    The environment variables must have !ENV before them and be in this format
    to be parsed: ${VAR_NAME}.
    E.g.:
    database:
        host: !ENV ${HOST}
        port: !ENV ${PORT}
    app:
        log_path: !ENV '/var/${LOG_PATH}'
        something_else: !ENV '${AWESOME_ENV_VAR}/var/${A_SECOND_AWESOME_VAR}'
    :param str path: the path to the yaml file
    :param str data: the yaml data itself as a stream
    :param str tag: the tag to look for
    :return: the dict configuration
    :rtype: dict[str, T]
    """
    # pattern for global vars: look for ${word}
    pattern = re.compile('.*?\${(\w+)}.*?')
    # loader = yaml.SafeLoader  # SafeLoader broken in pyyaml > 5.2, see https://github.com/yaml/pyyaml/issues/266
    loader = yaml.Loader
    tag = "!ENV"

    # the tag will be used to mark where to start searching for the pattern
    # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
    loader.add_implicit_resolver(tag, pattern, None)

    def constructor_env_variables(loader, node):
        """
        Extracts the environment variable from the node's value
        :param yaml.Loader loader: the yaml loader
        :param node: the current node in the yaml
        :return: the parsed string that contains the value of the environment
        variable
        """
        value = loader.construct_scalar(node)
        match = pattern.findall(value)  # to find all env variables in line
        if match:
            full_value = value
            for g in match:
                full_value = full_value.replace(
                    f'${{{g}}}', os.environ.get(g, g)
                )
            return full_value
        return value

    loader.add_constructor(tag, constructor_env_variables)

    with open(cfile, 'r') as f:
        config = yaml.load(f, Loader=loader)
        config = config[experiment]

        var_list = [(k, config[k]) for k in config.keys() if k.startswith('__') and k.endswith('__')]

        for var_key, var_val in var_list:
            del config[var_key]
            
        for var_key, var_val in var_list:
            for key in config.keys():
                if isinstance(config[key], str):
                    config[key] = config[key].replace(var_key, f'{var_val}')

    return config