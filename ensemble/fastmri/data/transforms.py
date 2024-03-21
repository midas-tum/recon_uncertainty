import numpy as np
import medutils
import h5py
import os
import fastmri.data.utils
import fastmri.utils

class Transpose():
    def __init__(self, transpose_list):
        self.transpose_list = transpose_list

    def __call__(self, sample):
        for key, axes in self.transpose_list:
            sample[key] = np.ascontiguousarray(np.transpose(sample[key], axes))
        return sample

class CenterCrop():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = fastmri.utils.center_crop(sample[key], (sample['attrs']['metadata']['rec_y'], sample['attrs']['metadata']['rec_x']), channel_last=True)
        return sample   

class ToTorchIO():
    ''' Move to merlin!'''
    def __init__(self, input_keys, output_keys):
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, sample):
        import torch
        inputs = []
        outputs = []
        for key in self.input_keys:
            inputs.append(torch.from_numpy(sample[key]))
        for key in self.output_keys:
            outputs.append(torch.from_numpy(sample[key]))
        return inputs, outputs

class Magnitude():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            sample[key] = np.abs(sample[key])
        return sample

class GeneratePatches():
    def __init__(self, patch_ny, offset_y, remove_feos=False, multicoil=True):
        self.patch_ny = patch_ny
        self.offset_y = offset_y
        self.remove_feos = remove_feos
        self.multicoil = multicoil

    def __call__(self, sample):
        np_kspace = sample['kspace']

        if 'patch_ny' in sample['attrs'].keys():
            patch_ny = sample['attrs']['patch_ny']
            offset_y = sample['attrs']['offset_y']
        else:
            patch_ny = self.patch_ny
            offset_y = self.offset_y

        np_mask = sample['mask'][..., 0:patch_ny, :]

        # remove FE Oversampling
        if self.remove_feos:
            k_offset = np_kspace.shape[-2] // 4
            k_fe = np_kspace.shape[-2] // 2
        else:
            k_offset = 0
            k_fe = np_kspace.shape[-2]

        # extract patch in Ny direction
        max_ny = k_fe - patch_ny - offset_y + 1

        shape_reduced = list(np_kspace.shape)
        shape_reduced[-2] = patch_ny
        np_kspace_reduced = np.zeros(shape_reduced, dtype=np.complex64)

        is_fg_mask = True if 'fg_mask' in sample else False
        if self.multicoil:
            np_smaps = sample['smaps']
            smaps_shape_reduced = list(np_smaps.shape)
            smaps_shape_reduced[-2] = patch_ny
            np_smaps_reduced = np.zeros(smaps_shape_reduced, dtype=np.complex64)
        if is_fg_mask:
            np_fg_mask = sample['fg_mask']
            fg_mask_shape_reduced = list(np_fg_mask.shape)
            fg_mask_shape_reduced[-2] = patch_ny
            np_fg_mask_reduced = np.zeros(fg_mask_shape_reduced, dtype=np.float32)

        for i in range(np_kspace.shape[0]):
            start_idx = np.random.randint(offset_y, max_ny) + k_offset
            start, end = start_idx, start_idx + patch_ny

            #TODO add padl padr here!
            # coil-wise ifft of kspace, then patch extraction and coil-wise fft
            np_img = medutils.mri.ifft2c(np_kspace[i])
            # if self.remove_feos:
            #     np_img = medutils.mri.removeFEOversampling(np_img)
            np_kspace_reduced[i] = medutils.mri.fft2c(np_img[..., start:end, :])

            if self.multicoil:
                np_smaps_reduced[i] = np_smaps[i, ..., start:end, :]

            if is_fg_mask:
                np_fg_mask_reduced[i] = np_fg_mask[i, ..., start:end, :]

        # create new adjoint
        if self.multicoil:  
            np_target_reduced = medutils.mri.mriAdjointOp(
                np_kspace_reduced,
                smaps=np_smaps_reduced,
                mask=np.ones_like(np_mask),
                coil_axis=(1),
            )
        else:
            np_target_reduced = medutils.mri.ifft2c(np_kspace_reduced)

        sample['reference'] = np_target_reduced
        sample['kspace'] = np_kspace_reduced
        sample['mask'] = np_mask
        if self.multicoil:
            sample['smaps'] = np_smaps_reduced
        if is_fg_mask:
            sample['fg_mask'] = np_fg_mask_reduced

        return sample

#TODO SetupRandomFastMRIChallengeMask
#TODO GenerateCartesianMask:
class GenerateRandomFastMRIChallengeMask:
    """
    [Code from https://github.com/facebookresearch/fastMRI]

    GenerateRandomFastMRIChallengeMask creates a sub-sampling mask of
    a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies
        2. The other columns are selected uniformly at random with a
           probability equal to:
           prob = (N / acceleration - N_low_freqs) / (N - N_low_freqs).
    This ensures that the expected number of columns selected is
    equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the MaskFunc object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def __init__(
        self, center_fractions, accelerations,
        is_train=True, seed=None,
    ):
        """
        Args:
            center_fractions (List[float]): Fraction of low-frequency columns
                to be retained. If multiple values are provided, then one of
                these numbers is chosen uniformly each time.

            accelerations (List[int]): Amount of under-sampling. This should
                have the same length as center_fractions. If multiple values
                are provided, then one of these is chosen uniformly each time.
                An acceleration of 4 retains 25% of the columns, but they may
                not be spaced evenly.
        """
        if len(center_fractions) != len(accelerations):
            raise ValueError('Number of center fractions should'
                             'match number of accelerations')

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.is_train = is_train
        self.seed = seed

    def __call__(self, sample):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created. The
                shape should have at least 3 dimensions. Samples are drawn
                along the second last dimension.
            seed (int, optional): Seed for the random number generator. Setting
                the seed ensures the same mask is generated each time for the
                same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """
        # reset seed
        if self.seed:
            np.random.seed(self.seed)

        # extract information
        np_kspace = sample["kspace"]
        nFE, nPEOS = np_kspace.shape[-2:]
        nPE = sample['attrs']['nPE']
        np_full_mask = []
        np_acc = []
        np_acc_true = []

        data_ndim = len(np_kspace.shape[1:])
        for _ in range(np_kspace.shape[0]):
            # FAIR code starts here
            # num_cols = nPE <-- This should be the correct one...
            num_cols = nPEOS # <-- This is wrong, because lines should not be sampled in the "oversampling" regions...
            if not self.is_train:
                state = np.random.get_state()
                fnum = ''.join([n for n in sample['fname'] if n.isdigit()])
                seed = int(fnum[0:9])
                np.random.seed(seed)
                
            choice = np.random.randint(0, len(self.accelerations))
            center_fraction = self.center_fractions[choice]
            acc = self.accelerations[choice]
            np_acc.append(acc)

            # Create the sampling line
            num_low_freqs = int(round(num_cols * center_fraction))
            prob = (num_cols / acc - num_low_freqs) / \
                   (num_cols - num_low_freqs)

            line = np.random.uniform(size=num_cols) < prob
            pad = (num_cols - num_low_freqs + 1) // 2
            line[pad:pad + num_low_freqs] = True

            # FAIR code ends here
            if not self.is_train:
                np.random.set_state(state)

            # parameter to mask oversampling
            padl = int(np.floor((nPEOS - nPE) / 2))
            padr = int(np.ceil((nPEOS - nPE) / 2))


            if padl > 0 and padr > 0:
                # mask base resolution
                line[0:padl] = 1 # Set oversampling regions in mask to one to force data consistency to be zero there
                line[-padr:] = 1
                acc_true = line[padl:-padr].size / np.sum(line[padl:-padr]) # True acceleration factor only in *sampled* region
            else:
                acc_true = line.size / np.sum(line)
            np_acc_true.append(acc_true)

            # Reshape the mask to match the input size
            np_mask = np.repeat(line[np.newaxis, ...], nFE, axis=0)
            np_mask = np_mask.reshape((1,) * (data_ndim - 2) + np_mask.shape)

            np_full_mask.append(np_mask)

        sample['mask'] = np.array(np_full_mask)
        sample['acceleration'] = np.array(np_acc)
        sample['acceleration_true'] = np.array(np_acc_true)

        return sample

class ComputeInit():
    def __init__(self, multicoil=True):
        self.multicoil = multicoil

    def __call__(self, sample):
        """ Data should have folloing shapes:

        kspace: [nslice, nc, nx, ny]
        mask: [1, 1, nx, ny]
        smaps: [nslice, nc, nx, ny]
        target: [nslice, nx, ny]

        """
        np_kspace = sample["kspace"]
        np_mask = sample["mask"]

        # shift data to avoid fftshift / ifftshift in network training
        np_mask = np.fft.ifftshift(np_mask, axes=(-2, -1))
        Ny, Nx = np_kspace.shape[-2:]
        x, y = np.meshgrid(np.arange(1, Nx + 1), np.arange(1, Ny + 1))
        adjust = (-1) ** (x + y)
        np_kspace = np.fft.ifftshift(np_kspace, axes=(-2, -1)) * adjust

        # compute init and extract norm
        np_kspace *= np_mask

        if self.multicoil:
            np_smaps = sample["smaps"]
            np_input = medutils.mri.mriAdjointOpNoShift(
                np_kspace, np_smaps, np_mask, fft_axes=(-2, -1), coil_axis=(1))
        else:
            np_input = medutils.mri.ifft2(np_kspace)

        norm = np.max(medutils.mri.rss(np_input, 1), axis=(1,2))

        def _batch_normalize(x):
            # match the shape of norm array
            return x / norm.reshape(len(norm), *[1]*len(x.shape[1:]))

        # if singlecoil, remove "coil_dim"
        if not self.multicoil:
            np_input = np_input[:,0]
        if not self.multicoil and 'reference' in sample:
            sample['reference'] = sample['reference'][:,0]
            
        sample['norm'] = norm
        sample['mask'] = np_mask.astype(np.float32)
        sample['noisy'] = _batch_normalize(np_input)[...,np.newaxis].astype(np.complex64)
        sample['kspace'] = _batch_normalize(np_kspace).astype(np.complex64)
        if 'reference' in sample:
            sample['reference'] = _batch_normalize(sample['reference'])[...,np.newaxis].astype(np.complex64)
        return sample

class LoadCoilSensitivities():
    """Load Coil Sensitivities
    """
    def __init__(self, smaps_dir, num_smaps):
        assert isinstance(smaps_dir, str)
        self.smaps_dir = smaps_dir
        self.num_smaps = num_smaps

    def __call__(self, sample):
        fname = sample["fname"].split('/')[-1]
        # norm = sample['attrs']['norm_str']

        # load smaps for current slice index
        h5_data = h5py.File(
            os.path.join(sample['rootdir'], self.smaps_dir, fname),
            'r',
        )

        rss_key = 'rss_max'
        is_testset = rss_key not in h5_data.attrs.keys()

        np_smaps = []
        np_target = []
        # np_normval = []
        # np_max = []
        # np_mean = []
        # np_cov = []
        # np_rss_max = []

        for i in range(sample['acceleration'].shape[0]):
            # check whether sensitivity map for acl15 or acl30 is loaded
            if 'acl' in sample.keys() and sample['acl'] is not None:
                acl = sample['acl']
            else:
                acl = 15 if sample['acceleration'][i] >= 8 else 30

            # training data
            # use only num_smaps set of espirit coil sensitivity maps
       #     try:
            smaps_sl = h5_data[f"smaps_acl{acl}"]
            smaps_sl = fastmri.data.utils.np_ensure_complex64(
            smaps_sl[sample["slidx"][i], :, :self.num_smaps:]
        )
            # except:
            #     smaps_sl = h5_data[f"smaps"][sample["slidx"][i]]
            np_smaps.append(smaps_sl)

            # try:
                # normval = h5_data.attrs[f"{norm}_acl{acl}"]
                # np_normval.append(normval)
                # np_mean.append(h5_data.attrs[f'lfimg_mean_acl{acl}'] / normval)
                # np_cov.append(
                #     np.array(h5_data.attrs[f'lfimg_cov_acl{acl}']) / normval)

            if not is_testset:
                ref = h5_data[f"reference_acl{acl}"]
                np_target.append(fastmri.data.utils.np_ensure_complex64(
                    ref[sample["slidx"][i], :self.num_smaps:]
                ))
                    # np_max.append(
                    #     h5_data.attrs[f'reference_max_acl{acl}'] / normval)
                    # np_rss_max.append(h5_data.attrs[rss_key]/normval)
            # except:
            #     pass

        h5_data.close()

        # try:
        #     sample['attrs']['norm'] = np.array(np_normval)
        #     sample['attrs']['mean'] = np.array(np_mean)
        #     sample['attrs']['cov'] = np.array(np_cov)
        # except:
        #     pass

        sample['smaps'] = np.array(np_smaps)

        if not sample['kspace'].ndim == sample['smaps'].ndim:
            sample['smaps'] = np.ascontiguousarray(sample['smaps'][:, :, None, ...])

        if not is_testset:
            sample['reference'] = np.array(np_target)
            # sample['attrs']['ref_max'] = np.array(np_max)
            # sample['attrs']['rss_max'] = np.array(np_rss_max)

        return sample

class InitForegroundMask():
    def __call__(self, sample):
        if 'fg_mask' in sample:
            raise ValueError('Foreground mask exists!')
        fg_mask = np.ones_like(sample['noisy'], dtype=np.float32)
        if fg_mask.ndim == 5:
            fg_mask = np.sum(fg_mask, 2)
        sample['fg_mask'] = fg_mask
        return sample
        
class LoadForegroundMask():
    def __init__(self, fg_dir):
        assert isinstance(fg_dir, (str))
        self.fg_dir = fg_dir

    def __call__(self, sample):
        fname = sample["fname"].split('/')[-1]
        h5_data = h5py.File(
            os.path.join(sample['rootdir'], self.fg_dir, fname),
            'r',
        )

        sample['fg_mask'] = fastmri.data.utils.np_ensure_float32(h5_data['foreground'][sample["slidx"]][:,None])
        h5_data.close()
        return sample

# def get_keras_transform(mode, config):
#     from merlintf.keras.utils import ToKerasIO
#     if mode == 'singlecoil_train':
#         transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
#                                                         accelerations=config['accelerations'],
#                                                         is_train=True),
#                     GeneratePatches(**config[f'{mode}_ds']['patch'], multicoil=False),
#                     ComputeInit(multicoil=False),
#                     ToKerasIO(['noisy', 'kspace', 'mask'], ['reference'])
#                     ]
#     elif mode == 'singlecoil_val':
#         transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
#                                                         accelerations=config['accelerations'],
#                                                         is_train=False),
#                     GeneratePatches(**config[f'{mode}_ds']['patch'], multicoil=False),
#                     ComputeInit(multicoil=False),
#                     ToKerasIO(['noisy', 'kspace', 'mask'], ['reference'])
#                     ]

#     elif mode == 'singlecoil_test':
#         transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
#                                                         accelerations=config['accelerations'],
#                                                         is_train=False),
#                     ComputeInit(multicoil=False),
#                     ]
#     else:
#         raise ValueError(f'Mode {mode} does not exist!')
#     return transform

def get_torch_transform(mode, config):
    #from merlinth.utils import ToTorchIO
    fg_mask = ['fg_mask'] if 'use_fg_mask' in config and config['use_fg_mask'] else []
    if mode == 'singlecoil_train':
        transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
                                                        accelerations=config['accelerations'],
                                                        is_train=True),
                    GeneratePatches(**config[f'{mode}_ds']['patch'], multicoil=False),
                    ComputeInit(multicoil=False),
                    Transpose([('noisy', (0, 3, 1, 2)), ('reference',  (0, 3, 1, 2))]),
                    ToTorchIO(['noisy', 'kspace', 'mask'] + fg_mask, ['reference'])
                    ]
    elif mode == 'singlecoil_val':
        transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
                                                        accelerations=config['accelerations'],
                                                        is_train=False),
                    GeneratePatches(**config[f'{mode}_ds']['patch'], multicoil=False),
                    ComputeInit(multicoil=False),
                    Transpose([('noisy', (0, 3, 1, 2)), ('reference',  (0, 3, 1, 2))]),
                    ToTorchIO(['noisy', 'kspace', 'mask'] + fg_mask, ['reference'])
                    ]

    elif mode == 'singlecoil_test':
        transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
                                                        accelerations=config['accelerations'],
                                                        is_train=False),
                    ComputeInit(multicoil=False),
                    Transpose([('noisy', (0, 3, 1, 2))]),
                    ]
    elif mode == 'denoising_singlecoil_train':
        transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
                                                        accelerations=config['accelerations'],
                                                        is_train=True),
                    ComputeInit(multicoil=False),
                    CenterCrop(['noisy']),
                    Transpose([('noisy', (0, 3, 1, 2)), ('reference',  (0, 3, 1, 2))]),
                    #TODO normalization
                    ToTorchIO(['noisy',], ['reference'])
                    ]
    elif mode == 'denoising_singlecoil_val':
        transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
                                                        accelerations=config['accelerations'],
                                                        is_train=False),
                    ComputeInit(multicoil=False),
                    CenterCrop(['noisy']),
                    Transpose([('noisy', (0, 3, 1, 2)), ('reference',  (0, 3, 1, 2))]),
                    ToTorchIO(['noisy'], ['reference'])
                    ]
    elif mode == 'denoising_singlecoil_test':
        transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
                                                        accelerations=config['accelerations'],
                                                        is_train=False),
                    ComputeInit(multicoil=False),
                    CenterCrop(['noisy']),
                    Transpose([('noisy', (0, 3, 1, 2))]),
                    ]
    elif mode == 'multicoil_train':
        transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
                                                        accelerations=config['accelerations'],
                                                        is_train=True),
                    LoadCoilSensitivities(config[f'{mode}_ds']['sens_dir'], num_smaps=config['num_smaps']),
                    GeneratePatches(**config[f'{mode}_ds']['patch'], multicoil=True),
                    ComputeInit(multicoil=True),
                    Transpose([('noisy', (0, 4, 1, 3, 2)), ('reference',  (0, 4, 1, 3, 2))]),
                    ToTorchIO(['noisy', 'kspace', 'mask', 'smaps'] + fg_mask, ['reference'])
                    ]
    elif mode == 'multicoil_val':
        transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
                                                        accelerations=config['accelerations'],
                                                        is_train=True),
                    LoadCoilSensitivities(config[f'{mode}_ds']['sens_dir'], num_smaps=config['num_smaps']),
                    GeneratePatches(**config[f'{mode}_ds']['patch'], multicoil=True),
                    ComputeInit(multicoil=True),
                    Transpose([('noisy', (0, 4, 1, 3, 2)), ('reference',  (0, 4, 1, 3, 2))]),
                    ToTorchIO(['noisy', 'kspace', 'mask', 'smaps'] + fg_mask, ['reference'])
                    ]
    elif mode == 'multicoil_test':
        transform = [GenerateRandomFastMRIChallengeMask(center_fractions=config['center_fractions'],
                                                        accelerations=config['accelerations'],
                                                        is_train=True),
                    LoadCoilSensitivities(config[f'{mode}_ds']['sens_dir'], num_smaps=config['num_smaps']),
                    ComputeInit(multicoil=True),
                    Transpose([('noisy', (0, 4, 1, 3, 2))]),
                    ]
    else:
        raise ValueError(f'Mode {mode} does not exist!')

    # insert foreground mask
    if config['use_fg_mask']:
        transform.insert(1, LoadForegroundMask(config[f'{mode}_ds']['fg_dir']),)
    elif 'test' in mode:
        transform.append(InitForegroundMask())
    else:
        transform.insert(-1, InitForegroundMask())

    # magnitude
    if mode.startswith('denoising') and 'magnitude' in config and config['magnitude']:
        if 'test' in mode:
            transform.insert(-1, Magnitude(['noisy']))
        else:
            transform.insert(-1, Magnitude(['noisy', 'reference']))
    return transform