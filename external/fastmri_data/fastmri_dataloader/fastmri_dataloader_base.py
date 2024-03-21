from __future__ import print_function

import os
import h5py
import numpy as np
import pandas as pd
import merlinpy

class FastmriCartesianDatasetBase(object):
    """MRI data set."""

    def __init__(
        self,
        config,
        mode
    ):
        """
        Args:
            csv_file (string): Path to the csv data set descirption file.
            root_dir (string): Directory with all the data.
            data_filter (dict): Dict of filter options that should be applied
                to csv_file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = config['root_dir']
        self.batch_size = config['batch_size']
        self.full = True if 'test' in mode else config[f'{mode}_ds']['full']
        self.mode = mode

        data_set = pd.read_csv(config[f'{mode}_ds']['csv_file'])
        data_filter = config['data_filter'] if 'data_filter' in config else {}
        slices = config[f'{mode}_ds']['slices'] if 'slices' in config[f'{mode}_ds'] else {}

        for key in data_filter.keys():
            if key != 'loc':
                data_set = data_set[data_set[key].isin(data_filter[key])]
        
        if 'loc' in data_filter:
            data_set = pd.DataFrame(
                data_set.loc[data_set.filename == data_filter['loc']])
        elif 'enc_y' in data_set and ('train' in self.mode or 'val' in self.mode):
            print('Discard samples with "enc_y > 500" ')
            data_set = data_set[data_set.enc_y <= 500]

        self.data_set = []
        self.full_data_set = []

        #minsl = slices['min'] if 'min' in slices else 0
        for ii in range(len(data_set)):
            subj = data_set.iloc[ii]
            fname = subj.filename
            nPE = subj.nPE
            h5_data = h5py.File(os.path.join(self.root_dir, fname), 'r')
            kspace = h5_data['kspace']
            num_slices = kspace.shape[0]

            if 'slices_min' in subj:
                minsl = subj.slices_min
            elif 'min' in slices:
                minsl = slices['min']
            else:
                minsl = 0
                
            if 'slices_max' in subj and subj.slices_max > 0:
                maxsl = np.minimum(subj.slices_max, num_slices - 1)
            elif 'max' in slices and slices['max'] > 0:
                maxsl = np.minimum(slices['max'], num_slices - 1)
            else:
                maxsl = num_slices - 1

            assert minsl <= maxsl
            attrs = {'nPE': nPE, 'metadata': subj.to_dict()}
            if 'patch_ny' in subj:
                attrs.update({'patch_ny' : subj.patch_ny, 'offset_y' : subj.offset_y})

            self.data_set += [(fname, minsl, maxsl, attrs)]
            self.full_data_set += [
                (fname, si, np.minimum(si + self.batch_size - 1, maxsl), attrs)
                for si in range(minsl, maxsl, self.batch_size)
            ]
            h5_data.close()

        if self.full:
            self.data_set = self.full_data_set

    def __len__(self):
        return len(self.data_set)

    def __getitem__(self, idx):
        fname, minsl, maxsl, attrs = self.data_set[idx]

        slice_range = np.arange(minsl, maxsl + 1)
        slice_prob = np.ones_like(slice_range, dtype=float)
        slice_prob /= slice_prob.sum()

        slidx = list(np.sort(np.random.choice(
            slice_range,
            min(self.batch_size, maxsl + 1 - minsl),
            p=slice_prob,
            replace=False,
        )))

        # load the kspace data for the given slidx
        with h5py.File(
                os.path.join(self.root_dir, fname),
                'r',
                # libver='latest',
                # swmr=True,
        ) as data:
            np_kspace = data['kspace']
            if self.batch_size > np_kspace.shape[0]:
                np_kspace = np_kspace[:]  # faster than smart indexing
            else:
                np_kspace = np_kspace[slidx]
            np_kspace = merlinpy.utils.np_ensure_complex(np_kspace)
            np_kspace_bg = merlinpy.utils.np_ensure_complex(data['kspace'][0])

            # load extra metadata for test data
            np_line = np_acc = np_acl = None
            if 'mask' in data.keys():
                np_line = data['mask'][()]
            if 'acceleration' in data.attrs.keys():
                np_acc = data.attrs['acceleration']
            if 'num_low_frequency' in data.attrs.keys():
                np_acl = data.attrs['num_low_frequency']

        if not 'singlecoil' in self.mode:
            # add dimension for smaps
            np_kspace = np_kspace[:, :, np.newaxis]

        sample = {
            "kspace": np_kspace,
            "kspace_bg": np_kspace_bg,
            "line": np_line,
            "acceleration": np_acc,
            "acl": np_acl,
            "attrs": attrs,
            "slidx": slidx,
            "fname": fname,
            "rootdir": os.path.join(self.root_dir, fname.split('multicoil')[0])
        }

        if self.transform:
            sample = self._apply_transform(sample)

        return sample
    
    def _apply_transform(self):
        return NotImplementedError