from __future__ import print_function

import merlinpy
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision

from ensemble.dataset.fastmri_dataloader_base import FastmriCartesianDatasetBase
from ensemble.dataset import transforms


class FastmriCartesianDataset(FastmriCartesianDatasetBase, torch.utils.data.Dataset):

    def __init__(self, config, mode):
        super().__init__(config, mode)
        self.transform = torchvision.transforms.Compose(transforms.get_torch_transform(mode, config))

    def __len__(self):
        return len(self.data_set)

    def _apply_transform(self, sample):
        return self.transform(sample)


if __name__ == '__main__':
	import merlinth
	import medutils

	config = merlinpy.experiment.loadYaml('./config/config.yml', 'BaseExperiment')
	batch_size = 8
	num_workers = 1
	mode = 'singlecoil_train'
	ds_train = FastmriCartesianDataset(config, mode=mode)	

	mode = 'singlecoil_train'
	inputs, outputs = ds_train[0]

	# extract inputs
	img_np = merlinth.utils.torch_to_complex_abs_numpy(inputs[0][:,0])
	kspace_np = merlinth.utils.torch_to_complex_abs_numpy(inputs[1])
	mask_np = inputs[2].numpy()

	ref = outputs[0][:,0]
	ref_np = merlinth.utils.torch_to_complex_abs_numpy(ref)
	medutils.visualization.imsave(medutils.visualization.plot_array(ref_np),   f'{mode}_ref.png')

	kspace_np = np.fft.fftshift(kspace_np, (-2,-1))
	mask_np = np.fft.fftshift(mask_np, (-2,-1))

	medutils.visualization.imsave(medutils.visualization.plot_array(img_np),   f'{mode}_img.png')
	medutils.visualization.ksave(medutils.visualization.plot_array(kspace_np), f'{mode}_kspace.png')
	medutils.visualization.imsave(medutils.visualization.plot_array(mask_np),  f'{mode}_mask.png')

	dl = DataLoader(ds_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
	for batch in dl:
		print(batch)
		break