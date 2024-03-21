from __future__ import print_function

import os
import numpy as np
from . import transforms
from functools import reduce
import tensorflow as tf
import unittest
import merlinpy
from .fastmri_dataloader_base import FastmriCartesianDatasetBase

class FastmriCartesianDataset(FastmriCartesianDatasetBase, tf.keras.utils.Sequence):
    """MRI data set."""

    def __init__(
        self,
        config,
        mode,
        shuffle=False
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
        super().__init__(config, mode)
        self.transform = transforms.get_keras_transform(mode, config)
        self.shuffle = shuffle

    def keras_inputs(self):
        if 'singlecoil' in self.mode:
            return [
                    tf.keras.Input(name='noisy', shape=(None, None, 1), size=self.batch_size, dtype=tf.complex64),
                    tf.keras.Input(name='kspace', shape=(None, None),   size=self.batch_size, dtype=tf.complex64),
                    tf.keras.Input(name='mask', shape=(None, None), size=self.batch_size, dtype=tf.float32)
                    ]
        else:
            return [
                tf.keras.Input(name='noisy', shape=(None, None, 1), size=self.batch_size, dtype=tf.complex64),
                tf.keras.Input(name='kspace', shape=(None, None, None),   size=self.batch_size, dtype=tf.complex64),
                tf.keras.Input(name='mask', shape=(None, None, None), size=self.batch_size, dtype=tf.float32),
                tf.keras.Input(name='smaps', shape=(None, None, None),   size=self.batch_size, dtype=tf.complex64),
                ]

    def on_epoch_end(self):
        'Updates indeces after each epoch'
        self.indeces = np.arange(len(self.data_set))
        if self.shuffle == True:
            np.random.shuffle(self.indeces)

    def _apply_transform(self, sample):
        return reduce(lambda x, f: f(x), self.transform, sample)

class TestDataloader(unittest.TestCase):
    def _test(self, mode):
        import medutils

        path = os.path.dirname(os.path.realpath(__file__))
        config = merlinpy.loadYaml(f'{path}/config.yml', 'BaseExperiment')
        ds = FastmriCartesianDataset(config, mode=mode)

        if not 'test' in mode:
            inputs, outputs = ds.__getitem__(0)
            # extract inputs
            img = inputs[0][...,0]
            kspace = inputs[1]
            mask = inputs[2]

            ref = outputs[0][...,0]
            ref_np = np.abs(ref)
            medutils.visualization.imsave(medutils.visualization.plot_array(ref_np),   f'{mode}_ref.png')
        else:
            inputs = ds.__getitem__(0)
            img = inputs['noisy'][...,0]
            kspace = inputs['kspace']
            mask = inputs['mask']
        
        img_np = np.abs(img)
        kspace_np = np.abs(kspace)
        kspace_np = np.fft.fftshift(kspace_np, (-2,-1))
        mask_np = np.fft.fftshift(mask, (-2,-1))

        medutils.visualization.imsave(medutils.visualization.plot_array(img_np),   f'{mode}_img.png')
        medutils.visualization.ksave(medutils.visualization.plot_array(kspace_np), f'{mode}_kspace.png')
        medutils.visualization.imsave(medutils.visualization.plot_array(mask_np),  f'{mode}_mask.png')
        

    def testSinglecoilTrain(self):
        self._test('singlecoil_train')

    def testSinglecoilVal(self):
        self._test('singlecoil_val')

    def testSinglecoilTest(self):
        self._test('singlecoil_test')
        
if __name__ == '__main__':
    unittest.test()