import glob
import logging
import os

import matplotlib.colors
import pandas as pd
#import yaml
import pathlib as plb
from PIL import ImageDraw, Image
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import h5py

import torch
import torchvision
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import wandb
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

import merlinpy
import merlinth
from ensemble.models.recon_model import UnrolledNetwork
#from ensemble.models.architecture.vn import VNet
from ensemble.models.architecture.model_vn import VN
from ensemble.dataset.fastmri import FastmriCartesianDataset
import ensemble.models.losses as losses
from ensemble.models.BlockAdam import BlockAdam
from merlinth.losses.ssim import ssim
from merlinth.losses.pairwise_loss import psnr, nmse
from ensemble.models.architecture.postprocess import postprocess
from torch.nn import functional as F
import sklearn.metrics as skm
#from merlinth.mytorch.utils import State
import warnings
warnings.filterwarnings("ignore")


class EnsembleModel(pl.LightningModule):

    def __init__(self, config, predictions_dir=None, **kwargs):
        super().__init__()
        # save the config 
        self.save_hyperparameters(config)
        is_training = not config['predict']
        if config['model']['R_type'] == 'foe':
            model_name = f'VN-{args.experiment}'
            self.model_recon = VN(args.config['model'], is_training, args.config['multicoil'])
        else:
            model_name = f'UnrolledNet-{args.experiment}'
            self.model_recon = UnrolledNetwork(args.config['model'], is_training, args.config['multicoil'])
        self.optim = self.hparams['optim']
        self.lr = self.hparams['lr']
        self.magnitude = True if self.hparams['model']['R_type'] == 'mag_unet' else False
        self.vn = True if config['model']['R_type'] == 'foe' else False
        self.R_type = config['model']['R_type']
        self.parameter_estimation = config['model']['parameter_estimation']
        self.multicoil = args.config['multicoil']

        if self.R_type == 'foe':
            self.R_input = 'foe'
        else:
            if self.magnitude:
                self.R_input = 'mag'
            else:
                self.R_input = 'real2ch'

        # build metric functions
        self.metrics = {
            'MSE': lambda x, y: F.mse_loss(losses.complex_abs(x), losses.complex_abs(y)),
            'NMSE': lambda x, y: nmse(x, y),
            'RMSE': lambda x, y: losses.loss_complex_rmse(x, y),
            'PSNR': lambda x, y: psnr(
                x, y, data_range=config['ref_max']),
            'SSIM': lambda x, y: ssim(
                losses.complex_abs(x), losses.complex_abs(y), data_range=config['ref_max']),
            'ABSMSE': lambda x, y: losses.loss_abs_mse(x, y),
            'COMPLEXMSE': lambda x, y: losses.loss_complex_mse(x, y)
        }

        # loss function
        self.lossfunc = losses.get_loss(self.hparams['loss'])
        self.plot_interval = self.hparams['plot_interval']
        self.predictions_dir = predictions_dir

    def configure_optimizers(self):
        if self.optim.lower() == 'sgd':
            return optim.SGD(self.model_recon.parameters(), lr=self.lr, momentum=0.9)
        elif self.optim.lower() == 'adam':
            return optim.Adam(self.model_recon.parameters(), lr=self.lr)
        elif self.optim.lower() == 'blockadam':
            return BlockAdam(self.model_recon.parameters(), lr=self.lr)

        
    def forward(self, img, kspace, mask, smaps, ref):
        if self.multicoil:
            return self.model_recon([img, kspace, mask, smaps])
        else:
            return self.model_recon([img, kspace, mask])

    def on_pretrain_routine_start(self):
        #merlinpy.wandb.fix_dict_in_wandb_config(wandb)
        pass

    def training_step(self, batch, batch_idx):
        batch, attrs, fname = batch
        inputs, outputs = batch

        if self.R_input == 'mag':
            img = inputs[0][:, 0, ...]
            kspace = inputs[1][:, 0, ...]
            mask = inputs[2][:, 0, ...]
            ref = outputs[0][:, 0, ...]
            if self.multicoil:
                smaps = inputs[3][:, 0, ...]
            else:
                smaps = 1

            if self.hparams['loss'] == 'loss_abs_mse':
                if self.parameter_estimation:
                    y, sigma = self(img, kspace, mask, smaps, ref)
                    loss = self.lossfunc(ref, y, sigma)
                else:
                    y = self(img, kspace, mask, smaps, ref)
                    loss = self.lossfunc(ref, y)
            else:
                # SURE loss
                y = self(img, kspace, mask, smaps, ref)
                sigma_in = torch.normal(torch.zeros(ref.size()[0]), torch.arange(torch.min(ref), torch.max(ref), ref.size()[0] * torch.abs(torch.max(ref) - torch.min(ref))))
                b_prime = torch.randn(img.size()).to('cuda')
                #eps_tf = 1.6 * sigma_in * 0.0001
                eps = torch.max(img) * 0.001
                imgptb = img + torch.multiply(b_prime, eps)
                yptb = self(imgptb, kspace, mask, smaps, ref)
                loss = self.lossfunc(ref, y, yptb, b_prime, eps, sigma_in)

        elif self.R_input == 'real2ch':
            img = merlinth.complex2real(inputs[0][:, 0, ...])
            kspace = inputs[1]  # leave complex, since only used in DC
            mask = inputs[2]
            ref = outputs[0][:, 0, ...]
            if self.multicoil:
                smaps = inputs[3]
            else:
                smaps = 1

            if self.hparams['loss'] == 'loss_abs_mse':
                if self.parameter_estimation:
                    y, sigma = self(img, kspace, mask, smaps, ref)
                    loss = self.lossfunc(ref, merlinth.real2complex(y), merlinth.real2complex(sigma))
                else:
                    y = self(img, kspace, mask, smaps, ref)
                    loss = self.lossfunc(ref, merlinth.real2complex(y))
            else:
                # SURE loss
                y = self(img, kspace, mask, smaps, ref)
                y = merlinth.real2complex(y)
                sigma_in = torch.normal(torch.zeros(ref.size()[0]), torch.arange(torch.min(torch.abs(ref)), torch.max(torch.abs(ref)), ref.size()[0] * torch.abs(torch.max(torch.abs(ref)) - torch.min(torch.abs(ref)))))
                b_prime = torch.randn(img.size()).to('cuda')
                #eps_tf = 1.6 * sigma_in * 0.0001
                eps = torch.max(img) * 0.001
                imgptb = img + torch.multiply(b_prime, eps)
                yptb = self(imgptb, kspace, mask, smaps, ref)
                yptb = merlinth.real2complex(yptb)
                loss = self.lossfunc(ref, y, yptb, b_prime, eps, sigma_in)

        elif self.R_input == 'foe':
            img = inputs[0][:, 0, ...]
            kspace = inputs[1]
            mask = inputs[2]
            ref = outputs[0][:, 0, ...]
            if self.multicoil:
                smaps = inputs[3]
            else:
                smaps = 0

            y = self(img, kspace, mask, smaps, ref)

            loss = self.lossfunc(ref, y)
            
        else:
            img = inputs[0][:, 0, ...].permute([0, 4, 2, 3, 1])[..., 0]  # move real/imag into channel dimension
            kspace = inputs[1][:, 0, ...]
            mask = inputs[2][:, 0, ...]
            ref = outputs[0][:, 0, ...]
            if self.multicoil:
                smaps = inputs[3][:, 0, ...]
            else:
                smaps = 1

            y = self(img, kspace, mask, smaps, ref)

            loss = self.lossfunc(ref, y)

        self.log('train_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        batch, attrs, fname = batch
        inputs, outputs = batch

        file_name = os.path.basename(fname[0]).split('.')[0]
        slice = int(attrs['slidx'][0].detach().cpu().numpy())
        if os.path.exists(f'{self.predictions_dir}/{file_name}_{slice:03d}_ref.npy'):
            return []

        if self.R_input == 'mag':
            img = inputs[0][:, 0, ...]
            kspace = inputs[1][:, 0, ...]
            mask = inputs[2][:, 0, ...]
            ref = outputs[0][:, 0, ...]
            if self.multicoil:
                smaps = inputs[3][:, 0, ...]
            else:
                smaps = 1

            if self.hparams['loss'] == 'loss_abs_mse':
                if self.parameter_estimation:
                    y, sigma = self(img, kspace, mask, smaps, ref)
                else:
                    y = self(img, kspace, mask, smaps, ref)
            else:
                # SURE loss
                y = self(img, kspace, mask, smaps, ref)

        elif self.R_input == 'real2ch':
            img = merlinth.complex2real(inputs[0][:, 0, ...])
            kspace = inputs[1]  # leave complex, since only used in DC
            mask = inputs[2]
            ref = merlinth.complex2real(outputs[0][:, 0, ...])
            if self.multicoil:
                smaps = inputs[3]
            else:
                smaps = 1

            if self.hparams['loss'] == 'loss_abs_mse':
                if self.parameter_estimation:
                    y, sigma = self(img, kspace, mask, smaps, ref)
                else:
                    y = self(img, kspace, mask, smaps, ref)
            else:
                # SURE loss
                y = self(img, kspace, mask, smaps, ref)

        elif self.R_input == 'foe':
            img = inputs[0][:, 0, ...]
            kspace = inputs[1]
            mask = inputs[2]
            ref = outputs[0][:, 0, ...]
            if self.multicoil:
                smaps = inputs[3]
            else:
                smaps = 1

            y = self(img, kspace, mask, smaps, ref)

        else:
            img = inputs[0][:, 0, ...].permute([0, 4, 2, 3, 1])[..., 0]  # move real/imag into channel dimension
            kspace = inputs[1][:, 0, ...]
            mask = inputs[2][:, 0, ...]
            ref = outputs[0][:, 0, ...]
            if self.multicoil:
                smaps = inputs[3][:, 0, ...]
            else:
                smaps = 1

            y = self(img, kspace, mask, smaps, ref)

        if torch.sum(y).cpu().numpy() > 0:
            if not os.path.exists(f'{self.predictions_dir}/{file_name}_{slice:03d}_y.npy'):
                np.save(f'{self.predictions_dir}/{file_name}_{slice:03d}_y.npy', y.squeeze().cpu().numpy())   # {batch_idx}_
            if not os.path.exists(f'{self.predictions_dir}/{file_name}_{slice:03d}_ref.npy'):
                np.save(f'{self.predictions_dir}/{file_name}_{slice:03d}_ref.npy', ref.squeeze().cpu().numpy())   #{batch_idx}_
            if self.parameter_estimation:
                if not os.path.exists(f'{self.predictions_dir}/{file_name}_{slice:03d}_sigma.npy'):
                    np.save(f'{self.predictions_dir}/{file_name}_{slice:03d}_sigma.npy', sigma.squeeze().cpu().numpy())   #{batch_idx}_

            # evaluations
            metrics = {}
            metrics['slice'] = slice
            for d in self.metrics.keys():
                metrics[d] = np.asscalar(self.metrics[d](ref, y).detach().cpu().numpy())

            if self.parameter_estimation:
                metrics['MNNLL'] = self.lossfunc(ref, y, sigma).detach().cpu().numpy()

            if os.path.exists(f'{self.predictions_dir}/{file_name}_metrics.csv'):
                df = pd.read_csv(f'{self.predictions_dir}/{file_name}_metrics.csv')
                df = df.append(metrics, ignore_index=True)
            else:
                df = pd.DataFrame(metrics, index=[0])
            df.to_csv(f'{self.predictions_dir}/{file_name}_metrics.csv')   # {batch_idx}_

        return y, ref


    def predict_simple(self, img, kspace, mask, smaps, ref, fname, slice):
        if self.parameter_estimation:
            y, sigma = self(img, kspace, mask, smaps, ref)
        else:
            y = self(img, kspace, mask, smaps, ref)

        file_name = os.path.basename(fname[0]).split('.')[0]
        #slice = int(attrs['slidx'][0].cpu().numpy())

        np.save(f'{self.predictions_dir}/{file_name}_{slice:03d}_y.npy', y.squeeze().cpu().numpy())  # {batch_idx}_
        np.save(f'{self.predictions_dir}/{file_name}_{slice:03d}_ref.npy', ref.squeeze().cpu().numpy())  # {batch_idx}_
        if self.parameter_estimation:
            np.save(f'{self.predictions_dir}/{file_name}_{slice:03d}_sigma.npy', sigma.squeeze().cpu().numpy())  # {batch_idx}_

    def validation_step(self, batch, batch_idx):
        batch, attrs, fname = batch
        inputs, outputs = batch

        if self.R_input == 'mag':
            img = inputs[0][:, 0, ...]
            kspace = inputs[1][:, 0, ...]
            mask = inputs[2][:, 0, ...]
            ref = outputs[0][:, 0, ...]
            if self.multicoil:
                smaps = inputs[3][:, 0, ...]
            else:
                smaps = 1

            if self.hparams['loss'] == 'loss_abs_mse':
                if self.parameter_estimation:
                    y, sigma = self(img, kspace, mask, smaps, ref)
                else:
                    y = self(img, kspace, mask, smaps, ref)
            else:
                # SURE loss
                y = self(img, kspace, mask, smaps, ref)
                sigma_in = torch.normal(torch.zeros(ref.size()[0]), torch.arange(torch.min(ref), torch.max(ref), ref.size()[0] * torch.abs(torch.max(ref) - torch.min(ref))))
                b_prime = torch.randn(img.size()).to('cuda')
                #eps_tf = 1.6 * sigma_in * 0.0001
                eps = torch.max(img) * 0.001
                imgptb = img + torch.multiply(b_prime, eps)
                yptb = self(imgptb, kspace, mask, smaps, ref)

        elif self.R_input == 'real2ch':
            img = merlinth.complex2real(inputs[0][:, 0, ...])  # move real/imag into channel dimension
            kspace = inputs[1]  # leave complex, since only used in DC
            mask = inputs[2]
            ref = outputs[0][:, 0, ...]
            if self.multicoil:
                smaps = inputs[3]
            else:
                smaps = 1

            if self.hparams['loss'] == 'loss_abs_mse':
                if self.parameter_estimation:
                    y, sigma = self(img, kspace, mask, smaps, ref)
                    sigma = merlinth.real2complex(sigma)
                else:
                    y = self(img, kspace, mask, smaps, ref)
                y = merlinth.real2complex(y)
            else:
                # SURE loss
                y = self(img, kspace, mask, smaps, ref)
                y = merlinth.real2complex(y)
                sigma_in = torch.normal(torch.zeros(ref.size()[0]), torch.arange(torch.min(torch.abs(ref)), torch.max(torch.abs(ref)), ref.size()[0] * torch.abs(torch.max(torch.abs(ref)) - torch.min(torch.abs(ref)))))
                b_prime = torch.randn(img.size()).to('cuda')
                #eps_tf = 1.6 * sigma_in * 0.0001
                eps = torch.max(img) * 0.001
                imgptb = img + torch.multiply(b_prime, eps)
                yptb = self(imgptb, kspace, mask, smaps, ref)
                yptb = merlinth.real2complex(yptb)

        elif self.R_input == 'foe':
            img = inputs[0][:, 0, ...]
            kspace = inputs[1]
            mask = inputs[2]
            ref = outputs[0][:, 0, ...]
            if self.multicoil:
                smaps = inputs[3]
            else:
                smaps = 1

            y = self(img, kspace, mask, smaps, ref)

            y = merlinth.complex_abs(y)
            ref = merlinth.complex_abs(ref)

        else:
            img = merlinth.complex2real(inputs[0])[:, :, 0, ...]  # move real/imag into channel dimension
            kspace = inputs[1][:, 0, ...]
            mask = inputs[2][:, 0, ...]
            ref = outputs[0][:, 0, ...]
            if self.multicoil:
                smaps = inputs[3][:, 0, ...]
            else:
                smaps = 1

            y = self(img, kspace, mask, smaps, ref)

        if batch_idx % self.plot_interval == 0:
            grid = torchvision.utils.make_grid(torch.cat([torch.abs(y[:3]).detach().cpu(), torch.abs(ref[:3]).detach().cpu()]), nrow=3, normalize=True)
            self.logger[1].experiment.log({'reconstructed_samples': [wandb.Image(grid, caption=f'epoch {self.current_epoch}')]})

        if self.hparams['loss'] == 'loss_abs_mse':
            if self.parameter_estimation:
                loss = self.lossfunc(ref, y, sigma)
            else:
                loss = self.lossfunc(ref, y)
        else:
            loss = self.lossfunc(ref, y, yptb, b_prime, eps, sigma_in)
        self.log('val_loss', loss)
        for d in self.metrics.keys():
            self.log(d, np.ndarray.item(self.metrics[d](ref, y).detach().cpu().numpy()))

        return loss

class FastMriDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        self.config_ds = config.copy()
        self.config_ds['magnitude'] = True if self.config_ds['model']['R_type'] == 'mag_unet' else False
        self.config_ds['vn'] = True if self.config_ds['model']['R_type'] == 'foe' else False
        self.batch_size = self.config_ds['batch_size']
        self.num_workers = self.config_ds['num_workers']
        self.predict = self.config_ds['predict']
        self.ood = self.config_ds['ood']
        self.multicoil = self.config_ds['multicoil']

    def setup(self, stage=None):
        if not self.predict:
            if self.multicoil:
                self.ds_train = FastmriCartesianDataset(self.config_ds, mode='multicoil_train')
                self.ds_val = FastmriCartesianDataset(self.config_ds, mode='multicoil_val')
            else:
                self.ds_train = FastmriCartesianDataset(self.config_ds, mode='singlecoil_train')
                self.ds_val = FastmriCartesianDataset(self.config_ds, mode='singlecoil_val')
        else:
            if self.ood == 'brain':
                self.ds_test = FastmriCartesianDataset(self.config_ds, mode='multicoil_test_brain')
            elif self.ood == 'knee':
                self.ds_test = FastmriCartesianDataset(self.config_ds, mode='multicoil_test_knee')
            else:
                if self.multicoil:
                    self.ds_test = FastmriCartesianDataset(self.config_ds, mode='multicoil_test')
                else:
                    self.ds_test = FastmriCartesianDataset(self.config_ds, mode='singlecoil_test')

    def train_dataloader(self):
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, drop_last=True)


def plot_bounding_box(image, labels):
    plotted_image = ImageDraw.Draw(image)
    for label in labels:
        _, _, _, x0, y0, w, h, label_txt = label
        x1 = x0 + w
        y1 = y0 + h
        plotted_image.rectangle(((x0,y0), (x1,y1)), outline="white")
        plotted_image.text((x0, max(0, y0 - 10)), label_txt, fill= "white")
    return np.array(image)


def plot_bounding_box_plt(ax, labels, offset=[0, 0]):
    # offset in x, y direction due to padded image
    for label in labels:
        _, _, _, x0, y0, w, h, label_txt = label
        #x1 = x0 + w
        #y1 = y0 + h
        rect = matplotlib.patches.Rectangle((x0+offset[0], y0+offset[1]), w, h, edgecolor='w', facecolor='none')
        ax.add_patch(rect)
        plt.text(x0+offset[0], max(0, y0+offset[1] - 10), label_txt, color="w")


def real2complex(z, splitdim=1):
    if splitdim == 0:  # no batch dim
        real = z[0, ...]
        imag = z[1, ...]
    elif splitdim == 1:
        real = z[:, 0, ...]
        imag = z[:, 1, ...]
    elif splitdim == 2:
        real = z[:, :, 0, ...]
        imag = z[:, :, 1, ...]
    return real +1j * imag

def np_abs(z, eps=1e-9):
    if np.iscomplex(z).any():
        return np.sqrt(np.real(np.conj(z) * z) + eps)
    else:
        return np.abs(z)

def calculate_metrics(y, ref, sigma, config, metric_dir, subject):
    metrics = {
        'MSE': lambda x, y: F.mse_loss(losses.complex_abs(x), losses.complex_abs(y)),
        'NMSE': lambda x, y: nmse(x, y),
        'RMSE': lambda x, y: losses.loss_complex_rmse(x, y),
        'PSNR': lambda x, y: psnr(
            x, y, data_range=config['ref_max']),
        'SSIM': lambda x, y: ssim(
            losses.complex_abs(x), losses.complex_abs(y), data_range=config['ref_max']),
        'ABSMSE': lambda x, y: losses.loss_abs_mse(x, y),
        'COMPLEXMSE': lambda x, y: losses.loss_complex_mse(x, y)
    }

    metrics_voxel = { 'MAE': lambda x, y: np_abs(np_abs(x) - np_abs(y)).flatten(),
                      'MSE': lambda x, y: (np_abs(x - y) ** 2).flatten(),
                      'NMSE': lambda x, y: (np_abs(x - y) ** 2 / (np.mean(np_abs(x)) + 1e-9)).flatten(),
                      'RMSE': lambda x, y: np.sqrt(np_abs(x - y) ** 2).flatten(),
                      'PSNR': lambda x, y: (20 * np.log10(config['ref_max'] / np.sqrt(np.mean(np_abs(x - y) ** 2)))).flatten()
    }

    # loss function
    lossfunc = losses.get_loss(config['loss'])

    y_ensemble = np.mean(y, axis=0)
    std_ensemble = np.std(y, axis=0)
    if config['model']['parameter_estimation']:
        aleatoric_uncertainty = np.mean(sigma, axis=0)
        epistemic_uncertainty = np.mean(y ** 2, axis=0) - y_ensemble ** 2
        sigma_ensemble = aleatoric_uncertainty + epistemic_uncertainty
        var = aleatoric_uncertainty + 1/std_ensemble * epistemic_uncertainty
        np.save(f'{metric_dir}/{subject}_alep.npy', {'aleatoric_uncertainty': aleatoric_uncertainty, 'epistemic_uncertainty': epistemic_uncertainty, 'var': var})

    def unsqeeze_tensor(x):  # add batch and channel dim
        return torch.unsqueeze(torch.unsqueeze(x, 0), 0)

    metrics_all = []
    metrics_voxel_all = pd.DataFrame()
    for slice in range(y_ensemble.shape[2]):
        metrics_out = {}
        metrics_out['slice'] = slice
        metrics_out_voxel = pd.DataFrame()

        for d in metrics.keys():
            metrics_out[d] = np.asscalar(metrics[d](unsqeeze_tensor(torch.from_numpy(ref[:,:,slice].copy())),
                                                    unsqeeze_tensor(torch.from_numpy(y_ensemble[:,:,slice].copy()))).detach().cpu().numpy())

        for d in metrics_voxel.keys():
            metrics_out_voxel[d] = pd.Series(metrics_voxel[d](ref[:,:,slice], y_ensemble[:,:,slice]))
        metrics_out_voxel['std_ensemble'] = pd.Series(np.asarray(std_ensemble[:,:,slice]).flatten())

        metrics_out['mean_std_ensemble'] = np.asscalar(np.mean(std_ensemble[:,:,slice]))
        if config['model']['parameter_estimation']:
            metrics_out['MNNLL'] = np.asscalar(lossfunc(unsqeeze_tensor(torch.from_numpy(ref[:,:,slice].copy())),
                                            unsqeeze_tensor(torch.from_numpy(y[:,:,:,slice].copy())),
                                            unsqeeze_tensor(torch.from_numpy(sigma[:,:,:,slice].copy()))).detach().cpu().numpy())
            metrics_out['mean_aleatoric'] = np.asscalar(np.mean(aleatoric_uncertainty[:, :, slice]))
            metrics_out['mean_epistemic'] = np.asscalar(np.mean(epistemic_uncertainty[:, :, slice]))
            metrics_out['mean_sigma_ensemble'] = np.asscalar(np.mean(sigma_ensemble[:, :, slice]))
            metrics_out['mean_var'] = np.asscalar(np.mean(var[:, :, slice]))

            metrics_out_voxel['aleatoric'] = pd.Series(np.asarray(aleatoric_uncertainty[:,:,slice]).flatten())
            metrics_out_voxel['epistemic'] = pd.Series(np.asarray(epistemic_uncertainty[:,:,slice]).flatten())
            metrics_out_voxel['sigma_ensemble'] = pd.Series(np.asarray(sigma_ensemble[:,:,slice]).flatten())
            metrics_out_voxel['var'] = pd.Series(np.asarray(var[:,:,slice]).flatten())

        metrics_all.append(metrics_out)
        metrics_out_voxel['slice'] = slice
        metrics_voxel_all = pd.concat([metrics_voxel_all, metrics_out_voxel], ignore_index=True)

    df = pd.DataFrame(metrics_all)
    df.to_csv(f'{metric_dir}/{subject}_metrics.csv')

    metrics_voxel_all.to_csv(f'{metric_dir}/{subject}_metrics_voxel.csv')


def main_predict(args):
    logging.info(args)

    print('===== Experiment: {} ====='.format(args.experiment))
    if os.getenv('ROOT_ENSEMBLE_DIR'):
        root_ensemble_dir = os.getenv('ROOT_ENSEMBLE_DIR')
    else:
        root_ensemble_dir = '/home/rakuest1/Documents/ensemble'
    args.config['result_dir'] = os.path.join(root_ensemble_dir, args.config['result_dir'])
    ckpt_dir = os.path.join(args.config['result_dir'], 'checkpoints')  # , args.config['ckpt_name'])
    ckpts_names = args.config['ckpt_name'].split('_')
    idx = [idx for idx, item in enumerate(ckpts_names) if item.startswith('eX')][0]

    ckpt_dirs = [os.path.join(ckpt_dir,
                              '_'.join(ckpts_names[:idx]) + '_' + ''.join(ckpts_names[idx].replace('eX', 'e{}'.format(i))) + '_' + '_'.join(ckpts_names[idx+1:]) + '_0',
                              '_'.join(ckpts_names[:idx]) + '_' + ''.join(ckpts_names[idx].replace('eX', 'e{}'.format(i))) + '_' + '_'.join(ckpts_names[idx+1:]) + '_0_ensemble') for i in range(20)]
    if not args.config['ckpts']:
        ckpts = {}
        for idxchk, ckpt_dir in enumerate(ckpt_dirs):
            if not os.path.exists(ckpt_dir):
                continue
            dirs = os.listdir(ckpt_dir)
            ckpt_curr = []
            for dircurr in dirs[::-1]:
                filesdircurr = [f for f in os.listdir(os.path.join(ckpt_dir, dircurr, 'checkpoints')) if f.endswith('.ckpt') and not f.startswith('epoch=0')]  # allow early stopping ## and f.startswith('epoch=199')]
                if len(filesdircurr) > 0:
                    dirselected = dircurr
                    ckpt_curr = filesdircurr
                    break
            ckptdir_curr = os.path.join(ckpt_dir, dirselected, 'checkpoints')
            if ckpt_curr:
                ckpts['e{}'.format(idxchk)] = os.path.join(ckptdir_curr, ckpt_curr[-1])
    else:
        ckpts = args.config['ckpts']
        for idxchk, c_name in enumerate(ckpts):
            ckpts[c_name] = os.path.join(ckpt_dirs[idxchk], ckpts[c_name])

    # OOD data
    if args.config['ood'] == 'brain':
        pred_dir = os.path.join(args.config['result_dir'], 'predictions', args.experiment)
        Path(pred_dir).mkdir(exist_ok=True, parents=True)
        plot_dir = Path(pred_dir).joinpath('plots')
        plot_dir.mkdir(exist_ok=True, parents=True)
        data_dir = os.path.join(args.config['root_dir'], 'multicoil_val')
    elif args.config['ood'] == 'knee':
        pred_dir = os.path.join(args.config['result_dir'], 'predictions', args.experiment)
        Path(pred_dir).mkdir(exist_ok=True, parents=True)
        plot_dir = Path(pred_dir).joinpath('plots')
        plot_dir.mkdir(exist_ok=True, parents=True)
        data_dir = os.path.join(args.config['root_dir'])
    else:  # None
        pred_dir = os.path.join(args.config['result_dir'], 'predictions', args.experiment)
        plot_dir = Path(pred_dir).joinpath('plots')
        plot_dir.mkdir(exist_ok=True, parents=True)
        data_dir = os.path.join(args.config['root_dir'], 'knee/data/singlecoil_val')

    selected_subjects = args.config['data_filter']['filename']
    sel_subjects = [os.path.splitext(os.path.basename(s))[0].strip() for s in selected_subjects]
    metric_dir = os.path.join(args.config['result_dir'], 'metrics', args.experiment)
    metric_dir = Path(metric_dir)
    metric_dir.mkdir(exist_ok=True, parents=True)

    # data module
    dm = FastMriDataModule(args.config)
    dm.setup()
    ''''''
    for c_name in ckpts:
        # define model
        print('Predicting with {}'.format(c_name))
        m = EnsembleModel(args.config, predictions_dir=str(pred_dir +'/'+ c_name))
        plb.Path(m.predictions_dir).mkdir(exist_ok=True, parents=True)
        try:
            #print(torch.load(str(ckpts[c_name]))['state_dict'])
            m.load_state_dict(torch.load(str(ckpts[c_name]))['state_dict'])
            trainer = pl.Trainer(gpus=[args.config['gpu']])
            trainer.predict(m, dm.test_dataloader())
        except:
            print('Error loading ' + c_name)

        torch.cuda.empty_cache()

    if args.config['annotation_path']:
        df = pd.read_csv(args.config['annotation_path'], index_col=None, header=0)

    # get back results for plotting the ensemble and variances
    if args.config['model']['R_type'] == 'mag_unet' or args.config['model']['R_type'] == 'foe':
        stackdim = 2
    else:  # real2ch
        stackdim = 3
    for subject in sel_subjects:
        print('Processing subject {}'.format(subject))
        y = []
        ref = None
        sigma = []
        if args.config['annotation_path']:
            labels_for_file = df.loc[df['file'] == subject]

        # get original shape
        if args.config['ood'] == 'knee':
            sub_dir = [s for s in selected_subjects if subject in s]
            orig_shape = np.array(h5py.File(os.path.join(data_dir, sub_dir[0]), 'r')['reference'].shape)
        else:
            orig_shape = np.array(h5py.File(os.path.join(data_dir, subject + '.h5'), 'r')['reconstruction_rss'].shape)

        # labels_for_file['label'].unique()
        for c_name in ckpts:
            print(' |--- Checkpoint {}'.format(c_name))
            yfiles = sorted(filter(os.path.isfile, glob.glob(os.path.join(pred_dir, c_name, subject + '_*_y.npy'))))
            if len(yfiles) == 0:
                continue
            ycurr = []
            refcurr = []
            sigmacurr = []
            for islice, filename in enumerate(yfiles):
                ycurr.append(np.load(os.path.join(pred_dir, c_name, filename)))
                refcurr.append(np.load(os.path.join(pred_dir, c_name, os.path.basename(filename).split('.')[0][:-1] + 'ref' + os.path.splitext(filename)[1])))  # .replace('y', 'ref')
                if args.config['model']['parameter_estimation']:
                    sigmacurr.append(np.load(os.path.join(pred_dir, c_name, os.path.basename(filename).split('.')[0][:-1] + 'sigma' + os.path.splitext(filename)[1])))

            y.append(np.stack(ycurr, axis=stackdim))
            if ref is None:
                ref = np.stack(refcurr, axis=stackdim)
            if args.config['model']['parameter_estimation']:
                sigma.append(np.stack(sigmacurr, axis=stackdim))

        # stack ensembles
        y = np.stack(y, axis=0)  # ensembles x X x Y x slices (x 2)
        if args.config['model']['parameter_estimation']:
            sigma = np.stack(sigma, axis=0)

        # convert real2ch -> complex
        if not (args.config['model']['R_type'] == 'mag_unet' or args.config['model']['R_type'] == 'foe'):
            y = real2complex(y, 1)
            ref = real2complex(ref, 0)
            if args.config['model']['parameter_estimation']:
                sigma = real2complex(sigma, 1)

        # flip images
        y = y[:, ::-1, ...]  # flipped up down
        ref = ref[::-1, ...]  # flipped up down
        diff_shape = np.abs(np.shape(ref) - orig_shape[[1,2,0]])
        offset_box = [int(diff_shape[1]/2), int(diff_shape[0]/2)]
        if args.config['model']['parameter_estimation']:
            sigma = sigma[:, ::-1, ...]  # flipped up down
            y_ensemble = np.mean(y, axis=0)
            std_ensemble = np.std(y, axis=0)
            aleatoric_uncertainty = np.mean(sigma, axis=0)
            epistemic_uncertainty = np.mean(y ** 2, axis=0) - y_ensemble ** 2
            sigma_ensemble = aleatoric_uncertainty + epistemic_uncertainty
            var = std_ensemble + aleatoric_uncertainty
            sigma_ensemble_plot = sigma_ensemble
        else:
            sigma = []
            var = []

        # calculate metrics
        print(' |--- Calculating metrics')
        calculate_metrics(y, ref, sigma, args.config, metric_dir, subject)

        # plot ensemble
        avoid_plotting = True
        if avoid_plotting:
            continue

        print(' |--- Plotting ensemble')
        if args.config['annotation_path']:
            slices_plot = labels_for_file['slice'].unique()
        else:
            slices_plot = []
            labels_for_file = {'slice': []}
            labels_for_file = pd.DataFrame(labels_for_file)
        slices_plot = np.unique(np.append(slices_plot, np.asarray(args.config['selected_slices'].split(','), dtype=int)))
        std_ensemble_plot = np.std(y, axis=0)

        if args.config['model']['R_type'] == 'mag_unet':
            iscale = 0.025
        else:
            iscale = 0.6

        for idx in slices_plot:
            if idx >= np.shape(ref)[2]:
                continue
            # plot ymean w/ label
            if idx in labels_for_file['slice'].unique():
                labels_for_slice = labels_for_file.loc[labels_for_file['slice'] == idx].values.tolist()
                plt.imshow(np.abs(y.mean(axis=0)[..., int(idx)]), cmap='gray')
                plot_bounding_box_plt(plt.gca(), labels_for_slice, offset_box)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_meanlabel.png'))
                plt.close()

                # plot ymean + sigma w/ label
                plt.imshow(np.abs(y.mean(axis=0)[..., int(idx)]), cmap='gray')
                if args.config['model']['parameter_estimation']:
                    plt.imshow(sigma_ensemble_plot[..., int(idx)], vmin=0, vmax=iscale, cmap='inferno', alpha=0.5)  # vmin=0, vmax=0.025
                else:
                    plt.imshow(std_ensemble_plot[..., int(idx)], vmin=0, vmax=iscale, cmap='inferno', alpha=0.5)
                plot_bounding_box_plt(plt.gca(), labels_for_slice, offset_box)
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_meansigmalabel.png'))
                plt.close()

                # plot ref + sigma w/ label
                plt.imshow(np.abs(ref[..., int(idx)]), cmap='gray')
                if args.config['model']['parameter_estimation']:
                    plt.imshow(sigma_ensemble_plot[..., int(idx)], vmin=0, vmax=iscale, cmap='inferno',
                               alpha=0.5)
                else:
                    plt.imshow(std_ensemble_plot[..., int(idx)], vmin=0, vmax=iscale, cmap='inferno', alpha=0.5)
                plot_bounding_box_plt(plt.gca(), labels_for_slice, offset_box)
                plt.colorbar()
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_refsigmalabel.png'))
                plt.close()

            # plot w/o label
            plt.imshow(np.abs(y.mean(axis=0)[..., int(idx)]), cmap='gray')
            if args.config['model']['parameter_estimation']:
                plt.imshow(sigma_ensemble_plot[..., int(idx)], vmin=0, vmax=iscale, cmap='inferno', alpha=0.5)
            else:
                plt.imshow(std_ensemble_plot[..., int(idx)], vmin=0, vmax=iscale, cmap='inferno', alpha=0.5)
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_ensemble.png'))
            plt.close()

            # plot error map to reference
            plt.imshow(np.abs(ref[..., int(idx)] - y.mean(axis=0)[..., int(idx)]), cmap='inferno')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_errormap.png'))
            plt.close()

            # plot recon
            plt.imshow(np.abs(y.mean(axis=0)[..., int(idx)]), cmap='gray')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_recon.png'))
            plt.close()

            if args.config['model']['parameter_estimation']:
                plt.imshow(aleatoric_uncertainty[..., int(idx)]/np.amax(aleatoric_uncertainty), vmin=0, vmax=iscale, cmap='inferno')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_aleatoric_uncertainty.png'))
                plt.close()
                plt.imshow(epistemic_uncertainty[..., int(idx)]/np.amax(epistemic_uncertainty), vmin=0, vmax=iscale, cmap='inferno')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_epistemic_uncertainty.png'))
                plt.close()
                plt.imshow(var[..., int(idx)]/np.amax(var), vmin=0, vmax=iscale, cmap='inferno')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_var.png'))
                plt.close()
                plt.imshow(sigma_ensemble_plot[..., int(idx)], vmin=0, vmax=iscale, cmap='inferno', alpha=0.5)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_sigma.png'))
            else:
                plt.imshow(std_ensemble_plot[..., int(idx)]/np.amax(std_ensemble_plot), vmin=0, vmax=0.6, cmap='inferno')
                plt.xticks([])
                plt.yticks([])
                plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_epistemic_uncertainty.png'))
                plt.close()

            # plot reference
            plt.imshow(np.abs(ref[..., int(idx)]), cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(plot_dir, subject + '_' + f'{int(idx):02d}' + '_ref.png'))
            plt.close()



def main(args):
    logging.info(args)

    # general
    plb.Path(args.config['exp_dir']).mkdir(parents=True, exist_ok=True)
    plb.Path(args.config['result_dir']).mkdir(parents=True, exist_ok=True)

    # data module
    dm = FastMriDataModule(args.config)
    dm.setup()

    # define model
    m = EnsembleModel(args.config)

    # setup wandb logging
    run_name = args.experiment + '_' + str(args.task)
    wandb_logger = WandbLogger(project='ensemble',  # entity='lab-midas',
                               name=run_name,
							   tags=args.config['wandb_tags'],
							   offline=args.config['wandb_offline'],
							   save_dir=args.config['result_dir'])
    tb_logger = TensorBoardLogger(str(plb.Path(args.config['result_dir'])/'tensorboard'), 
                                  name=run_name)
    loggers = [tb_logger, wandb_logger]
    
    trainer = pl.Trainer(gpus=[args.config['gpu']], 
                         max_epochs=args.config['max_epochs'],
                         logger=loggers, default_root_dir=args.config['result_dir'],
                         callbacks=[EarlyStopping(monitor='val_loss', check_finite=True, patience=200)],
						 weights_save_path=plb.Path(os.path.join(args.config['result_dir'], 'checkpoints', run_name)))
    trainer.fit(m, dm)


if __name__ == '__main__':
    # args: config file, experiment
    args = merlinpy.Experiment()
    args.parse()
    if args.debug:
        args.config['wandb_offline'] = True
        args.config['num_workers'] = 0
    
    pl.utilities.seed.seed_everything(args.seed)

    if args.predict:
        main_predict(args)
    elif args.train:
        main(args)