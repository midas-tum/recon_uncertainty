import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import sys
sys.path.append('/homes/khammern/medic02/projects/fastmri_challenge/reconstruction/')

import os
import imageio
import pandas as pd
import numpy as np
import torch
import merlinth.mytorch.utils as utils
import medutils

class LinePlotSequence(object):
    def __init__(self, path, name, line_names, xlabel='epoch'):
        self._name = name
        self._path = path

        self._xlabel = xlabel
        self._ylabel = line_names

        self._lines = {}
        for l in line_names:
            self._lines[l] = ([], [])

        if not os.path.exists(self._path):
            os.makedirs(self._path)
            
    def add(self, x, y_dict):
        for line_name in y_dict.keys():
            assert line_name in self._lines.keys()
            self._lines[line_name][0].append(x)
            self._lines[line_name][1].append(y_dict[line_name])

    def save(self):
        with PdfPages(os.path.join(self._path, '{}.pdf'.format(self._name))) as pdf:
            num_pages = len(self._lines[self._ylabel[0]][1][0])
            num_pairs = self._lines[self._ylabel[0]][1][0][0].size
            num_lines = len(self._lines)

            #print(f'num_pages={num_pages} num_pairs={num_pairs} num_lines={num_lines}')

            for page in range(num_pages):
                fig, ax = plt.subplots(num_pairs, num_lines, figsize=(2*num_lines, 2*num_pairs))
                for lidx in range(num_lines):
                    val_x, val_y = self._lines[self._ylabel[lidx]]
                    val_y = np.array(val_y)

                    for pidx in range(num_pairs):
                        ax[pidx, lidx].set_ylabel(f"{self._ylabel[lidx]} {pidx+1}")
                        ax[pidx, lidx].plot(val_x, val_y[:, page, pidx], '-o', linewidth=1, markersize=1.2)

                fig.tight_layout()
                pdf.savefig(fig, dpi=100)
                plt.close(fig)

class LinePlot(object):
    def __init__(self, path, name, line_names, ylabel=None, xlabel='epoch'):
        self._name = name
        self._path = path

        self._xlabel = xlabel
        self._ylabel = ylabel

        self._lines = {}
        for l in line_names:
            self._lines[l] = ([], [])

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def add(self, x, y, line_name):
        assert line_name in self._lines.keys()

        self._lines[line_name][0].append(x)
        self._lines[line_name][1].append(y)

    def save(self, single_figure=True, crop=True):
        if single_figure:
            # create the plot
            fig, ax = plt.subplots(1,1)

            y_min = np.Inf
            y_max = -np.Inf
            for n, xy in self._lines.items():
                p = ax.plot(xy[0], xy[1], '-o', label=n, linewidth=1, markersize=1.2)
                # add a color annotation
                plt.text(xy[0][-1]+0.2, xy[1][-1], '{:.3f}'.format(xy[1][-1]), fontsize=12, color=p[0].get_color())

                y_min = min(y_min, np.quantile(xy[1], .2))
                y_max = max(y_max, np.quantile(xy[1], .9))

            if crop:
                ax.set_ylim(.9*y_min, 1.1*y_max)
            ax.set_xlabel(self._xlabel)
            ax.set_ylabel(self._ylabel)
            ax.set_title(self._name)
            ax.legend()
            plt.grid(True)
            fig.savefig(os.path.join(self._path, '{}.pdf'.format(self._name)))
            plt.close(fig)
        else:
            with PdfPages(os.path.join(self._path, '{}.pdf'.format(self._name))) as pdf:
                for n, xy in self._lines.items():
                    fig, ax = plt.subplots(1,1)
                    p = ax.plot(xy[0], xy[1], '-o', label=n, linewidth=1, markersize=1.2)
                    # add a color annotation
                    plt.text(xy[0][-1]+0.2, xy[1][-1], '{:.3f}'.format(xy[1][-1]), fontsize=12, color=p[0].get_color())

                    ax.set_ylim(.9*np.quantile(xy[1], .2), 1.1*np.quantile(xy[1], .8))
                    ax.set_xlabel(self._xlabel)
                    ax.set_ylabel(self._ylabel)
                    ax.set_title('{}-{}'.format(self._name, n))
                    ax.legend()
                    plt.grid(True)

                    pdf.savefig(fig, dpi=100)
                    plt.close(fig)

    def get_series(self, line_name):
        assert line_name in self._lines.keys()
        xy = self._lines[line_name]
        return pd.Series(xy[1], index=xy[0])


class SingleLinePlot(object):
    def __init__(self, path, name, line_names, ylabel=None, xlabel=None):
        self._name = name
        self._path = path

        self._xlabel = xlabel
        self._ylabel = ylabel

        self._lines = {}
        for l in line_names:
            self._lines[l] = ([], [])

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def add(self, x, y, line_name):
        assert line_name in self._lines.keys()

        self._lines[line_name][0] = x
        self._lines[line_name][1] = y

    def save(self, single_figure=True):
        if single_figure:
            # create the plot
            fig, ax = plt.subplots(1,1)

            y_min = np.Inf
            y_max = -np.Inf
            for n, xy in self._lines.items():
                p = ax.plot(xy[0], xy[1], '-o', label=n, linewidth=1, markersize=1.2)
                # add a color annotation
                plt.text(xy[0][-1]+0.2, xy[1][-1], '{:.3f}'.format(xy[1][-1]), fontsize=12, color=p[0].get_color())

                y_min = min(y_min, np.quantile(xy[1], .2))
                y_max = max(y_max, np.quantile(xy[1], .9))

            ax.set_ylim(.9*y_min, 1.1*y_max)
            ax.set_xlabel(self._xlabel)
            ax.set_ylabel(self._ylabel)
            ax.set_title(self._name)
            ax.legend()
            plt.grid(True)
            fig.savefig(os.path.join(self._path, '{}.pdf'.format(self._name)))
            plt.close(fig)
        else:
            with PdfPages(os.path.join(self._path, '{}.pdf'.format(self._name))) as pdf:
                for n, xy in self._lines.items():
                    fig, ax = plt.subplots(1,1)
                    p = ax.plot(xy[0], xy[1], '-o', label=n, linewidth=1, markersize=1.2)
                    # add a color annotation
                    plt.text(xy[0][-1]+0.2, xy[1][-1], '{:.3f}'.format(xy[1][-1]), fontsize=12, color=p[0].get_color())

                    ax.set_ylim(.9*np.quantile(xy[1], .2), 1.1*np.quantile(xy[1], .8))
                    ax.set_xlabel(self._xlabel)
                    ax.set_ylabel(self._ylabel)
                    ax.set_title('{}-{}'.format(self._name, n))
                    ax.legend()
                    plt.grid(True)

                    pdf.savefig(fig, dpi=100)
                    plt.close(fig)

    def get_series(self, line_name):
        assert line_name in self._lines.keys()
        xy = self._lines[line_name]
        return pd.Series(xy[1], index=xy[0])


class LineSequencePlot(object):
    def __init__(self, path, name, line_names, xlabel='epoch'):
        self._name = name
        self._path = path

        self._xlabel = xlabel
        self._ylabel = line_names

        self._lines = {}
        for l in line_names:
            self._lines[l] = ([], [])

        if not os.path.exists(self._path):
            os.makedirs(self._path)
            
    def add(self, x, y_dict):
        for line_name in y_dict.keys():
            assert line_name in self._lines.keys()
            self._lines[line_name][0].append(x)
            self._lines[line_name][1].append(y_dict[line_name])

    def save(self):
        with PdfPages(os.path.join(self._path, '{}.pdf'.format(self._name))) as pdf:
            num_pages = len(self._ylabel)
            num_lines = self._lines[self._ylabel[0]][1][0].shape[0]

            for page in range(num_pages):
                fig, ax = plt.subplots(1,1)
                val_x, val_y = self._lines[self._ylabel[page]]
                val_y = np.array(val_y)
                for lidx in range(num_lines):
                    ax.plot(val_x, val_y[:, lidx], '-o', linewidth=1, markersize=1.2, label='{:d}'.format(lidx))
                    
                ax.set_ylabel("{}".format(self._ylabel[page]))
                ax.legend()

                fig.tight_layout()
                pdf.savefig(fig, dpi=100)
                plt.close(fig)

class ImageSequence(object):
    def __init__(self, path, name, normalize=False, complex=False):
        self._name = name
        self._path = path
        self._normalize = normalize
        self._complex = complex

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def save(self, epoch, th_sequence, name, sample_offset=0):
        epoch_path = os.path.join(self._path, '{:03d}'.format(epoch), self._name, name)
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)

        for t, batch in enumerate(th_sequence):
            # get the numpy batch
            batch = batch.detach().cpu()
            if self._complex:
                batch = utils.torch_to_complex_abs_numpy(batch)
            else:
                batch = utils.torch_to_numpy(batch)

            for s in range(batch.shape[0]):
                img = batch[s]
                # post-process mri images
                img = medutils.mri.removePEOversampling(img, axes=(0,1))
                img = medutils.visualization.flip(img)
                #img = medutils.visualization.contrastStretching(img, 0.004)

                if self._normalize:
                    # normalize image
                    img -= img.min()
                    img /= img.max()
                img = np.clip(img, 0, 1) * 255
                imageio.imwrite(os.path.join(epoch_path, 's{:03d}_t{:02d}.png'.format(s + sample_offset, t)), img.astype(np.uint8))


class FeatureSequence(object):
    def __init__(self, path, name, normalize=False):
        self._name = name
        self._path = path
        self._normalize = normalize

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def save(self, epoch, th_sequence, name, sample_offset=0):
        epoch_path = os.path.join(self._path, '{:03d}'.format(epoch), self._name, name)
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)

        for t, batch in enumerate(th_sequence):
            # get the numpy batch
            batch = batch.detach().cpu().numpy()
            for s in range(batch.shape[0]):
                img = np.hstack([b for b in batch[s]])
                # normalize image
                img -= img.min()
                img /= 1. if img.max() == 0 else img.max()
                img = np.clip(img, 0, 1) * 255
                imageio.imwrite(os.path.join(epoch_path, 's{:03d}_t{:02d}.png'.format(s + sample_offset, t)), img.astype(np.uint8))


class KernelFunctionSequence(object):
    def __init__(self, path, name):
        self._name = name
        self._path = path

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def make_img(self, np_tensor, padding=2):
        if np_tensor.ndim != 3:
            raise RuntimeError('only 3D tensors are supported')

        if np_tensor.shape[0] == 1:
            return np_tensor[0]
        elif np_tensor.shape[0] == 3:
            return np.transpose(np_tensor, (1, 2, 0))
        else:
            nrow = int(np.ceil(np.sqrt(np_tensor.shape[0])))
            nmaps = np_tensor.shape[0]
            xmaps = min(nrow, nmaps)
            ymaps = int(np.ceil(float(nmaps) / xmaps))
            height, width = int(np_tensor.shape[1] + padding), int(np_tensor.shape[2] + padding)
            grid = np.zeros((height * ymaps + padding, width * xmaps + padding))
            k = 0
            for y in range(ymaps):
                for x in range(xmaps):
                    if k >= nmaps:
                        break
                    grid[y*height+padding:y*height+padding+height-padding, x*width+padding:x*width+padding+width-padding] = np_tensor[k]
                    k = k + 1
        return grid

    def save(self, epoch, th_sequence):
        epoch_path = os.path.join(self._path, '{:03d}'.format(epoch))
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)

        with PdfPages(os.path.join(epoch_path, '{}.pdf'.format(self._name))) as pdf:
            for s, pair in enumerate(th_sequence):
                if pair is None:
                    break
                # get the kernel function pair
                kernels = pair[0]
                if isinstance(kernels, dict):
                    names = list(kernels.keys())
                    plot_names = True
                    kernels = kernels.values()
                else:
                    plot_names = False
                num_k = len(kernels)
                np_kernels = [k.detach().cpu().numpy() for k in kernels]
                if pair[1] is not None:
                    np_x = pair[1][0].detach().cpu().numpy()
                    np_phi = pair[1][1].detach().cpu().numpy()
                else:
                    np_phi = None
                num_lines = max([k.shape[0] for k in np_kernels])
                fig, ax = plt.subplots(num_lines, (np_phi is not None) + num_k, figsize=(2+2*num_k, 2*num_lines))
                ax = np.reshape(ax, (-1, (np_phi is not None) + num_k))
                for i in range(num_lines):
                    # first plot the kernel
                    for j in range(num_k):
                        if i < np_kernels[j].shape[0]:
                            np_k_img = self.make_img(np_kernels[j][i])
                            np_k_img -= np_k_img.min()
                            np_k_img /= np_k_img.max()
                            ax[i,j].imshow(np_k_img, interpolation="nearest", cmap="gray",vmin=0,vmax=1)
                            ax[i,j].get_xaxis().set_ticks([])
                            ax[i,j].get_yaxis().set_ticks([])
                        else:
                            ax[i,j].axis('off')
                        if j == 0:
                            ax[i,j].set_ylabel("{}".format(i+1))
                        if plot_names and i == 0:
                            ax[i,j].set_title(f'{names[j]}')

                    # then the corresponding activation function
                    if np_phi is not None:
                        if i < np_phi.shape[0]:
                            ax[i,num_k].plot(np_x[i], np_phi[i], linewidth=2.0)
                        else:
                            ax[i,num_k].axis('off')

                fig.tight_layout()
                pdf.savefig(fig, dpi=100)
                plt.close(fig)


class HistSequence(object):
    def __init__(self, path, name):
        self._name = name
        self._path = path

        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def save(self, epoch, sequence):
        epoch_path = os.path.join(self._path, '{:03d}'.format(epoch))
        if not os.path.exists(epoch_path):
            os.makedirs(epoch_path)

        with PdfPages(os.path.join(epoch_path, '{}.pdf'.format(self._name))) as pdf:
            for s, histograms in enumerate(sequence):
                # get the histograms
                fig, ax = plt.subplots(len(histograms), 1, figsize=(2, 2*len(histograms)))
                if len(histograms) == 1:
                    ax = [ax]
                if isinstance(histograms, dict):
                    for i, h in enumerate(histograms.keys()):
                        bin_edges = histograms[h][1]
                        bins = histograms[h][0]
                        ax[i].plot(bin_edges[:-1], bins / max(bins), '-o', linewidth=1, markersize=1.2)
                        ax[i].set_xlim(min(bin_edges), max(bin_edges))
                        ax[i].set_ylabel("{}".format(h))
                else:
                    for i, h in enumerate(histograms):
                        # first plot the kernel
                        bin_edges = h[1]
                        ax[i].plot(bin_edges[:-1], h[0] / max(h[0]), '-o', linewidth=1, markersize=1.2)
                        ax[i].set_xlim(min(bin_edges), max(bin_edges))
                        ax[i].set_ylabel("{}".format(i+1))

                fig.tight_layout()
                pdf.savefig(fig, dpi=100)
                plt.close(fig)
