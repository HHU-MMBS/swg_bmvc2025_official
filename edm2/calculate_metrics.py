# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Calculate evaluation metrics (FID and FD_DINOv2)."""

import os
from typing import Dict, List
import click
import tqdm
import pickle
import numpy as np
import scipy.linalg
import torch
import PIL.Image

import wandb
import dnnlib
from torch_utils import distributed as dist
from torch_utils import misc
import pandas as pd
from pathlib import Path

from typing import Tuple
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.image.inception import InceptionScore
from torchmetrics import Metric
from torchvision.transforms.v2.functional import rgb_to_grayscale
from training.dataset import get_dataset, ImageFolderDataset
# from generate_images import generate_images

# Epsilon for numerical stability in divisions
_EPS = torch.finfo(torch.float32).eps

# Abstract base class for feature detectors.
class Detector:
    def __init__(self, feature_dim):
        self.feature_dim = feature_dim

    def __call__(self, x): # NCHW, uint8, 3 channels => NC, float32
        raise NotImplementedError # to be overridden by subclass

#----------------------------------------------------------------------------
# InceptionV3 feature detector.
# This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

class InceptionV3Detector(Detector):
    def __init__(self):
        super().__init__(feature_dim=2048)
        url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
        with dnnlib.util.open_url(url, verbose=True) as f:
            self.model = pickle.load(f)

    def __call__(self, x):
        return self.model.to(x.device)(x, return_features=True)

#----------------------------------------------------------------------------
# DINOv2 feature detector.
# Modeled after https://github.com/layer6ai-labs/dgm-eval

class DINOv2Detector(Detector):
    def __init__(self, resize_mode='torch'):
        super().__init__(feature_dim=1024)
        self.resize_mode = resize_mode
        import warnings
        warnings.filterwarnings('ignore', 'xFormers is not available')
        from dnnlib import check_internet
        if not check_internet():
            print(f"Warning: No internet connection detected. Attempting to download DINOv2 model from local cache.")
        torch.hub.set_dir(dnnlib.make_cache_dir_path('torch_hub'))
        self.model = torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vitl14', trust_repo=True, verbose=False, skip_validation=True)
        self.model.eval().requires_grad_(False)

    def __call__(self, x):
        # Resize images.
        if self.resize_mode == 'pil': # Slow reference implementation that matches the original dgm-eval codebase exactly.
            device = x.device
            x = x.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            x = np.stack([np.uint8(PIL.Image.fromarray(xx, 'RGB').resize((224, 224), PIL.Image.Resampling.BICUBIC)) for xx in x])
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
        elif self.resize_mode == 'torch': # Fast practical implementation that yields almost the same results.
            x = torch.nn.functional.interpolate(x.to(torch.float32), size=(224, 224), mode='bicubic', antialias=True)
        else:
            raise ValueError(f'Invalid resize mode "{self.resize_mode}"')

        # Adjust dynamic range.
        x = x.to(torch.float32) / 255
        x = x - misc.const_like(x, [0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        x = x / misc.const_like(x, [0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)

        # Run DINOv2 model.
        return self.model.to(x.device)(x)

#----------------------------------------------------------------------------
# Metric specifications.

metric_specs = {
    'fid':          dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=InceptionV3Detector)),
    'fd_dinov2':    dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=DINOv2Detector)),
    'inception':    dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name=InceptionV3Detector)),
    'color':       dnnlib.EasyDict(detector_kwargs=dnnlib.EasyDict(class_name="ColorSpaceStats", channel=1)),
}

#----------------------------------------------------------------------------
# Get feature detector for the given metric.

_detector_cache = dict()

def get_detector(metric, verbose=True):
    # Lookup from cache.
    if metric in _detector_cache:
        return _detector_cache[metric]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        dist.barrier()

    # Construct detector.
    kwargs = metric_specs[metric].detector_kwargs
    if verbose:
        name = kwargs.class_name.split('.')[-1] if isinstance(kwargs.class_name, str) else kwargs.class_name.__name__
        dist.print0(f'Setting up {name}...')
    detector = dnnlib.util.construct_class_by_name(**kwargs)
    _detector_cache[metric] = detector

    # Other ranks follow.
    if dist.get_rank() == 0:
        dist.barrier()
    return detector

#----------------------------------------------------------------------------
# Load feature statistics from the given .pkl or .npz file.

def load_stats(path, verbose=True):
    if verbose:
        print(f'Loading feature statistics from {path} ...')
    with dnnlib.util.open_url(path, verbose=verbose) as f:
        if path.lower().endswith('.npz'): # backwards compatibility with https://github.com/NVlabs/edm
            return {'fid': dict(np.load(f))}
        return pickle.load(f)

#----------------------------------------------------------------------------
# Save feature statistics to the given .pkl file.

def save_stats(stats, path, verbose=True):
    if verbose:
        print(f'Saving feature statistics to {path} ...')
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True, mode=0o777)
    with open(path, 'wb') as f:
        pickle.dump(stats, f)


def calculate_and_save_metrics(metrics, ref_path, outdir, num_images, max_batch_size, use_wandb=False, num_workers=4):
    """
    Calculate and save evaluation metrics for generated images.
    Args:
        metrics (list): List of metric names to calculate (e.g., ['fid', 'fd_dinov2']).
        ref_path (str): Path to the reference dataset for metric calculation.
        outdir (str): Directory where generated images and results will be saved.
        num_images (int): Number of images to evaluate.
        max_batch_size (int): Maximum batch size for metric calculation.
        save_tar (bool, optional): Whether to read images from a tar file. Defaults to False.
        use_wandb (bool, optional): Whether to log metrics to Weights & Biases. Defaults to False.
    Returns:
        dict: A dictionary containing the calculated metrics.
    Raises:
        NotImplementedError: If an unsupported metric is specified in the `metrics` list.
    """
    if num_images > 50000:
        num_images = 50000

    metric_kwargs = {'image_path': outdir,
                        'ref_path': ref_path,
                        'metrics': metrics,
                        'num_images': num_images,
                        'seed': 0,
                        'max_batch_size': min(min(1024, max_batch_size), num_images),
                        'num_workers': num_workers,}
    
    results = calc_python(**metric_kwargs)
    
    # only rank 0 saves the results
    if dist.get_rank() == 0:
        out_dir_samples = Path(outdir)
        if "samples" in out_dir_samples.name:
            out_path_csv = Path(f"{out_dir_samples.parent}.csv") 
        else:
            out_path_csv = Path(f"{out_dir_samples}.csv") 
        # Save results to CSV
        print('Saving results to ', out_path_csv)
        pd.DataFrame(results, index=[0]).to_csv(out_path_csv)
    
    # Since this function is now called only on rank 0, remove distributed logic
    if use_wandb:
        log = {}
        for metric in metrics:
            if metric == 'fid':
                name = "FID"
            elif metric == 'fd_dinov2':
                name = "FDD"
            else:
                raise NotImplementedError(f'Invalid metric: {metric}')
            log[name] = results[metric]
        wandb.config.update(log)
    
    return results

#----------------------------------------------------------------------------
# Calculate feature statistics for the given image batches
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_iterable(
    image_iter,                         # Iterable of image batches: NCHW, uint8, 3 channels.
    metrics     = ['fid', 'fd_dinov2'], # Metrics to compute the statistics for.
    verbose     = True,                 # Enable status prints?
    dest_path   = None,                 # Where to save the statistics. None = do not save.
    device      = torch.device('cuda'), # Which compute device to use.
):
    # Initialize.
    num_batches = len(image_iter)
    detectors = [get_detector(metric, verbose=verbose) for metric in metrics]
    if verbose:
        dist.print0('Calculating feature statistics...')

    # Convenience wrapper for torch.distributed.all_reduce().
    def all_reduce(x):
        x = x.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(x)
        return x

    # Return an iterable over the batches.
    class StatsIterable:
        def __len__(self):
            return num_batches

        def __iter__(self):
            state = [dnnlib.EasyDict(metric=metric, detector=detector) for metric, detector in zip(metrics, detectors)]
            for s in state:
                s.cum_mu = torch.zeros([s.detector.feature_dim], dtype=torch.float64, device=device)
                s.cum_sigma = torch.zeros([s.detector.feature_dim, s.detector.feature_dim], dtype=torch.float64, device=device)
            cum_images = torch.zeros([], dtype=torch.int64, device=device)

            # Loop over batches.
            for batch_idx, images in enumerate(image_iter):
                if isinstance(images, dict) and 'images' in images: # dict(images)
                    images = images['images']
                elif isinstance(images, (tuple, list)) and len(images) == 2: # (images, labels)
                    images = images[0]
                images = torch.as_tensor(images).to(device)

                # Accumulate statistics.
                if images is not None:
                    for s in state:
                        features = s.detector(images).to(torch.float64)
                        s.cum_mu += features.sum(0)
                        s.cum_sigma += features.T @ features
                    cum_images += images.shape[0]

                # Output results.
                r = dnnlib.EasyDict(stats=None, images=images, batch_idx=batch_idx, num_batches=num_batches)
                r.num_images = int(all_reduce(cum_images).cpu())
                if batch_idx == num_batches - 1:
                    assert r.num_images >= 2
                    r.stats = dict(num_images=r.num_images)
                    for s in state:
                        mu = all_reduce(s.cum_mu) / r.num_images
                        sigma = (all_reduce(s.cum_sigma) - mu.ger(mu) * r.num_images) / (r.num_images - 1)
                        r.stats[s.metric] = dict(mu=mu.cpu().numpy(), sigma=sigma.cpu().numpy())
                    if dest_path is not None and dist.get_rank() == 0:
                        save_stats(stats=r.stats, path=dest_path, verbose=False)
                yield r

    return StatsIterable()

#----------------------------------------------------------------------------
# Calculate feature statistics for the given directory or ZIP of images
# in a distributed fashion. Returns an iterable that yields
# dnnlib.EasyDict(stats, images, batch_idx, num_batches)

def calculate_stats_for_files(
    image_path,             # Path to a directory or ZIP file containing the images.
    num_images      = None, # Number of images to use. None = all available images.
    seed            = 0,    # Random seed for selecting the images.
    max_batch_size  = 64,   # Maximum batch size.
    num_workers     = 2,    # How many subprocesses to use for data loading.
    prefetch_factor = 2,    # Number of images loaded in advance by each worker.
    verbose         = True, # Enable status prints?
    **stats_kwargs,         # Arguments for calculate_stats_for_iterable().
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        dist.barrier()

    # List images.
    dataset_obj = get_dataset(image_path)
    if verbose:
        dist.print0(f'Loading {len(dataset_obj)} images from {image_path} ...')
    if num_images is not None and len(dataset_obj) < num_images:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but expected at least {num_images}')
    if len(dataset_obj) < 2:
        raise click.ClickException(f'Found {len(dataset_obj)} images, but need at least 2 to compute statistics')

    # Other ranks follow.
    if dist.get_rank() == 0:
        dist.barrier()

    # Divide images into batches.
    num_batches = max((len(dataset_obj) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(dataset_obj)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    data_loader = torch.utils.data.DataLoader(dataset_obj, batch_sampler=rank_batches,
        num_workers=num_workers, prefetch_factor=(prefetch_factor if num_workers > 0 else None))

    # Return an interable for calculating the statistics.
    return calculate_stats_for_iterable(image_iter=data_loader, verbose=verbose, **stats_kwargs)

#----------------------------------------------------------------------------
# Calculate metrics based on the given feature statistics.

def calculate_metrics_from_stats(
    stats,                          # Feature statistics of the generated images.
    ref,                            # Reference statistics of the dataset. Can be a path or URL.
    metrics = ['fid', 'fd_dinov2'], # List of metrics to compute.
    verbose = True,                 # Enable status prints?
):
    if isinstance(ref, str):
        ref = load_stats(ref, verbose=verbose)
    results = dict()
    for metric in metrics:
        if metric not in stats or metric not in ref:
            if verbose:
                dist.print0(f'No statistics computed for {metric} -- skipping.')
            continue
        if verbose:
            dist.print0(f'Calculating {metric}...')
        m = np.square(stats[metric]['mu'] - ref[metric]['mu']).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(stats[metric]['sigma'], ref[metric]['sigma']), disp=False)
        value = float(np.real(m + np.trace(stats[metric]['sigma'] + ref[metric]['sigma'] - s * 2)))
        results[metric] = value
        if verbose:
            dist.print0(f'{metric} = {value:g}')
    return results

#----------------------------------------------------------------------------
# Parse a comma separated list of strings.

def parse_metric_list(s):
    if s == 'none':
        return None
    metrics = s if isinstance(s, list) else s.split(',')
    for metric in metrics:
        if metric not in metric_specs:
            raise click.ClickException(f'Invalid metric "{metric}"')
    return metrics

#----------------------------------------------------------------------------
# Main command line.

@click.group()
def cmdline():
    """Calculate evaluation metrics (FID and FD_DINOv2).

    Examples:

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 sample_edm2.py \\
        --preset=edm2-img512-xxl-guid-fid --outdir=out --subdirs --seeds=0-49999

    \b
    # Calculate metrics for a random subset of 50000 images in out/
    python calculate_metrics.py calc --images=out \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl

    \b
    # Calculate metrics directly for a given model without saving any images
    torchrun --standalone --nproc_per_node=8 calculate_metrics.py gen \\
        --net=https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.130.pkl \\
        --ref=https://nvlabs-fi-cdn.nvidia.com/edm2/dataset-refs/img512.pkl \\
        --seed=123456789

    \b
    # Compute dataset reference statistics
    python calculate_metrics.py ref --data=datasets/my-dataset.zip \\
        --dest=fid-refs/my-dataset.pkl
    """

#----------------------------------------------------------------------------
# 'calc' subcommand.

@cmdline.command()
@click.option('--images', 'image_path',     help='Path to the images', metavar='PATH|ZIP',                  type=str, required=True)
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to use', metavar='INT',                  type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                     help='Random seed for selecting the images', metavar='INT',     type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT',     type=click.IntRange(min=0), default=2, show_default=True)

def calc(ref_path, metrics, **opts):
    """Calculate metrics for a given set of images."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    stats_iter = calculate_stats_for_files(metrics=metrics, **opts)
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    if dist.get_rank() == 0:
        results = calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)
        # save dict out as csv with pandas
        img_path = Path(opts["image_path"])
        if "samples" in img_path.name:
            out_path_csv = Path(f"{img_path.parent}.csv") 
        else:
            out_path_csv = Path(f"{img_path}.csv")
        pd.DataFrame(results, index=[0]).to_csv(out_path_csv)
    dist.barrier()
    return results

def calc_python(ref_path, metrics, **opts):
    """Identical to "calc", but without click."""
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    stats_iter = calculate_stats_for_files(metrics=metrics, **opts)
    dist.print0("Iterating over stats")
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0), desc="Calculating FD metrics"):
        pass
    dist.print0("Iterated over stats")
    if dist.get_rank() == 0:
        results = calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)
    else:
        results = None
    dist.barrier()
    return results

#----------------------------------------------------------------------------
# 'gen' subcommand.

@cmdline.command()
@click.option('--net',                      help='Network pickle filename', metavar='PATH|URL',             type=str, required=True)
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to generate', metavar='INT',             type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--seed',                     help='Random seed for the first image', metavar='INT',          type=int, default=0, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=32, show_default=True)

def gen(net, ref_path, metrics, num_images, seed, **opts):
    """Calculate metrics for a given model using default sampler settings."""
    dist.init()
    if dist.get_rank() == 0:
        ref = load_stats(path=ref_path) # do this first, just in case it fails
    image_iter = generate_images(net=net, seeds=range(seed, seed + num_images), **opts)
    stats_iter = calculate_stats_for_iterable(image_iter, metrics=metrics)
    for r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass
    if dist.get_rank() == 0:
        calculate_metrics_from_stats(stats=r.stats, ref=ref, metrics=metrics)
    dist.barrier()

#----------------------------------------------------------------------------
# 'ref' subcommand.

@cmdline.command()
@click.option('--data', 'image_path',       help='Path to the dataset', metavar='PATH|ZIP',             type=str, required=True)
@click.option('--dest', 'dest_path',        help='Destination file', metavar='PKL',                     type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',          type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                   type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--workers', 'num_workers',   help='Subprocesses to use for data loading', metavar='INT', type=click.IntRange(min=0), default=2, show_default=True)

def ref(**opts):
    """Calculate dataset reference statistics for 'calc' and 'gen'."""
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    stats_iter = calculate_stats_for_files(**opts)
    for _r in tqdm.tqdm(stats_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------
import concurrent.futures

@cmdline.command()
@click.option('--outdir',                   help='Directory for evaluated images', metavar='PATH',   type=str, required=True)
def validate(outdir, max_workers=16):
    """
    Validate images in a dataset in parallel.

    Args:
        outdir (str): Directory containing the dataset (used by get_dataset).
        max_workers (int, optional): The maximum number of worker threads.
            If None, defaults to sensible value.

    Returns:
        list: A list of dictionaries, each containing info about a failed image.
    """
    dataset = get_dataset(outdir)
    dataset_size = len(dataset)

    if dataset_size < 49999:
        print(f"Warning: Found {dataset_size} images, but expected at least 50000 in {outdir}")

    print(f"Validating {dataset_size} images using {max_workers} worker threads...")

    failed_images = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_validate_single_image, dataset, i): i for i in range(dataset_size)}

        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=dataset_size):
            index = futures[future]
            try:
                idx_returned, success, path, error = future.result()

                if not success:
                    failed_images.append({'index': idx_returned, 'path': path, 'error': error})
                    # Print error immediately for visibility, but avoid excessive Index messages
                    if path and "Index" not in path:
                         print(f"\nError loading image {path} (index {idx_returned}): {error}")
                    else:
                         print(f"\nError loading image at index {idx_returned}: {error}")

            except Exception as exc:
                # Catch unexpected errors from the worker itself
                print(f'\nWorker for index {index} generated an unhandled exception: {exc}')
                failed_images.append({'index': index, 'path': 'Unknown (worker error)', 'error': exc})

    print("\nValidation complete.")
    if failed_images:
        print(f"Found {len(failed_images)} failed images.")

    return failed_images

def _validate_single_image(dataset, index):
    """Worker function to validate a single image by index."""
    path = dataset._image_fnames[index] if isinstance(dataset, ImageFolderDataset) else f"Index {index}"
    
    try:

        img = dataset[index] # Attempt to load and decode the image

        return index, True, path, None # Success

    except Exception as e:
        # If loading fails, report failure and error

        return index, False, path, e # Failure


EVAL_STORAGE = "./samples/eval_metrics_all"

#----------------------------------------------------------------------------
@cmdline.command()
@click.option('--ref', 'ref_path',          help='Dataset reference statistics ', metavar='PKL|NPZ|URL',    type=str, required=True)
@click.option('--outdir',                   help='Directory for evaluated images', metavar='PATH',   type=str, required=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST',              type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--num', 'num_images',        help='Number of images to evaluate', metavar='INT',             type=click.IntRange(min=2), default=50000, show_default=True)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                       type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--save_path',             help='Path to save the metrics', metavar='PATH',   type=str, default=EVAL_STORAGE, show_default=True)
@click.option('--recalculate', help='Recalculate metrics even if they exist', is_flag=True, default=False)
def fill(ref_path, outdir, metrics, num_images, max_batch_size, save_path, recalculate):
    """Calculate metrics for a given set of images. Skips already calculated metrics."""
    # torch.multiprocessing.set_start_method('spawn')
    # dist.init()
    fill_missing_metrics(ref_path=ref_path,
                         outdir=outdir,
                         save_path=save_path,
                         metrics=metrics,
                         num_images=num_images,
                         max_batch_size=max_batch_size,
                         recalculate=recalculate)
    # # Destroy the process group
    # dist.destroy_process_group()

def fill_missing_metrics(ref_path: str,
                         outdir: str,
                         save_path: str = EVAL_STORAGE,
                         metrics: List[str] = ['fd_dinov2'],
                         max_batch_size = 128,
                         num_images = 50000,
                         recalculate = False):
    def construct_new_path_short(original_path_str, base_path_str):
        original_path = Path(original_path_str)
        base_path = Path(base_path_str)
        parts = original_path.parts
        in_index = -1
        for i, part in enumerate(parts):
            if "IN" in part:
                in_index = i
                break
        if in_index == -1:
            return str(base_path)
        suffix_parts = parts[in_index:]
        suffix_path_obj = Path(*suffix_parts)
        new_path = base_path / suffix_path_obj
        return str(new_path)
    
    # replace inception with internally used inc_mean
    if 'inception' in metrics:
        inception_ind = metrics.index('inception')
        metrics[inception_ind] = 'inc_mean'

    if 'color' in metrics:
        color_ind = metrics.index('color')
        metrics[color_ind] = 'contrast'

    # Try to find .csv file with metrics that could've been already calculated
    out_dir_samples = Path(outdir)
    if "samples" in out_dir_samples.name:
        out_path_csv = Path(f"{out_dir_samples.parent}.csv") 
    else:
        out_path_csv = Path(f"{out_dir_samples}.csv") 
    dist.print0(f"Original metrics will be read from {out_path_csv}")
    results = {}
    if out_path_csv.exists():
        # Read the csv file and check if the metrics are already calculated
        results = pd.read_csv(out_path_csv).to_dict(orient='records')[0]
        results.pop("Unnamed: 0", None)

    new_csv_path = construct_new_path_short(out_path_csv, save_path)

    if os.path.exists(new_csv_path):
        dist.print0(f"Some metrics were calculated previously. Reading...")
        results_new = pd.read_csv(new_csv_path).to_dict(orient='records')[0]
        results_new.pop("Unnamed: 0", None)
        results.update(results_new)

    if not recalculate:
        if len(results) > 0:
            dist.print0(f"Found existing metrics: {results.keys()}, they will be skipped.")
        # From metrics list remove the entries that already exist in the results dict
        metrics = [metric for metric in metrics if metric not in results]

    if "inc_mean" in metrics:
        # Calculate inception score
        inc_score = calc_inception_score(img_path=outdir,
                                          batch_size=min(max_batch_size, num_images))
        results.update(inc_score)
        metrics.remove("inc_mean")
        dist.print0(f"Calculated Inception Score: {inc_score}")

    if "contrast" in metrics:
        # Calculate color score
        color_score = calc_color_score(img_path=outdir,
                                        batch_size=min(max_batch_size, num_images))
        results.update(color_score)
        metrics.remove("contrast")
        dist.print0(f"Calculated Color Score: {color_score}")

    if len(metrics) != 0:
        # Calculate the remaining metrics (FID and FD_DINOv2)
        fd_scores = calc_python(image_path=outdir,
                                   ref_path=ref_path,
                                   metrics=metrics,
                                   num_images=num_images,
                                   max_batch_size=min(max_batch_size, num_images),
                                   seed = 0)
        results.update(fd_scores)
        dist.print0(f"Calculated metrics: {fd_scores}")
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(new_csv_path), exist_ok=True, mode=0o777)
    dist.print0(f"Saving metrics to {new_csv_path}")
    pd.DataFrame(results, index=[0]).to_csv(new_csv_path)

    return results

def calc_inception_score(
    img_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    dataset = get_dataset(img_path)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    is_distributed = world_size > 1
    
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            # drop_last=True is important for torchmetrics sync across ranks
            drop_last=True
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler, # Use the distributed sampler if applicable
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False)

    is_score = InceptionScore().to(device)

    for batch in tqdm.tqdm(dataloader, disable=rank != 0, desc="Calculating Inception Score"):
        batch: torch.Tensor = batch[0] if isinstance(batch, (tuple, list)) else batch
        batch = batch.byte().to(device)
        is_score.update(batch)
        
    dist.barrier()

    score = is_score.compute()
    
    result = {
        "inc_mean": score[0].item(),
        "inc_std": score[1].item(),
        }
    return result    
#----------------------------------------------------------------------------

class ColorSpaceStats(Metric):
    """Calculates color space statistics (mean and std) for RGB, HSV, or HSL channels.
    This class is designed to be used with PyTorch Lightning and can be used to compute
    metrics across multiple batches of images. It supports distributed training and
    automatically handles the reduction of metrics across multiple GPUs or nodes.
    Args:
        channel: The channel to compute the mean and std for. 0 for H, 1 for S, 2 for V or L.
        **kwargs: Additional keyword arguments for the Metric class.
    """
    def __init__(self, channel = 1, **kwargs):
        super().__init__(**kwargs)
        self.channel = channel
        self.add_state("satur_means", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("satur_stds", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("contrasts", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("num_images", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, images: torch.Tensor) -> None:
        hsv_images = ColorSpaceStats.rgb_to_hsv_torch(images)
        s_mean, s_std = ColorSpaceStats.calc_color_metrics(hsv_images, channel=self.channel)
        contrast = ColorSpaceStats.calc_contrast_metrics(images)
        self.satur_means+=s_mean.sum()
        self.satur_stds+=s_std.sum()
        self.contrasts+=contrast.sum()
        self.num_images+=images.shape[0]

    def compute(self) -> Dict[str, float]:
        return {
            "satur_mean": (self.satur_means / self.num_images).item(),
            "satur_std": (self.satur_stds / self.num_images).item(),
            "contrast": (self.contrasts / self.num_images).item()
        }

    @staticmethod
    def rgb_to_hsl_torch(rgb: torch.Tensor) -> torch.Tensor:
        """Converts batches of RGB images to HSL format.

        Input tensor shape should be (N, 3, H, W) or (N, 3).
        RGB values are assumed to be in the [0, 1] range.
        Output HSL values will be in the range:
        H: [0, 1] (representing 0-360 degrees)
        S: [0, 1]
        L: [0, 1]

        Args:
            rgb: Input RGB tensor.

        Returns:
            HSL tensor of the same shape as input.
        """
        assert rgb.shape[1] == 3, "Input tensor must have 3 channels (RGB)"
        
        # Add small epsilon to prevent division by zero
        rgb = rgb + _EPS

        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin, _ = torch.min(rgb, dim=1, keepdim=True)
        delta = cmax - cmin

        # HSL Lightness (L)
        hsl_l = (cmax + cmin) / 2.0

        # HSL Saturation (S)
        # Saturation is 0 if delta is 0 (achromatic)
        # Otherwise, formula depends on L
        denominator_s_low = (cmax + cmin) # Equivalent to 2.0 * hsl_l
        denominator_s_high = (2.0 - cmax - cmin) # Equivalent to 2.0 - 2.0 * hsl_l
        
        # Use where for conditional calculation, avoid division by zero if L=0 or L=1
        # Note: If L=0, cmax+cmin=0. If L=1, 2-cmax-cmin=0. delta is 0 in these cases too.
        is_achromatic = (delta == 0) | (hsl_l == 0) | (hsl_l == 1)
        
        hsl_s = torch.where(
            is_achromatic,
            torch.zeros_like(delta),
            torch.where(
                hsl_l <= 0.5,
                delta / denominator_s_low,
                delta / denominator_s_high
            )
        )

        # HSL Hue (H) - Calculation is the same as for HSV
        # Calculate hue based on which channel was maximum
        hue = torch.empty_like(delta)

        # Case 1: R is max
        mask_r = cmax_idx == 0
        hue[mask_r] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[mask_r]

        # Case 2: G is max
        mask_g = cmax_idx == 1
        hue[mask_g] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[mask_g]

        # Case 3: B is max
        mask_b = cmax_idx == 2
        hue[mask_b] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[mask_b]

        # Handle case where delta is zero (achromatic colors) -> hue is 0
        hue = torch.where(delta > 0, hue / 6.0, torch.zeros_like(hue)) # Normalize H to [0, 1]

        # Clamp values to ensure they are within [0, 1]
        hsl = torch.cat([hue, hsl_s, hsl_l], dim=1)
        # Clamp S and L. Hue wraps around.
        hsl[:, 1:3] = torch.clamp(hsl[:, 1:3], 0.0, 1.0)
        
        return hsl

    @staticmethod
    def hsl_to_rgb_torch(hsl: torch.Tensor) -> torch.Tensor:
        """Converts batches of HSL images to RGB format.

        Input tensor shape should be (N, 3, H, W) or (N, 3).
        HSL values are assumed to be in the range:
        H: [0, 1] (representing 0-360 degrees)
        S: [0, 1]
        L: [0, 1]
        Output RGB values will be in the [0, 1] range.

        Args:
            hsl: Input HSL tensor.

        Returns:
            RGB tensor of the same shape as input.
        """
        assert hsl.shape[1] == 3, "Input tensor must have 3 channels (HSL)"

        h, s, l = hsl[:, 0:1], hsl[:, 1:2], hsl[:, 2:3]

        # Ensure H is in [0, 1)
        h = h % 1.0
        
        # Calculate intermediate values
        # Chroma depends on Lightness
        c = (1.0 - torch.abs(2.0 * l - 1.0)) * s
        h_prime = h * 6.0
        x = c * (1.0 - torch.abs(h_prime % 2.0 - 1.0))
        m = l - c / 2.0 # Note the difference in 'm' compared to HSV

        # Determine segment and compute RGB triplet
        h_idx = torch.floor(h_prime).type(torch.int64) % 6 # Ensure index is integer and in [0, 5]

        # Create empty tensor for RGB
        rgb = torch.zeros_like(hsl)

        # Assign R, G, B based on hue sector (same logic as hsv_to_rgb)
        # Sector 0: [C, X, 0]
        mask0 = (h_idx == 0)
        rgb[:, 0:1][mask0] = c[mask0]
        rgb[:, 1:2][mask0] = x[mask0]
        rgb[:, 2:3][mask0] = 0.0

        # Sector 1: [X, C, 0]
        mask1 = (h_idx == 1)
        rgb[:, 0:1][mask1] = x[mask1]
        rgb[:, 1:2][mask1] = c[mask1]
        rgb[:, 2:3][mask1] = 0.0

        # Sector 2: [0, C, X]
        mask2 = (h_idx == 2)
        rgb[:, 0:1][mask2] = 0.0
        rgb[:, 1:2][mask2] = c[mask2]
        rgb[:, 2:3][mask2] = x[mask2]

        # Sector 3: [0, X, C]
        mask3 = (h_idx == 3)
        rgb[:, 0:1][mask3] = 0.0
        rgb[:, 1:2][mask3] = x[mask3]
        rgb[:, 2:3][mask3] = c[mask3]

        # Sector 4: [X, 0, C]
        mask4 = (h_idx == 4)
        rgb[:, 0:1][mask4] = x[mask4]
        rgb[:, 1:2][mask4] = 0.0
        rgb[:, 2:3][mask4] = c[mask4]

        # Sector 5: [C, 0, X]
        mask5 = (h_idx == 5)
        rgb[:, 0:1][mask5] = c[mask5]
        rgb[:, 1:2][mask5] = 0.0
        rgb[:, 2:3][mask5] = x[mask5]
        
        # Add lightness adjustment (m)
        rgb += m

        # Clamp to ensure output is in [0, 1]
        return torch.clamp(rgb, 0.0, 1.0)

    @staticmethod
    def rgb_to_hsv_torch(rgb: torch.Tensor) -> torch.Tensor:
        """Converts batches of RGB images to HSV format.

        Input tensor shape should be (N, 3, H, W) or (N, 3).
        RGB values are assumed to be in the [0, 1] range.
        Output HSV values will be in the range:
        H: [0, 1] (representing 0-360 degrees)
        S: [0, 1]
        V: [0, 1]

        Args:
            rgb: Input RGB tensor.

        Returns:
            HSV tensor of the same shape as input.
        """
        assert rgb.shape[1] == 3, "Input tensor must have 3 channels (RGB)"
        
        # Add small epsilon to prevent division by zero
        rgb = rgb + _EPS

        cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
        cmin, _ = torch.min(rgb, dim=1, keepdim=True)
        delta = cmax - cmin

        # HSV Value (V)
        hsv_v = cmax

        # HSV Saturation (S)
        # Saturation is 0 if cmax is 0 (black) or if delta is 0 (grayscale)
        hsv_s = torch.where(
            cmax == 0,
            torch.zeros_like(delta),
            delta / (cmax + _EPS) # Add EPS for robustness
        )

        # HSV Hue (H)
        # Calculate hue based on which channel was maximum
        hue = torch.empty_like(delta)

        mask_r = cmax_idx == 0
        mask_g = cmax_idx == 1
        mask_b = cmax_idx == 2

        # Use where for conditional calculation, handle delta=0 case explicitly
        hue = torch.where(
            delta > 0,
            torch.where(
                mask_r,
                (((rgb[:, 1:2] - rgb[:, 2:3]) / (delta + _EPS)) % 6),
                torch.where(
                    mask_g,
                    (((rgb[:, 2:3] - rgb[:, 0:1]) / (delta + _EPS)) + 2),
                    (((rgb[:, 0:1] - rgb[:, 1:2]) / (delta + _EPS)) + 4) # mask_b
                )
            ),
            torch.zeros_like(delta) # delta == 0, hue is 0
        )

        # Normalize H to [0, 1]
        hue = hue / 6.0

        hsv = torch.cat([hue, hsv_s, hsv_v], dim=1)
        # Clamp S and V. Hue wraps around.
        hsv[:, 1:3] = torch.clamp(hsv[:, 1:3], 0.0, 1.0)
        
        return hsv

    @staticmethod
    def hsv_to_rgb_torch(hsv: torch.Tensor) -> torch.Tensor:
        """Converts batches of HSV images to RGB format.

        Input tensor shape should be (N, 3, H, W) or (N, 3).
        HSV values are assumed to be in the range:
        H: [0, 1] (representing 0-360 degrees)
        S: [0, 1]
        V: [0, 1]
        Output RGB values will be in the [0, 1] range.

        Args:
            hsv: Input HSV tensor.

        Returns:
            RGB tensor of the same shape as input.
        """
        assert hsv.shape[1] == 3, "Input tensor must have 3 channels (HSV)"

        h, s, v = hsv[:, 0:1], hsv[:, 1:2], hsv[:, 2:3]

        # Ensure H is in [0, 1)
        h = h % 1.0
        
        # Calculate intermediate values
        c = v * s # Chroma
        h_prime = h * 6.0
        x = c * (1.0 - torch.abs(h_prime % 2.0 - 1.0))
        m = v - c # Note the difference in 'm' compared to HSL

        # Determine segment and compute RGB triplet
        h_idx = torch.floor(h_prime).type(torch.int64) % 6 # Ensure index is integer and in [0, 5]

        # Create empty tensor for RGB
        rgb = torch.zeros_like(hsv)

        # Assign R, G, B based on hue sector using masks
        mask0 = (h_idx == 0)
        mask1 = (h_idx == 1)
        mask2 = (h_idx == 2)
        mask3 = (h_idx == 3)
        mask4 = (h_idx == 4)
        mask5 = (h_idx == 5)

        # Sector 0: [C, X, 0]
        rgb[:, 0:1][mask0] = c[mask0]
        rgb[:, 1:2][mask0] = x[mask0]
        rgb[:, 2:3][mask0] = 0.0

        # Sector 1: [X, C, 0]
        rgb[:, 0:1][mask1] = x[mask1]
        rgb[:, 1:2][mask1] = c[mask1]
        rgb[:, 2:3][mask1] = 0.0

        # Sector 2: [0, C, X]
        rgb[:, 0:1][mask2] = 0.0
        rgb[:, 1:2][mask2] = c[mask2]
        rgb[:, 2:3][mask2] = x[mask2]

        # Sector 3: [0, X, C]
        rgb[:, 0:1][mask3] = 0.0
        rgb[:, 1:2][mask3] = x[mask3]
        rgb[:, 2:3][mask3] = c[mask3]

        # Sector 4: [X, 0, C]
        rgb[:, 0:1][mask4] = x[mask4]
        rgb[:, 1:2][mask4] = 0.0
        rgb[:, 2:3][mask4] = c[mask4]

        # Sector 5: [C, 0, X]
        rgb[:, 0:1][mask5] = c[mask5]
        rgb[:, 1:2][mask5] = 0.0
        rgb[:, 2:3][mask5] = x[mask5]
        
        # Add value adjustment (m)
        rgb += m

        # Clamp to ensure output is in [0, 1]
        return torch.clamp(rgb, 0.0, 1.0)

    @staticmethod
    def calc_color_metrics(hsv_batch: torch.Tensor, channel = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates mean and standard deviation of RGB/HSV/HSL channels.

        Args:
            hsv_batch: Input RGB/HSV/HSL tensor.
            channel: Channel to calculate mean and std over.

        Returns:
            Dictionary with mean and std for selected channel.
        """
        assert hsv_batch.shape[1] == 3, "Input tensor must have 3 channels (RGB/HSV/HSL)"
        
        # Calculate mean and std for the channel
        c_mean = hsv_batch[:, channel].mean(dim = [1,2])
        c_std = hsv_batch[:, channel].std(dim = [1,2])

        return c_mean, c_std

    @staticmethod
    def calc_contrast_metrics(rgb_batch: torch.Tensor) -> torch.Tensor:
        """Calculates contrast metrics for RGB images.

        Args:
            rgb_batch: Input RGB tensor.

        Returns:
            Contrast metric for the batch.
        """
        assert rgb_batch.shape[1] == 3, "Input tensor must have 3 channels (RGB)"
        
        grayscale = rgb_to_grayscale(rgb_batch)
        contrast = torch.std(grayscale, dim=[1, 2, 3])

        return contrast
    
#----------------------------------------------------------------------------

def calc_color_score(
    img_path: str,
    batch_size: int = 4,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    dataset = get_dataset(img_path)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    is_distributed = world_size > 1
    
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            # drop_last=True is important for torchmetrics sync across ranks
            drop_last=True
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler, # Use the distributed sampler if applicable
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=False)

    color_score = ColorSpaceStats(channel=1).to(device)

    for batch in tqdm.tqdm(dataloader, disable=rank != 0, desc="Calculating Color Scores"):
        batch: torch.Tensor = batch[0] if isinstance(batch, (tuple, list)) else batch
        batch = batch.float()/255.0 # Maybe add a check if the batch is already in [0, 1] range
        batch = batch.to(device)
        color_score.update(batch)
        
    dist.barrier()

    score = color_score.compute()
    
    return score    
#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()
