# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import wandb
import json
from pathlib import Path
from torch_utils import distributed as dist
from calculate_metrics import calc_python, parse_metric_list
import tarfile
import io
import gc

warnings.filterwarnings('ignore', '`resume_download` is deprecated')
warnings.filterwarnings("ignore", category=FutureWarning)

#----------------------------------------------------------------------------
# Configuration presets.

model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'

config_presets = {
    'edm2-img512-xs-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),  # fid = 3.53
    'edm2-img512-s-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),   # fid = 2.56
    'edm2-img512-m-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),   # fid = 2.25
    'edm2-img512-l-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.085.pkl'),   # fid = 2.06
    'edm2-img512-xl-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.085.pkl'),  # fid = 1.96
    'edm2-img512-xxl-fid':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'), # fid = 1.91
    'edm2-img64-s-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),    # fid = 1.58
    'edm2-img64-m-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-m-2147483-0.060.pkl'),    # fid = 1.43
    'edm2-img64-l-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),    # fid = 1.33
    'edm2-img64-xl-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img64-xl-0671088-0.040.pkl'),   # fid = 1.33
    'edm2-img512-xs-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.200.pkl'),  # fd_dinov2 = 103.39
    'edm2-img512-s-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.190.pkl'),   # fd_dinov2 = 68.64
    'edm2-img512-m-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.155.pkl'),   # fd_dinov2 = 58.44
    'edm2-img512-l-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.155.pkl'),   # fd_dinov2 = 52.25
    'edm2-img512-xl-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.155.pkl'),  # fd_dinov2 = 45.96
    'edm2-img512-xxl-dino':      dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.150.pkl'), # fd_dinov2 = 42.84
    'edm2-img512-xs-guid-fid':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl',   net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl', guidance=1.4), # fid = 2.91
    'edm2-img512-s-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl',    net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.025.pkl', guidance=1.4), # fid = 2.23
    'edm2-img512-m-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl',    net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.2), # fid = 2.01
    'edm2-img512-l-guid-fid':    dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl',    net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.2), # fid = 1.88
    'edm2-img512-xl-guid-fid':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl',   net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.020.pkl', guidance=1.2), # fid = 1.85
    'edm2-img512-xxl-guid-fid':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',  net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.2), # fid = 1.81
    'edm2-img512-xs-guid-dino':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.150.pkl',   net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.150.pkl', guidance=1.7), # fd_dinov2 = 79.94
    'edm2-img512-s-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.085.pkl',    net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.085.pkl', guidance=1.9), # fd_dinov2 = 52.32
    'edm2-img512-m-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.015.pkl',    net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=2.0), # fd_dinov2 = 41.98
    'edm2-img512-l-guid-dino':   dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.035.pkl',    net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.035.pkl', guidance=1.7), # fd_dinov2 = 38.20
    'edm2-img512-xl-guid-dino':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.030.pkl',   net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.7), # fd_dinov2 = 35.67
    'edm2-img512-xxl-guid-dino': dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',  net_neg=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.7), # fd_dinov2 = 33.09
}

  
def edges(crop_size, stride, i, j):
    le = stride * i
    te = stride * j
    re = stride * i + crop_size
    be = stride * j + crop_size
    return le,re,te,be

def swg_guidance(x, t, labels, net, swg_sizes, swg_steps, dtype=torch.float32,
                 y_pos=None):
    bs, c, img_size, _ = x.shape
    y_neg = torch.zeros_like(x)
    w = torch.zeros_like(x) # for averaging
    if not isinstance(swg_sizes, int):
        swg_sizes = swg_sizes[0]
    if not isinstance(swg_steps, int):
        swg_steps = swg_steps[0]
    assert swg_steps in [1, 2, 3, 5], f'Invalid crop steps: {swg_steps}'
    swg_sizes = [swg_sizes] if isinstance(swg_sizes, int) else swg_sizes
    for crop_size in swg_sizes:
        net.img_resolution = crop_size
        stride = (img_size - crop_size) // (swg_steps - 1) if swg_steps > 1 else 0
        neg_crops = []
        for i in range(swg_steps):
            for j in range(swg_steps):
                le, re, te, be = edges(crop_size, stride, i, j)
                xc = x[:, :, te:be, le:re]  # [bc, c, crop_size, crop_size]
                y_crop = net(xc, t, labels).to(dtype)
                
                y_neg[:, :, te:be, le:re] += y_crop
                w[:, :, te:be, le:re] += torch.ones_like(y_crop)
                
    y_neg = y_neg / w          
                    
    net.img_resolution = img_size
    return y_neg , w
#----------------------------------------------------------------------------
# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# extended to support classifier-free guidance.

def edm_sampler(
    net_pos, noise,
    labels=None, net_neg=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like, **cfg_kwargs,
):

    def compute_score(x, t, i, labels, num_steps, g_weight, g_method, swg_sizes, swg_steps, g_interval,
                      scale_window_guide=1):
        """Compute the score function at the point x and time t."""
        if g_interval == [] or g_interval is None:
            g_interval = [0, num_steps - 1]

        net_pos.img_resolution = x.shape[-1]
        y_pos = net_pos(x, t, labels).to(dtype)
        if g_interval[0] <= i <= g_interval[-1] and g_weight:
            if g_method == 'cfg':  # CFG
                y_neg = net_neg(x, t, class_labels=None).to(dtype)
            elif g_method == 'wmg':
                y_neg = net_neg(x, t, class_labels=labels).to(dtype)
            elif g_method == 'swg':
                if net_neg is None:
                    # Sliding window guidance with pos model (default)
                    y_neg, w = swg_guidance(x, t, labels, net_pos, swg_sizes, swg_steps, dtype=torch.float32,  y_pos=y_pos) 
                else:
                    # Sliding window guidance with weak model (new!)
                    y_neg, w = swg_guidance(x, t, labels, net_neg, swg_sizes, swg_steps, dtype=torch.float32, y_pos=y_pos) 
            
            elif g_method == 'wmg-swg' or g_method == 'cfg-swg' :
                assert len(g_weight)==2 and isinstance(g_weight, list), f'wmg-swg requires list of guidance scales. User provided: {g_weight}'
                assert net_neg is not None, 'RCT+SWG requires a weak model!!!'
                wmg_scale, swg_scale = g_weight 
                crop_size = swg_sizes[0] if isinstance(swg_sizes, list) else swg_sizes
                labels = None if g_method == 'cfg-swg' else labels
                y_weak = net_neg(x, t, class_labels=labels).to(dtype)
                y_neg, w = swg_guidance(x, t, labels, net_pos, [crop_size], swg_steps, dtype=torch.float32)
                return y_pos + wmg_scale*(y_pos - y_weak) + swg_scale * (y_pos - y_neg)
            else:
                raise NotImplementedError(f'Invalid CFG method: {g_method}')
            
            # new recreational idea: pixel wise masking
            if g_method == 'swg':
                # Masked SWG guidance (M-SWG) default
                if scale_window_guide==1:
                    mask = (w>1).float()
                    return  y_pos + g_weight * mask * (y_pos - y_neg)                  
                
            # apply guidance equation without masking
            return y_pos + g_weight * (y_pos - y_neg)

    cfg_kwargs['num_steps'] = num_steps
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - compute_score(x_hat, t_hat, i, labels, **cfg_kwargs)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - compute_score(x_next, t_next, i, labels, **cfg_kwargs)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net_pos_pkl,                                # Main network. Path, URL, or torch.nn.Module.
    net_neg_pkl         = None,                 # Reference network for guidance. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    save_numpy = False,
    save_tar = False,    
    **sampler_kwargs, # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net_pos_pkl, str):
        if verbose:
            dist.print0(f'Loading positive network from {net_pos_pkl} ...')
        with dnnlib.util.open_url(net_pos_pkl, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net_pos = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net_pos is not None

    # Load negative network.
    net_neg = None
    if sampler_kwargs["g_weight"] != 0:
        if net_neg_pkl is None or net_neg_pkl == 'none':
            if sampler_kwargs['g_method'] != 'none':
                dist.print0(f"Using positive network as negative.")
                net_neg = net_pos
            else:
                net_neg = None
        elif isinstance(net_neg_pkl, str):
            if verbose:
                dist.print0(f'Loading negative network from {net_neg_pkl} ...')
            with dnnlib.util.open_url(net_neg_pkl, verbose=(verbose and dist.get_rank() == 0)) as f:
                net_neg = pickle.load(f)['ema'].to(device)
        else:
            raise ValueError(f'Invalid net_neg_pkl: {net_neg_pkl}')


    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net_pos.img_channels, net_pos.img_resolution, net_pos.img_resolution], device=device)
                    r.labels = None
                    # Set the class labels for the mini-batch.
                    if net_pos.label_dim > 0:
                        label_idx = rnd.randint(net_pos.label_dim, size=[len(r.seeds)], device=device)
                        r.labels = torch.eye(net_pos.label_dim, device=device)[label_idx]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                    # Generate images.
                    latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net_pos=net_pos, noise=r.noise,
                        labels=r.labels, net_neg=net_neg, randn_like=rnd.randn_like, **sampler_kwargs)
                    r.images = encoder.decode(latents)

                    # Gather all images from all ranks (optional)
                    if save_numpy or save_tar:
                        gathered_images = gather_images_from_all_ranks(r.images, device)
                        
                        # Only rank 0 will have the gathered data
                        if dist.get_rank() == 0 and gathered_images is not None:
                            # Save the gathered images
                            save_images(gathered_images, outdir, batch_idx, save_numpy, save_tar)
                        
                    else:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:07d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:07d}.png'))
                                
                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

def check_image_count(outdir: str, seeds: list, load_numpy: bool = False):
    """Check how many images have already been generated for the given seeds. And return the seeds that need to be generated.

    Args:
        outdir (str): Directory where the images are saved.
        seeds (list): List of seeds to check.
        load_numpy (bool, optional): Whether to load images from numpy files or tar archives. Defaults to False.
    """
    if not os.path.exists(outdir):
        return seeds
    
    img_num = 0
    if load_numpy:
        npz_paths = [f for f in os.listdir(outdir) if f.endswith('.npz')]
        for f in npz_paths:
            img_num += np.load(os.path.join(outdir, f))['arr_0'].shape[0]
    else:
        tar_path = os.path.join(outdir, 'images.tar')
        if os.path.exists(tar_path):
            with tarfile.open(tar_path, 'r') as tar:
                for member in tar.getmembers():
                    if member.name.endswith('.npy') and member.isfile():
                        f = tar.extractfile(member)
                        file_content = f.read()
                        chunk_data = np.load(io.BytesIO(file_content))
                        img_num += chunk_data.shape[0]

    dist.print0(f'Found {img_num} images in {outdir}. Truncating seeds list from {len(seeds)} to {max(0, len(seeds) - img_num)}.')

    return seeds[img_num:]
        
def save_images(images, outdir, batch_idx, save_numpy=False, save_tar=False):
    if outdir is not None:
        if save_numpy:
            _outdir_ = Path(outdir)
            save_path = _outdir_ / f'samples_{batch_idx}.npz'
            np.savez(save_path, images.permute(0, 2, 3, 1).cpu().numpy())
        elif save_tar:
            tar_path = os.path.join(outdir, 'images.tar')
            
            if not os.path.exists(tar_path):
                # Create a new tar file
                with tarfile.open(tar_path, 'w') as tar:
                    pass
            # Open the tar file in append mode
            with tarfile.open(tar_path, 'a') as tar:
                # Save all images in the batch as a single numpy array
                images_np = images.permute(0, 2, 3, 1).cpu().numpy()
                image_bytes = io.BytesIO()
                np.save(image_bytes, images_np)
                image_bytes.seek(0)
                rank = dist.get_rank()
                tarinfo = tarfile.TarInfo(name=f'batch_r_{rank}_{batch_idx:04d}.npy')
                tarinfo.size = image_bytes.getbuffer().nbytes
                tar.addfile(tarinfo, fileobj=image_bytes)           
#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    if s == "none": return []
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def parse_float_list(value):
    """Parses a space- or comma-separated list of float values from the command line.
    Returns a single float if only one value is provided.
    """
    try:
        values = [float(v) for v in value.replace(",", " ").split()]
        return values[0] if len(values) == 1 else list(values)  # Return float if one value, else list
    except ValueError:
        raise click.BadParameter("Ensure all values are valid float numbers.")

def get_cuda_memory_usage():
    """
    Prints CUDA memory usage for each GPU in GB and percentage.
    Returns a dictionary with memory usage details.
    """
    import torch
    import torch.cuda
    import time
    memory_usage = {}

    device = torch.cuda.current_device()
    gpu_id = str(device)
    if not torch.cuda.is_available():
        memory_usage[gpu_id] = {"error": "CUDA not available"}
        pass

    try:
        # Get device properties
        properties = torch.cuda.get_device_properties(device)
        total_memory_bytes = properties.total_memory

        # Get memory allocated by PyTorch
        allocated_memory_bytes = torch.cuda.memory_allocated(device)
        # Get memory reserved by PyTorch's caching allocator (can be slightly higher than allocated)
        reserved_memory_bytes = torch.cuda.memory_reserved(device)

        # Calculate free memory (approximately, as caching allocator is complex)
        free_memory_bytes = total_memory_bytes - reserved_memory_bytes

        # Convert to GB
        total_memory_gb = total_memory_bytes / (1024**3)
        allocated_memory_gb = allocated_memory_bytes / (1024**3)
        reserved_memory_gb = reserved_memory_bytes / (1024**3)
        free_memory_gb = free_memory_bytes / (1024**3)

        # Calculate percentage used (based on reserved memory as it's a better indicator of actual usage)
        used_percentage = (reserved_memory_bytes / total_memory_bytes) * 100

        memory_usage[gpu_id] = {
            "total_gb": total_memory_gb,
            "allocated_gb": allocated_memory_gb,
            "reserved_gb": reserved_memory_gb,
            "free_gb": free_memory_gb,
            "used_percent": used_percentage
        }

        print(f"GPU {gpu_id}:")
        print(f"  Total Memory:     {total_memory_gb:.2f} GB")
        print(f"  Allocated Memory: {allocated_memory_gb:.2f} GB")
        print(f"  Reserved Memory:  {reserved_memory_gb:.2f} GB") # Often closer to actual usage
        print(f"  Free Memory:      {free_memory_gb:.2f} GB")
        print(f"  Used Percentage:  {used_percentage:.2f}%")
        print("-" * 20)

    except Exception as e:
        memory_usage[gpu_id] = {"error": str(e)}
        print(f"Error getting memory info for GPU {gpu_id}: {e}")

    return memory_usage        

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net_pos_pkl',              help='Network pickle filename', metavar='PATH|URL',                     type=str, default=None)
@click.option('--net_neg_pkl',              help='Reference network for guidance', metavar='PATH|URL',              type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)
@click.option('--g_weight',                 type=parse_float_list, default=0., help="CFG weight. 0 = no guidance")
@click.option('--g_method',                 type=click.Choice(['cfg', 'wmg', 'swg', 'none', 'wmg-swg', 'cfg-swg']), default='none', help="Guidance method")
@click.option('--swg_sizes',                type=parse_int_list, help="Crop size")
@click.option('--swg_steps',                type=parse_int_list, default=[2], help="Crop stride")
@click.option('--scale_window_guide',       type=int, default=1)
@click.option('--g_interval',               type=parse_int_list, default=None, help="Interval for CFG guidance")
@click.option('--use_wandb',                help='Enable Weights & Biases logging', metavar='BOOL',  type=bool, default=False, show_default=True)
@click.option('--wandb_group',              help='Group name for wandb', type=str, default=None)
@click.option('--wandb_runname',            help='Run name for wandb', type=str, default=None)
@click.option('--optuna_metric',            help='Optimize this metric using Optuna', metavar='STR', type=str, default=None, show_default=True)
@click.option('--optuna_trials',            help='Number of Optuna trials', metavar='INT', type=int, default=10, show_default=True)
@click.option('--optuna_runname',           help='Run name for Optuna', type=str, default="optuna_run")
@click.option('--optuna_step',              help='Step size for hyperparameter optimization', type=float, default=0.025, show_default=True)
@click.option('--metrics',                  help='List of metrics to compute', metavar='LIST', type=parse_metric_list, default='fid,fd_dinov2', show_default=True)
@click.option('--ref_path',                 help='Path to reference for FID/FDD calculation', metavar='PATH', type=str, default=None)
@click.option('--save_numpy',               help='Path to reference for FID/FDD calculation', is_flag=True, default=False)
@click.option('--save_tar',                 help='Save images in a single tar file', is_flag=True, default=False)
def cmdline(preset, metrics, ref_path, use_wandb, wandb_group, wandb_runname, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python sample_edm2.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 sample_edm2.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)
    
    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net_pos_pkl is None:
        raise click.ClickException('Please specify either --preset or --net_pos_pkl')

    assert not ((opts.g_method == 'short' or opts.g_method == 'shallow') and opts.net_neg_pkl is None), 'WMG requires a shallow network'
    assert not (opts.g_method == 'cfg' and opts.net_neg_pkl is None), 'g_weight requires an uncond network'
    assert not ((opts.g_method == 'wmg-swg') and opts.net_neg_pkl is None), 'wmg-swg requires net_neg_pkl'
    if  opts.g_method=='swg':
        assert isinstance(opts.g_weight, float) and opts.g_weight >= 0, 'SWG requires a single float g_weight'
    elif '-swg' in opts.g_method:
        assert (opts.g_method == 'wmg-swg' or opts.g_method == 'cfg-swg') and isinstance(opts.g_weight, list), 'wmg-swg requires list of guidance scales'

    # Initialize torch.distributed
    dist.init()

    # Wandb initialization
    all_arguments = click.get_current_context().params
    all_arguments["seeds"] = len(opts.seeds)
    if use_wandb and dist.get_rank() == 0:
        wandb.init(project="inductive-bias", group='iccv')
        wandb.run.name = wandb_runname
        wandb.config.update(all_arguments)

    # Save arguments. change folder mode for all users to read/write/edit access
    os.makedirs(opts.outdir, exist_ok=True, mode=0o777)
    json_filename = Path(opts.outdir) / 'generate_args.json'
    # Write arguments to JSON file
    with open(json_filename, 'w') as jsonfile:
        json.dump(all_arguments, jsonfile, indent=4)

    # Set optimized matmul precision
    # torch.set_float32_matmul_precision("medium")

    # Default mode.
    if opts.optuna_metric is None:
        default_entry_point(preset, metrics, ref_path, use_wandb, wandb_group, wandb_runname, opts)
        return
    
    # Optuna mode.
    optuna_entry_point(preset, metrics, ref_path, use_wandb, wandb_group, wandb_runname, opts)

def optuna_entry_point(preset, metrics, ref_path, use_wandb, wandb_group, wandb_runname, opts):
    import optuna
    assert opts.optuna_metric in metrics, f'Invalid metric: {opts.optuna_metric}, choose from {metrics}'
    assert isinstance(opts.g_weight, list) and (len(opts.g_weight) == 2 if not (opts.g_method == 'wmg-swg' or opts.g_method == 'cfg-swg') else 4), 'Optuna mode requires g_weight to be a list of two or four floats'

    optuna_metric = opts.optuna_metric
    optuna_step = opts.optuna_step
    n_trials = opts.optuna_trials
    run_name = opts.optuna_runname

    outdir = opts.outdir 
    wmg_scales, swg_scales = opts.g_weight[:2], opts.g_weight[2:]

    dist.print0(f'Optimizing {optuna_metric} on method {opts.g_method} using Optuna with scales: {wmg_scales, swg_scales} and step {optuna_step} for {n_trials} trials')

    def objective(single_trial: optuna.Trial):
        trial = optuna.integration.TorchDistributedTrial(single_trial)
        match opts.g_method:
            case 'wmg-swg':
                wmg_scale = trial.suggest_float('wmg_scale', *wmg_scales, step=optuna_step)
                swg_scale = trial.suggest_float('swg_scale', *swg_scales, step=optuna_step)
                opts.g_weight = [wmg_scale, swg_scale]
            case 'cfg-swg':
                g_scale = trial.suggest_float('g_scale', *wmg_scales, step=optuna_step)
                swg_scale = trial.suggest_float('swg_scale', *swg_scales, step=optuna_step)
                opts.g_weight = [g_scale, swg_scale]
            case _:
                wmg_scale = trial.suggest_float('wmg_scale', *wmg_scales, step=optuna_step)
                opts.g_weight = wmg_scale

        if trial.should_prune():
            raise optuna.TrialPruned('Duplicate parameter set')
            
        # Override outdir to avoid overwriting
        outdir_suffix = f"optuna-g_{opts.g_weight}".replace('[', '').replace(']', '').replace(', ', '-')
        opts.outdir = os.path.join(outdir, outdir_suffix)
        os.makedirs(opts.outdir, exist_ok=True, mode=0o777)
        results = default_entry_point(preset, metrics, ref_path, use_wandb, wandb_group, wandb_runname, opts)
        # import socket
        # print(f"Completed trial on rank {dist.get_rank()}, host {socket.getfqdn()} with {optuna_metric}={results[optuna_metric]} and g_weight={opts.g_weight}")
        gc.collect()
        torch.cuda.empty_cache()
        return results[optuna_metric]
    
    from optuna.pruners import BasePruner
    class RepeatPruner(BasePruner):
        def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
            prev_trials = study.get_trials(deepcopy=False)
            for t in prev_trials:
                prev_state_cond = t.state == optuna.trial.TrialState.COMPLETE or t.state == optuna.trial.TrialState.RUNNING
                if prev_state_cond and t.params == trial.params and t.number != trial.number:
                    return True
            return False
    
    rank = dist.get_rank()
    slurm_proc_id = os.environ.get('SLURM_PROCID')
    host = "localhost" if slurm_proc_id == 0 else os.environ.get('SQL_ADDR')
    storage = f"postgresql://optuna_user:optuna_123456@{host}:5432/optuna_db"
    study = None
    dist.print0(f"Connecting to {storage} ...")
    
    if rank == 0:
        pruner: optuna.pruners.BasePruner = RepeatPruner()
        sampler: optuna.samplers.BaseSampler = optuna.samplers.TPESampler(constant_liar=True)
        study = optuna.create_study(
            study_name=run_name,
            storage=storage,
            direction="minimize",
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=n_trials)
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("Number of finished trials: ", len(study.trials))
        print("Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
    else:
        for _ in range(n_trials):
            try:
                objective(None)
            except optuna.TrialPruned:
                pass

def optuna_debug(opts):
    # print(f'Evaluating objective on host {socket.getfqdn()}, using cuda:{torch.cuda.current_device()}')
    if isinstance(opts.g_weight, list):
        x, y = opts.g_weight
        return {'fid' : (x - 3) ** 2 + (y - 1)**2, 'fd_dinov2' : (x - 5) ** 2 + (y - 2)**2}
    
    x = opts.g_weight
    return {'fid' : (x - 3) ** 2, 'fd_dinov2' : (x - 5) ** 2}

def default_entry_point(preset, metrics, ref_path, use_wandb, wandb_group, wandb_runname, opts):
    # return optuna_debug(opts)

    optuna_metric = opts.pop('optuna_metric', "fid")
    optuna_step = opts.pop('optuna_step', 0.1)
    n_trials = opts.pop('optuna_trials', 2)
    run_name = opts.pop('optuna_runname', "optuna_run_debug")

    if len(opts.seeds) > 0:  # Otherwise jump to metric computation
        method_str = opts.g_method 
        if opts.g_method == 'none':
            method_str = 'karras inference'
            opts.g_weight = 0

        image_iter = generate_images(**opts)
        for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
            # get_cuda_memory_usage()
            pass
    
    if metrics and len(opts.seeds)>=2:
        outdir_imgs = os.path.join(opts.outdir, 'images.tar') if opts.save_tar else opts.outdir
        metric_kwargs = {'image_path': outdir_imgs,
                         'ref_path': ref_path,
                         'metrics': metrics,
                         'num_images': len(opts.seeds),
                         'seed': 0,
                         'max_batch_size': min(min(1024, opts.max_batch_size), len(opts.seeds))}
        
        results = calc_python(**metric_kwargs)
        
        out_dir_samples = Path(opts.outdir)
        if "samples" in out_dir_samples.name:
            out_path_csv = Path(f"{out_dir_samples.parent}.csv") 
        else:
            out_path_csv = Path(f"{out_dir_samples}.csv") 
        
        import pandas as pd
        if dist.get_rank() == 0:
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
            # save dict out as csv with pandas 
            try:
                pd.DataFrame(results, index=[0]).to_csv(out_path_csv) 
            except ImportError:
                print("Pandas not installed, skipping csv export!")
                pass

        torch.distributed.barrier()
        # Load saved results on all ranks, read dataframe as dict
        if dist.get_rank() != 0:
            results = pd.read_csv(out_path_csv).to_dict(orient='records')[0]

    torch.cuda.empty_cache()

    return results

def gather_images_from_all_ranks(images, device):
    """Gather images from all ranks to rank 0.
    
    Args:
        images: Tensor of images from the current rank
        device: The device where tensors reside
    
    Returns:
        Gathered images tensor on rank 0, None on other ranks
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return images
    
    # Get batch size from each rank
    local_batch_size = torch.tensor([images.shape[0]], device=device)
    batch_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    torch.distributed.all_gather(batch_sizes, local_batch_size)
    batch_sizes = [b.item() for b in batch_sizes]
    
    # For images, we need to handle variable sizes differently
    if dist.get_rank() == 0:
        total_images = sum(batch_sizes)
        all_images = torch.zeros(
            (total_images,) + images.shape[1:], 
            dtype=images.dtype, 
            device=device
        )
        
        # Place images from rank 0 directly
        start_idx = 0
        all_images[start_idx:start_idx+batch_sizes[0]] = images
        start_idx += batch_sizes[0]
        
        # Receive images from other ranks
        for rank in range(1, world_size):
            if batch_sizes[rank] > 0:
                # Create a placeholder for receiving images
                recv_images = torch.zeros(
                    (batch_sizes[rank],) + images.shape[1:], 
                    dtype=images.dtype, 
                    device=device
                )
                
                # Receive images from rank
                torch.distributed.recv(recv_images, src=rank)
                
                # Store in all_images
                all_images[start_idx:start_idx+batch_sizes[rank]] = recv_images
                start_idx += batch_sizes[rank]
        
        return all_images
    else:
        # Send images to rank 0
        torch.distributed.send(images, dst=0)
        return None

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
