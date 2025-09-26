# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import gc
import torch
import torch.distributed as dist
from diffusion.models import DiT_models, find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import json
import warnings
from datetime import timedelta

import tarfile
from pathlib import Path
from torch_utils.distributed import print0
from calculate_metrics import calculate_and_save_metrics

warnings.filterwarnings('ignore', '`resume_download` is deprecated')
warnings.filterwarnings("ignore", category=FutureWarning)
TAR_NAME = "images.tar"

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    return npz_path

def pack_png_to_tar(sample_dir: str, remove_imgs: bool = False) -> Path | None:
    """
    Finds all .png images in a specified directory, packs them into a .tar archive
    within that same directory, sets the archive's permissions to 0o777,
    and optionally removes the original images.

    Args:
        sample_dir: The path to the folder containing the .png images.
        remove_imgs: If True, delete the original .png files after packing.
                     Defaults to False.

    Returns:
        The Path object pointing to the created .tar archive,
        or None if the input directory doesn't exist or contains no .png files.
    """
    source_path = Path(sample_dir)

    # --- Input Validation ---
    if not source_path.is_dir():
        print(f"Error: Source directory not found or is not a directory: {sample_dir}")
        return None

    # --- Find Images ---
    image_files = list(source_path.glob("*.png"))

    if not image_files:
        print(f"No .png images found in {source_path}. No archive created.")
        return None

    # Sort files for consistent ordering and better compression
    image_files.sort()

    # --- Prepare Archive Path ---
    tar_filename = TAR_NAME
    tar_path = source_path / tar_filename

    print(f"Found {len(image_files)} .png images in {source_path}.")
    print(f"Packing images into {tar_path} ...")

    files_packed_count = 0
    files_to_remove = []  # Batch file removal for better performance

    # --- Create Tar Archive ---
    try:
        with tarfile.open(tar_path, "w") as tar:  # No compression = fastest
            for image_file in tqdm(image_files, desc="Packing images"):
                try:
                    tar.add(image_file, arcname=image_file.name)
                    files_packed_count += 1
                    
                    # Collect files for batch removal instead of removing immediately
                    if remove_imgs:
                        files_to_remove.append(image_file)

                except Exception as e:
                    print(f"Error adding file {image_file} to archive: {e}")

        # Batch remove files after tar is complete (much faster than individual removal)
        files_removed_count = 0
        if remove_imgs and files_to_remove:
            print(f"Removing {len(files_to_remove)} original images...")
            for image_file in tqdm(files_to_remove, desc="Removing files"):
                try:
                    image_file.unlink()
                    files_removed_count += 1
                except OSError as e:
                    print(f"  Error removing {image_file}: {e}")

        print(f"Successfully packed {files_packed_count} images.")
        if remove_imgs:
            print(f"Successfully removed {files_removed_count} original images.")

        # --- Set Archive Permissions ---
        os.chmod(tar_path, 0o777)

        return tar_path

    except Exception as e:
        print(f"An error occurred during tar creation: {e}")
        # Clean up potentially partially created tar file
        if tar_path.exists():
            try:
                tar_path.unlink()
                print(f"Cleaned up partially created archive: {tar_path}")
            except OSError as cleanup_e:
                print(f"Error cleaning up archive {tar_path}: {cleanup_e}")
        return None

def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    # Determine device ID before initializing process group
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device_id = local_rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")
    
    dist.init_process_group("nccl", timeout=timedelta(minutes=40), device_id=device)
    rank = dist.get_rank()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        strict_img_size=False
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.g_scale[0] >= 0, "In almost all cases, g_scale be >= 0"
    using_guidance = args.g_scale[0] > 0 and args.uncond==0
    
    print0(f"Sampling {args.num_fid_samples} images from {args.model} with {args.num_sampling_steps} steps using cfg {using_guidance}.")

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"g-{args.g_scale}-seed-{args.global_seed}"
    if args.crop>0:
        folder_name += f"_swg_mask_{int(args.mask)}"
    if args.cfg_crop>0:
        folder_name += f"_cfg_swg_{args.crop_scale}_mask_{int(args.mask)}"
    if args.uncond>0:
        folder_name += f"_uncond"
    if args.vanila_cfg>0:
        folder_name += f"_vanila_cfg"
    
    folder_name = folder_name.replace(", ", "_").replace("[", "").replace("]", "").replace(" ", "_")

    sample_folder_dir = f"{args.sample_dir}/{folder_name}/samples"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True, mode=0o777)
        print0(f"Saving .png samples at {sample_folder_dir}")
        # save args in output dir as json file with proper spacing
        with open(f"{sample_folder_dir}/args.json", "w") as f:
            json.dump(vars(args), f, indent=4)
    dist.barrier()
    
    # Default mode.
    if args.optuna_metric is None:
        results = default_entry_point(
        model=model,
        diffusion=diffusion,
        vae=vae,
        device=device,
        rank=rank,
        sample_folder_dir=sample_folder_dir,
        latent_size=latent_size,
        using_guidance=using_guidance,
        args=args
        )
        return
    
    # Optuna mode.
    optuna_entry_point(
        model=model,
        diffusion=diffusion,
        vae=vae,
        device=device,
        rank=rank,
        sample_folder_dir=sample_folder_dir,
        latent_size=latent_size,
        using_guidance=using_guidance,
        opts=args
    )

def optuna_entry_point(
    model: torch.nn.Module,
    diffusion: torch.nn.Module,
    vae: torch.nn.Module,
    device: torch.device,
    rank: int,
    sample_folder_dir: str,
    latent_size: int,
    using_guidance: bool,
    opts: argparse.Namespace):
    
    import optuna
    assert opts.optuna_metric in opts.metrics, f'Invalid metric: {opts.optuna_metric}, choose from {opts.metrics}'
    assert len(opts.g_scale) == 2 or len(opts.crop_scale) == 2, 'Optuna mode requires g_scale or crop_scale to be a list of two floats'

    optuna_metric = opts.optuna_metric
    optuna_step = opts.optuna_step
    n_trials = opts.optuna_trials
    run_name = opts.optuna_runname

    outdir = sample_folder_dir
    g_scales, swg_scales = opts.g_scale, opts.crop_scale

    print0(f'Optimizing {optuna_metric} using Optuna with scales: {g_scales, swg_scales} and step {optuna_step} for {n_trials} trials')

    def objective(single_trial: optuna.Trial):
        trial = optuna.integration.TorchDistributedTrial(single_trial)
        
        if len(g_scales) == 2:
            g_scale = trial.suggest_float('g_scale', *g_scales, step=optuna_step)
            opts.g_scale = [g_scale]
        else:
            opts.g_scale = [g_scales[0]]
        if len(swg_scales) == 2:
            swg_scale = trial.suggest_float('swg_scale', *swg_scales, step=optuna_step)
            opts.crop_scale = [swg_scale]
        else:
            opts.crop_scale = [swg_scales[0]]

        if trial.should_prune():
            raise optuna.TrialPruned('Duplicate parameter set')
            
        # Override outdir to avoid overwriting
        outdir_suffix = f"optuna-g_{opts.g_scale[0]}_{opts.crop_scale[0]}".replace(', ', '-')
        sample_folder_dir = os.path.join(outdir, outdir_suffix)
        os.makedirs(sample_folder_dir, exist_ok=True, mode=0o777)
        results = default_entry_point(model=model,
                                        diffusion=diffusion,
                                        vae=vae,
                                        device=device,
                                        rank=rank,
                                        sample_folder_dir=sample_folder_dir,
                                        latent_size=latent_size,
                                        using_guidance=using_guidance,
                                        args=opts)
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
    print0(f"Connecting to {storage} ...")
    
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

def default_entry_point(
    model: torch.nn.Module,
    diffusion: torch.nn.Module,
    vae: torch.nn.Module,
    device: torch.device,
    rank: int,
    sample_folder_dir: str,
    latent_size: int,
    using_guidance: bool,
    args: argparse.Namespace):

    """
    Run sampling.
    """
    
    args.g_scale = args.g_scale[0]
    args.crop_scale = args.crop_scale[0]
    # Try to read the .tar file and check how many images are stored inside
    tar_path = f"{sample_folder_dir}/{TAR_NAME}"
    packed_images = 0
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r") as tar:
            members = [1 if member.isfile() and member.name.endswith('.png') else 0 for member in tar.getmembers()]
            packed_images = sum(members)
            print0(f"Found {packed_images} png images in the tar file.")
    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    num_samples = max(0, args.num_fid_samples - packed_images)
    total_samples = int(math.ceil(num_samples / global_batch_size) * global_batch_size)
    print0(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = packed_images
    for _ in pbar:
        # Sample inputs:
        exist = 0
        for i in range(n):
            index = i * dist.get_world_size() + rank + total
            # check if file exists with os
            if os.path.exists(f"{sample_folder_dir}/{index:06d}.png"):
                exist += 1
        if exist == n:
            total += global_batch_size
            continue 
          
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        
        if args.uncond>0:
            y = torch.tensor([1000] * n, device=device)
        else:
            y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        chunk_result=False
        if using_guidance:
            if args.vanila_cfg>0:
                # vanilla CFG
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, g_scale=args.g_scale)
                sample_fn = model.forward_with_cfg
                chunk_result=True
            elif args.cfg_crop>0:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * n, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, g_scale=args.g_scale, crop_scale=args.crop_scale, mask=args.mask)
                sample_fn = model.forward_with_cfg_crop
                chunk_result=True
            elif args.crop>0:
                # Crop guidance!
                model_kwargs = dict(y=y, g_scale=args.g_scale, mask=args.mask)
                sample_fn = model.forward_with_crop 
            else:
                raise ValueError("Invalid guidance type!!!!")
        else:
            # base model sampling
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_guidance and chunk_result:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size
    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()

    # Calculate global total across all processes
    total_tensor = torch.tensor(total, device=device)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    global_total = total_tensor.item()
    
    results = None
    if args.metrics and global_total>=2:
        results = calculate_and_save_metrics(
            metrics = args.metrics,
            ref_path = args.ref_path,
            outdir = sample_folder_dir,
            num_images = min(global_total, args.num_fid_samples),
            max_batch_size = global_batch_size,
            use_wandb = False,
            num_workers = 8)
    
    # Barrier to ensure evaluation is complete before proceeding
    dist.barrier()
    
    # Pack tar file after all distributed operations are complete
    if rank == 0 and args.save_tar:
        print0("Packing images into tar file... This may take several minutes for large datasets.")
        pack_png_to_tar(sample_folder_dir, remove_imgs=True)
    return results

def list_of_floats(arg):
    """
    Custom argparse type function to parse a comma-separated string into a list of floats.
    Raises ArgumentTypeError if parsing fails.
    """
    try:
        # Split the string by comma and convert each part to float
        return [float(x) for x in arg.split('-')]
    except ValueError:
        # Raise a specific argparse error if any part is not a valid float
        raise argparse.ArgumentTypeError(f"'{arg}' must be a dash-separated list of floats (e.g., '0.1-0.2').")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=50)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--g_scale",  type=list_of_floats, default=[0])
    parser.add_argument("--crop_scale",  type=list_of_floats, default=[0])
    parser.add_argument("--crop",  type=int, default=0) 
    parser.add_argument("--mask",  type=lambda x: bool(int(x)), default=False)
    parser.add_argument("--cfg_crop",  type=int, default=0) 
    parser.add_argument("--uncond",  type=int, default=0) 
    parser.add_argument("--vanila_cfg",  type=int, default=0) 
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--save_tar", type=bool, default=False, action=argparse.BooleanOptionalAction)
    # list of strings of metrics to compute
    parser.add_argument("--metrics", nargs="+", type=str, default=["fid", "fd_dinov2"], choices=["fid", "fd_dinov2"],
                        help="List of metrics to compute. Default is empty list.")
    parser.add_argument('--ref_path', help='Path to reference for FID/FDD calculation', type=str, default=None)
    parser.add_argument('--optuna_metric', help='Optimize this metric using Optuna', type=str, default=None)
    parser.add_argument('--optuna_trials', help='Number of Optuna trials', type=int, default=10)
    parser.add_argument('--optuna_runname', help='Run name for Optuna', type=str, default="optuna_run")
    parser.add_argument('--optuna_step', help='Step size for hyperparameter optimization', type=float, default=0.05)
    args = parser.parse_args()

    main(args)

    dist.destroy_process_group()

