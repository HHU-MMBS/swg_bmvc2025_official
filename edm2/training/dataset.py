# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import io
import os
import tarfile
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

import torchvision.transforms.v2 as T

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        use_labels  = True,     # Enable conditioning labels? False = label dimension is zero.
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
        p_uncond    = 0,      # Probability of sampling data-unconditional images.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None
        self.p_uncond = p_uncond

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self._raw_shape[1:], f'{image.shape} != {self._raw_shape[1:]}'
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return torch.from_numpy(image).float(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1 if np.random.rand() > self.p_uncond else 0
            label = onehot
        return torch.from_numpy(label)

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self): # [CHW]
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = anything goes.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in supported_ext)
        # Filter out corrupted files with size 0 
        self._image_fnames = [fname for fname in self._image_fnames if os.path.getsize(os.path.join(self._path, fname)) > 0]
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        ext = self._file_ext(fname)
        with self._open_file(fname) as f:
            if ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

class NumpyImageDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_size=50000, channels_first=True):
        self._path = path
        self.max_size = max_size
        self.channels_first = channels_first
        
        # Open the file with memory-mapping
        if self._path.endswith('.npz'):
            self.all_data = np.load(self._path, mmap_mode='r')["arr_0"]
        elif self._path.endswith('.npy'):
            self.all_data = np.load(self._path, mmap_mode='r')
        
        # Limit to max_size if specified
        if self.max_size is not None and self.all_data.shape[0] > self.max_size:
            self.data_len = self.max_size
        else:
            self.data_len = self.all_data.shape[0]

        # Determine shape and transpose setup without loading into memory
        sample_shape = self.all_data.shape[1:] if self.data_len > 0 else None
        if channels_first and sample_shape and sample_shape[-1] == 3:
            self.transpose_order = (2, 0, 1)  # For HWC to CHW
        else:
            self.transpose_order = None

        print("Dataset initialized with lazy loading.")
        print(f"Image shape: {sample_shape}, channels_first: {channels_first}")

    def __getitem__(self, idx):
        if idx >= self.data_len:
            raise IndexError("Index out of range")
        
        # Load only the specific slice for the requested index
        image = self.all_data[idx]
        
        # Transpose if necessary
        if self.transpose_order:
            image = np.transpose(image, self.transpose_order)
        # print("Img loaded",image.shape)
        return torch.from_numpy(image).float()

    def __len__(self):
        return self.data_len


#----------------------------------------------------------------------------

class TarDataset(Dataset):

    def __init__(self, path, max_size=50000):
        """
        Dataset that loads all images from a tar file containing numpy arrays into memory at once.
        
        Args:
            path: Path to the tar file containing numpy arrays
            max_size: Maximum number of images to load
        """
        self.path = path
        self.max_size = max_size
        
        # Load all images into memory
        # dist.print0(f"Loading images from {path} into memory...")
        self.images = []
        self.load_all_images()
        # dist.print0(f"Loaded {len(self.images)} images into memory")
    
    def load_all_images(self):
        """Load all images from tar file into memory at once"""
        with tarfile.open(self.path, 'r') as tar:
            # First pass: identify all numpy and png files
            numpy_files = []
            png_files = []
            
            for member in tar.getmembers():
                if member.isfile():
                    if member.name.endswith('.npy'):
                        numpy_files.append(member)
                    elif member.name.endswith('.png'):
                        png_files.append(member)

            if len(numpy_files) == 0 and len(png_files) == 0:
                raise ValueError(f"No valid files found in tar archive {self.path}.")
            
            # Second pass: load all images
            total_images = 0
            for member in numpy_files:
                file_ = tar.extractfile(member)
                file_content = file_.read()
                chunk_data = np.load(io.BytesIO(file_content))
                
                # Add all images from this chunk to our list
                for i in range(chunk_data.shape[0]):
                    self.images.append(chunk_data[i])
                    total_images += 1
                    
                    # Stop if we've reached max_size
                    if total_images >= self.max_size:
                        return
                    
            for member in png_files:
                file_ = tar.extractfile(member)
                file_content = file_.read()
                image = PIL.Image.open(io.BytesIO(file_content)).convert('RGB')
                image_np = np.array(image)
                self.images.append(image_np)
                total_images += 1
                
                # Stop if we've reached max_size
                if total_images >= self.max_size:
                    return
            
            if len(self.images)>self.max_size:
                self.images = self.images[:self.max_size]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")
        
        # Get image directly from memory
        image_np = self.images[idx]
        # Convert to PyTorch tensor with CHW format and normalize
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        return image_tensor

def get_dataset(samples_path : str):
    """
    Get dataset from samples path
    """
    if not os.path.exists(samples_path):
        raise ValueError(f"Path {samples_path} does not exist")
    # If in the parent directory there is a file with .npz extension, use NumpyDataset
    dataset_path = get_dataset_path(samples_path)
    if dataset_path.endswith(".npz"):
        dataset = NumpyImageDataset(dataset_path)
    elif dataset_path.endswith(".tar"):
        dataset = TarDataset(dataset_path)
    else:
        dataset = ImageFolderDataset(dataset_path)
    return dataset

def get_dataset_path(samples_path : str):
    """
    Get dataset path from samples path
    """
    # Trim trailing slashes
    samples_path = samples_path.rstrip("/")
    if not os.path.exists(samples_path):
        raise ValueError(f"Path {samples_path} does not exist")
    # If in the parent directory there is a file with .npz extension, use NumpyDataset
    for file in os.listdir(os.path.dirname(samples_path)):
        if file.endswith(".npz"):
            return os.path.join(os.path.dirname(samples_path), file)
    
    # If there is an images.tar file in the directory, use TarDataset
    for file in os.listdir(samples_path):
        if file.endswith(".tar"):
            return os.path.join(samples_path, file)
    
    # Otherwise, use ImageFolderDataset
    return samples_path
