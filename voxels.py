from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from util3d.voxel.dataset import BinvoxDataset
from util3d.voxel.convert import mesh_to_binvox
from .base import get_mode


_voxels_dir = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), '_voxels')


def get_voxel_dir(mode, category, dim):
    return os.path.join(_voxels_dir, str(dim), category, get_mode(mode))


def get_voxel_zip_path(mode, category, dim):
    folder = os.path.join(_voxels_dir, 'zipped', str(dim), category)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    return os.path.join(folder, '%s.zip' % get_mode(mode))


def create_voxel_data(
        dataset_id, mode, category, dim, overwrite=False, **kwargs):
    from progress.bar import IncrementalBar
    from parsed import get_saved_dataset
    mode = get_mode(mode)
    print('Saving voxels, %s - %d - %s - %s' % (
        dataset_id, dim, mode, category))
    folder = get_voxel_dir(mode, category, dim)
    if not os.path.isdir(folder):
        os.makedirs(folder)

    dataset = get_saved_dataset(dataset_id, mode, category)

    with dataset:
        bar = IncrementalBar(max=len(dataset))
        for key, value in dataset.items():
            bar.next()
            binvox_path = os.path.join(folder, '%s.binvox' % key)
            if os.path.isfile(binvox_path) and not overwrite:
                continue
            vertices, faces = (
                np.array(value[k]) for k in ('vertices', 'faces'))
            mins = np.min(vertices, axis=0)
            maxs = np.max(vertices, axis=0)
            center = (mins + maxs) / 2
            r = np.max(maxs - mins) / 2
            lower = center - r
            upper = center + r
            bounding_box = tuple(lower) + tuple(upper)
            mesh_to_binvox(
                vertices, faces, binvox_path, dim,
                bounding_box=bounding_box, **kwargs)
        bar.finish()


def get_voxel_dataset(dataset_id, mode, category, dim):
    folder = get_voxel_dir(mode, category, dim)
    if not os.path.isdir(folder):
        create_voxel_data(dataset_id, mode, category, dim)

    return BinvoxDataset(folder, mode='r')


def create_voxel_zip(dataset_id, mode, category, dim):
    import shutil
    src = get_voxel_dir(mode, category, dim)
    if not os.path.isdir(src):
        create_voxel_data(dataset_id, mode, category, dim)
        assert(os.path.isdir(src))
    dst = get_voxel_zip_path(mode, category, dim)
    shutil.make_archive(dst[:-4], 'zip', src)


def get_zipped_voxel_dataset(dataset_id, mode, category, dim):
    from util3d.voxel.binvox import Voxels
    from dids.file_io.zip_file_dataset import ZipFileDataset
    path = get_voxel_zip_path(mode, category, dim)
    if not os.path.isfile(path):
        create_voxel_zip(dataset_id, mode, category, dim)
        assert(os.path.isfile(path))
    dataset = ZipFileDataset(path)
    with dataset:
        dataset = dataset.subset(
            (k for k in dataset.keys() if k[-7:] == '.binvox'))

    def key_fn(key):
        return '%s.binvox' % key

    def inverse_key_fn(key):
        return key[:-7]

    return dataset.map(Voxels.from_file).map_keys(
        key_fn, inverse_fn=inverse_key_fn)
