#!/usr/bin/python
"""Demonstrates util3d.voxel.rle.get_contiguous_regions_2d."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from modelnet.voxels import get_zipped_voxel_dataset
from util3d.voxel.rle import get_contiguous_regions_2d
# from util3d.voxel.manip import get_interpolated_roots_1d
from util3d.mayavi_vis import vis_voxels
from mayavi import mlab

dataset_id = 'ModelNet40'
mode = 'train'
category = 'toilet'
dim = 32

with get_zipped_voxel_dataset(dataset_id, mode, category, dim) as dataset:
    for example_id in dataset:
        voxels = dataset[example_id]
        dense = voxels.dense_data()
        sparse = voxels.sparse_data()
        contiguous = get_contiguous_regions_2d(voxels.rle_data(), dim)
        reconstructed = np.zeros((dim,)*3, dtype=np.bool)
        for n, c in enumerate(contiguous):
            i = n // dim
            j = n % dim
            for a, b in c:
                reconstructed[i, j, a:b] = True
        print(np.all(dense == reconstructed))
        mlab.figure()
        vis_voxels(dense, color=(0, 0, 1))
        mlab.figure()
        vis_voxels(reconstructed, color=(0, 1, 0))
        mlab.show()

        # print(len(sparse))
        # i = sparse[0][10]
        # j = sparse[1][10]
        # data = np.array(dense[i, j], dtype=np.float32)*2 - 0.5
        # lower, frac = get_interpolated_roots_1d(data)
        # print(data)
        # print(lower)
        # print(frac)
        # exit()
