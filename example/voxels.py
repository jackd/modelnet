#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from modelnet.base import get_categories
from modelnet.voxels import get_zipped_voxel_dataset

dataset_id = 'ModelNet40'
cats = get_categories(dataset_id)
dim = 32


def vis(voxels):
    from mayavi import mlab
    from util3d.mayavi_vis import vis_voxels
    vis_voxels(voxels)
    mlab.show()


with get_zipped_voxel_dataset(dataset_id, 'train', cats[0], dim) as ds:
    for example_id in ds:
        vis(ds[example_id].to_sparse().dense_data())
