#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mayavi import mlab
from util3d.mayavi_vis import vis_point_cloud
from dids.core import ZippedDataset
from point_clouds import get_saved_cloud_dataset
from transformations import euler_matrix
from modelnet.alignment import get_manually_aligned_annotations_dataset
mode = 'train'
category = 'airplane'
clouds = get_saved_cloud_dataset('ModelNet40', mode, category)
alignments = get_manually_aligned_annotations_dataset()


def key_map_fn(key):
    return mode, category, key


def inverse_map_fn(key):
    m, c, example_id = key
    if m == mode and c == category:
        return example_id
    else:
        return None


alignments = alignments.map_keys(key_map_fn, inverse_map_fn)

zipped = ZippedDataset(clouds, alignments)


def transform(vals):
    cloud, alignment = vals
    R = euler_matrix(0, 0, (1 - alignment)*np.pi/2)[:3, :3]
    return np.matmul(np.array(cloud), R)


aligned = zipped.map(transform)

with aligned:
    for k in aligned:
        cloud = aligned[k]
        vis_point_cloud(cloud, color=(0, 0, 1), scale_factor=0.01)
        mlab.show()

# with zipped:
#     print(len(tuple(zipped)))
#     print(len(tuple(alignments)))
#     print(len(tuple(clouds)))
#     for k in zipped:
#         cloud, alignment = zipped[k]
