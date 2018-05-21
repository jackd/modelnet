#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modelnet.point_clouds import get_saved_cloud_normal_dataset


def vis(cloud):
    from mayavi import mlab
    from util3d.mayavi_vis import vis_point_cloud, vis_normals
    import numpy as np
    if hasattr(cloud, 'keys'):
        points = np.array(cloud['points'])
        normals = np.array(cloud['normals'])
    else:
        points = np.array(cloud)
        normals = None
    vis_point_cloud(points, scale_factor=0.01)
    if normals is not None:
        vis_normals(points, normals, scale_factor=0.002)
    mlab.show()


dataset_id = 'ModelNet40'
cat = 'airplane'

with get_saved_cloud_normal_dataset(dataset_id, 'train', cat) as ds:
    for example_id in ds:
        vis(ds[example_id])
