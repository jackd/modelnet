#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from mayavi import mlab
from util3d.mayavi_vis import vis_mesh
from util3d.mesh.bounding_mesh import BoundingMeshConfig
from modelnet.parsed import get_saved_dataset

print('Warning: extremely experimental')

dataset_id = 'ModelNet40'
mode = 'train'
category = 'toilet'

config = BoundingMeshConfig('bounding-convex-decomposition')

with get_saved_dataset(dataset_id, mode, category) as ds:
    for example_id in ds:
        mesh = ds[example_id]
        v, f = (np.array(mesh[k]) for k in ('vertices', 'faces'))
        v2, f2 = config.convert_mesh(v, f)
        mlab.figure()
        vis_mesh(v, f)
        mlab.figure()
        vis_mesh(v2, f2)
        print(v.shape, f.shape)
        print(v2.shape, f2.shape)
        mlab.show()
