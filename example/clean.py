#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from modelnet.base import get_categories
from modelnet.clean import get_clean_mesh_dataset

dataset_id = 'ModelNet40'
cats = tuple(get_categories(dataset_id))


def get_counts():
    from base import ModelNetDataset
    with ModelNetDataset(dataset_id, 'train') as ds:
        ids = []
        for cat in cats:
            ids.append((cat, len(ds.get_example_ids(cat))))
    return ids


ids = {k: v for k, v in get_counts()}

n_cats = len(cats)
for mode in ('train',):
    for i, cat in enumerate(cats):
        print('Category %d / %d' % (i+1, n_cats))
        get_clean_mesh_dataset(dataset_id, mode, cat)


def vis(mesh):
    from mayavi import mlab
    from util3d.mayavi_vis import vis_mesh, vis_normals
    from util3d.mesh.sample import sample_faces_with_normals
    v, f = (np.array(mesh[k]) for k in ('vertices', 'faces'))
    n_points = 1024
    p, n = sample_faces_with_normals(v, f, n_points)
    print(np.min(v, axis=0))
    print(np.max(v, axis=0))
    vis_mesh(v, f)
    vis_normals(p, n)
    mlab.show()


with get_clean_mesh_dataset(dataset_id, 'train', cats[10]) as ds:
    for example_id in ds:
        vis(ds[example_id])
