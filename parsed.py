from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from dids.file_io.hdf5 import Hdf5AutoSavingManager
from modelnet.base import get_mode

_parsed_dir = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), '_parsed')


class ModelNetDatasetManager(Hdf5AutoSavingManager):
    def __init__(self, dataset_id, mode, category):
        self.dataset_id = dataset_id
        self.category = category
        self.mode = get_mode(mode)

    @property
    def path(self):
        if not os.path.isdir(_parsed_dir):
            os.makedirs(_parsed_dir)
        return os.path.join(
            _parsed_dir, '%s_%s.hdf5' % (self.category, self.mode))

    @property
    def saving_message(self):
        return 'Parsing %s - %s: %s' % (
            self.dataset_id, self.category, self.mode)

    def get_lazy_dataset(self):
        from base import ModelNetDataset
        from util3d.mesh.geom import triangulated_faces
        base = ModelNetDataset(self.dataset_id, self.mode).category_dataset(
            self.category)

        def obj_to_mesh(obj):
            vertices = np.array(obj.vertices, dtype=np.float32)
            faces = np.array(
                tuple(triangulated_faces(obj.faces)), dtype=np.int32)
            min_vals = np.min(vertices, axis=0)
            max_vals = np.max(vertices, axis=0)
            offset = (min_vals + max_vals) / 2
            vertices -= offset
            scale = np.sqrt(np.max(np.sum(vertices**2, axis=-1)))
            vertices /= scale
            attrs = dict(offset=offset, scale=scale)
            return dict(vertices=vertices, faces=faces, attrs=attrs)

        return base.map(obj_to_mesh)

    def has_some_data(self):
        return os.path.isfile(self.path)

    def has_all_data(self):
        with self.get_lazy_dataset() as lazy:
            with self.get_saving_dataset() as saving:
                return len(lazy) == len(saving)


def get_saved_dataset(dataset_id, mode, category):
    manager = ModelNetDatasetManager(dataset_id, mode, category)
    # return manager.get_saved_dataset()
    if manager.has_some_data():
        return manager.get_saving_dataset()
    else:
        return manager.get_saved_dataset()


if __name__ == '__main__':
    from base import get_categories
    dataset_id = 'ModelNet40'
    cats = tuple(get_categories(dataset_id))

    n_cats = len(cats)
    for mode in ('train', 'test'):
        for i, cat in enumerate(cats):
            print('Category %d / %d' % (i+1, n_cats))
            get_saved_dataset(dataset_id, mode, cat)

    # def vis(mesh):
    #     from shapenet.mayavi_vis import vis_mesh
    #     from mayavi import mlab
    #     v, f = (np.array(mesh[k]) for k in ('vertices', 'faces'))
    #     print(mesh.attrs['scale'])
    #     print(np.min(v, axis=0))
    #     print(np.max(v, axis=0))
    #     vis_mesh(v, f)
    #     mlab.show()
    #
    # with get_saved_dataset(dataset_id, 'train', cats[10]) as ds:
    #     for example_id in ds:
    #         vis(ds[example_id])
