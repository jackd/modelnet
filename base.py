from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .path import get_zip_path
from dids.file_io.zip_file_dataset import ZipFileDataset
from dids.core import DelegatingDataset
import util3d.mesh.off as off
from util3d.mesh.geom import triangulated_faces


train_synonyms = ('train',)
test_synonyms = ('test', 'eval', 'predict', 'infer')


def get_mode(mode):
    if mode in test_synonyms:
        return 'test'
    elif mode in train_synonyms:
        return 'train'
    else:
        raise ValueError('Unrecognized mode %s' % mode)


def get_off_subpath(dataset_id, mode, category, example_id):
    return os.path.join(
        dataset_id, category, mode, '%s_%s.off' % (category, example_id))


class ModelNetDataset(DelegatingDataset):
    def __init__(self, dataset_id, mode):
        self.dataset_id = dataset_id
        self.mode = get_mode(mode)
        self._keys = None
        path = get_zip_path(dataset_id)
        if not os.path.isfile(path):
            raise ValueError(
                'invalid dataset_id %s: no file at path %s' %
                (dataset_id, path))
        self._clients = set()

        def key_map(key):
            category, example_id = key
            return get_off_subpath(dataset_id, mode, category, example_id)

        self._zip_dataset = ZipFileDataset(path)
        dataset = self._zip_dataset.map_keys(key_map).map(
            off.OffObject.from_file)
        super(ModelNetDataset, self).__init__(dataset)

    def _open_resource(self):
        super(ModelNetDataset, self)._open_resource()
        z = self._zip_dataset._file
        keys = {}
        for path in z.namelist():
            subpaths = path.split('/')
            if len(subpaths) == 4 and subpaths[-1] != '':
                _, category, mode, fn = subpaths
                if mode == self.mode:
                    keys.setdefault(
                        category, []).append(fn[len(category)+1:-4])
        self._keys = {k: frozenset(v) for k, v in keys.items()}

    def _close_resource(self):
        self._keys = None
        super(ModelNetDataset, self)._close_resource()

    def keys(self):
        for k, v in self._keys.items():
            for vi in v:
                yield k, vi

    def contains(self, key):
        category, example_id = key
        return category in self._keys and example_id in self._keys[category]

    def category_dataset(self, category):
        def inverse_map_keys(key):
            return key[1]

        def map_keys(key):
            return category, key

        return self.map_keys(map_keys, inverse_map_keys)

    @property
    def categories(self):
        return self._keys.keys()

    def get_example_ids(self, category):
        return self._keys[category]


def get_categories(dataset_id='ModelNet40'):
    with ModelNetDataset(dataset_id, 'train') as ds:
        return ds.categories


if __name__ == '__main__':
    def vis(obj):
        from shapenet.mayavi_vis import vis_mesh
        import numpy as np
        from mayavi import mlab
        vis_mesh(np.array(obj.vertices), list(triangulated_faces(obj.faces)),
                 axis_order='xyz')
        mlab.show()

    ds = ModelNetDataset('ModelNet40', 'train')
    with ds:
        # cats = list(ds.categories)
        # cats.sort()
        # for cat in cats:
        #     print(cat, len(ds.get_example_ids(cat)))
        # print(len(cats))

        for key in ds.keys():
            obj = ds[key]
            vis(obj)
