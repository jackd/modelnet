"""Experimental cleaned dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from dids.file_io.hdf5 import Hdf5AutoSavingManager
from modelnet.base import get_mode

_clean_dir = os.path.join(
    os.path.realpath(os.path.dirname(__file__)), '_clean')

print('Warning: modelnet.clean is highly experimental')


class CleanMeshDatasetManager(Hdf5AutoSavingManager):
    def __init__(self, dataset_id, mode, category):
        self.dataset_id = dataset_id
        self.category = category
        self.mode = get_mode(mode)

    @property
    def path(self):
        if not os.path.isdir(_clean_dir):
            os.makedirs(_clean_dir)
        return os.path.join(
            _clean_dir, '%s_%s.hdf5' % (self.category, self.mode))

    @property
    def saving_message(self):
        return 'Parsing %s - %s: %s' % (
            self.dataset_id, self.category, self.mode)

    def get_lazy_dataset(self):
        from modelnet.parsed import get_saved_dataset

        def clean_mesh(mesh):
            from util3d.mesh.clean import clean
            vertices, faces = (
                np.array(mesh[k]) for k in ('vertices', 'faces'))
            tol = 1e-3*(np.max(vertices) - np.min(vertices))
            vertices, faces = clean(vertices, faces, duplicate_tol=tol)
            return dict(vertices=vertices, faces=faces)

        dataset = get_saved_dataset(
            self.dataset_id, self.mode, self.category).map(clean_mesh)
        return dataset

    def has_some_data(self):
        return os.path.isfile(self.path)

    def has_all_data(self):
        with self.get_lazy_dataset() as lazy:
            with self.get_saving_dataset() as saving:
                return len(lazy) == len(saving)


def get_clean_mesh_dataset(dataset_id, mode, category):
    manager = CleanMeshDatasetManager(dataset_id, mode, category)
    # return manager.get_saved_dataset()
    # if manager.has_some_data():
    if manager.has_all_data():
        return manager.get_saving_dataset()
    else:
        return manager.get_saved_dataset()
