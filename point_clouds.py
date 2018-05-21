from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from dids.file_io.hdf5 import Hdf5AutoSavingManager
from .base import get_mode

_point_cloud_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                '_point_clouds')
_cloud_normal_dir = os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                 '_cloud_normals')


class PointCloudManager(Hdf5AutoSavingManager):
    def __init__(self, dataset_id, mode, category, n_points=16384):
        self.dataset_id = dataset_id
        self.mode = get_mode(mode)
        self.n_points = n_points
        self.category = category

    @property
    def path(self):
        folder = os.path.join(_point_cloud_dir, str(self.n_points))
        if not os.path.isdir(folder):
            os.makedirs(folder)
        return os.path.join(folder, '%s_%s.hdf5' % (self.category, self.mode))

    @property
    def saving_message(self):
        return 'Saving %d point cloud data, %s - %s: %s' % (
            self.n_points, self.dataset_id, self.category, self.mode)

    def get_lazy_dataset(self):
        import parsed
        from util3d.mesh.sample import sample_faces
        import numpy as np

        dataset = parsed.get_saved_dataset(
            self.dataset_id, self.mode, self.category)

        def sample_mesh(mesh):
            v, f = (np.array(mesh[k]) for k in ('vertices', 'faces'))
            return sample_faces(v, f, self.n_points)

        return dataset.map(sample_mesh)

    def has_some_data(self):
        return os.path.isfile(self.path)

    def has_all_data(self):
        with self.get_lazy_dataset() as lazy:
            with self.get_saving_dataset() as saving:
                return len(lazy) == len(saving)


def get_saved_cloud_dataset(dataset_id, mode, category, n_points=16384):
    manager = PointCloudManager(dataset_id, mode, category, n_points)
    if not manager.has_some_data():
        manager.save_all()
    return manager.get_saving_dataset()


class CloudNormalManager(Hdf5AutoSavingManager):
        def __init__(self, dataset_id, mode, category, n_points=16384):
            self.dataset_id = dataset_id
            self.mode = get_mode(mode)
            self.n_points = n_points
            self.category = category

        @property
        def path(self):
            folder = os.path.join(_cloud_normal_dir, str(self.n_points))
            if not os.path.isdir(folder):
                os.makedirs(folder)
            return os.path.join(
                folder, '%s_%s.hdf5' % (self.category, self.mode))

        @property
        def saving_message(self):
            return 'Saving %d cloud normal data, %s - %s: %s' % (
                self.n_points, self.dataset_id, self.category, self.mode)

        def get_lazy_dataset(self):
            from .parsed import get_saved_dataset
            from util3d.mesh.sample import sample_faces_with_normals
            import numpy as np

            dataset = get_saved_dataset(
                self.dataset_id, self.mode, self.category)

            def sample_mesh(mesh):
                v, f = (np.array(mesh[k]) for k in ('vertices', 'faces'))
                points, normals = sample_faces_with_normals(
                    v, f, self.n_points)
                return dict(points=points, normals=normals)

            return dataset.map(sample_mesh)

        def has_some_data(self):
            return os.path.isfile(self.path)

        def has_all_data(self):
            with self.get_lazy_dataset() as lazy:
                with self.get_saving_dataset() as saving:
                    return len(lazy) == len(saving)


def get_saved_cloud_normal_dataset(dataset_id, mode, category, n_points=16384):
    manager = CloudNormalManager(dataset_id, mode, category, n_points)
    if not manager.has_all_data():
        manager.save_all()
    return manager.get_saving_dataset()
