from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


def get_data_dir():
    key = 'MODELNET_PATH'
    if key in os.environ:
        dataset_dir = os.environ[key]
        if not os.path.isdir(dataset_dir):
            raise Exception('%s directory does not exist' % key)
        return dataset_dir
    else:
        raise Exception('%s environment variable not set.' % key)


def get_modelnet10_path():
    return get_zip_path('ModelNet10')


def get_modelnet40_path():
    return get_zip_path('ModelNet40')


def get_zip_path(dataset_id):
    return os.path.join(get_data_dir(), '%s.zip' % dataset_id)


def get_manual_alignment_folder():
    return os.path.join(get_data_dir(), 'modelnet40_manually_aligned')


def get_manual_alignment_subpath(mode, category, example_id):
    return os.path.join(
        category, mode, '%s_%s.off.annot' % (category, example_id))


def parse_manual_alignment_subpath(subpath):
    category, mode, fn = subpath.split('/')[-3:]
    example_id = fn[len(category)+1:-10]
    return mode, category, example_id
