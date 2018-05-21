#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse


def generate_voxels(dataset_id, dim=32, cats=None, modes=None):
    from modelnet.base import get_categories
    from modelnet.voxels import get_zipped_voxel_dataset
    if cats is None or len(cats) == 0:
        cats = tuple(get_categories(dataset_id))
    if modes is None or len(modes) == 0:
        modes = ('train', 'test')
    n_cats = len(cats)
    for mode in modes:
        for i, cat in enumerate(cats):
            print('Category %d / %d' % (i+1, n_cats))
            get_zipped_voxel_dataset(dataset_id, mode, cat, dim)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', default='ModelNet40', choices=('ModelNet40', 'ModelNet10'))
parser.add_argument('-n', default=16384, type=int)
parser.add_argument('--cats', '-c', default=None, type=str, nargs='*')
parser.add_argument(
    '-m', '--mode', nargs='*', choices=('train', 'test'), type=str)
args = parser.parse_args()

generate_voxels(args.i, args.n, args.cats, args.mode)
