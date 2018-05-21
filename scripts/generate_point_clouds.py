#!/usr/bin/python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse


def generate_point_clouds(
        dataset_id, n_points, cats=None, modes=('train', 'test')):
    from modelnet.base import get_categories
    from modelnet.point_clouds import get_saved_cloud_dataset
    from modelnet.point_clouds import get_saved_cloud_normal_dataset
    if cats is None or len(cats) == 0:
        cats = tuple(get_categories(dataset_id))
    n_cats = len(cats)
    for mode in modes:
        for i, cat in enumerate(cats):
            print('Category %d / %d' % (i+1, n_cats))
            get_saved_cloud_dataset(dataset_id, mode, cat, n_points)
            get_saved_cloud_normal_dataset(dataset_id, mode, cat, n_points)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', default='ModelNet40', choices=('ModelNet40', 'ModelNet10'))
parser.add_argument('-n', default=16384, type=int)
parser.add_argument('--cats', '-c', default=None, type=str, nargs='*')
parser.add_argument(
    '-m', '--mode', nargs='*', choices=('train', 'test'), type=str)
args = parser.parse_args()

generate_point_clouds(args.i, args.n, args.cats, args.mode)
