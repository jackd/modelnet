"""data from [here](https://github.com/lmb-freiburg/orion)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dids.file_io.file_dataset import FileDataset
from .path import get_manual_alignment_folder, get_manual_alignment_subpath
from .path import parse_manual_alignment_subpath


def get_manually_aligned_annotations_dataset():
    def filter_fn(key):
        if key[-6:] != '.annot':
            return False
        substrs = key.split('/')
        return len(substrs) == 3

    path = get_manual_alignment_folder()

    def map_fn(fp):
        return int(float(fp.readline().rstrip())) // 90 % 4

    def key_map(key):
        return get_manual_alignment_subpath(*key)

    def inverse_key_map(subpath):
        return parse_manual_alignment_subpath(subpath)

    dataset = FileDataset(path).filter_keys(filter_fn)
    dataset = dataset.map_keys(key_map, inverse_key_map)
    dataset = dataset.map(map_fn)
    return dataset
