Python functions for loading/manipulating the [modelnet](http://modelnet.cs.princeton.edu/) datasets.

# Setup
* Clone this repository and non-pip dependencies and add the parent directory to your `PYTHONPATH`
```
cd /path/to/parent_dir
git clone https://github.com/jackd/modelnet.git
git clone https://github.com/jackd/util3d.git
git clone https://github.com/jackd/dids.git
export PYTHONPATH=$PYTHONPATH:/path/to/parent_dir
```
Consider adding the `PYTHONPATH` modification to your `~/.bashrc`.
* Download the data for any/all of the following:
  - [ModelNet10](http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip)
  - [ModelNet40](http://modelnet.cs.princeton.edu/ModelNet40.zip)
  - [Aligned ModelNet40](https://github.com/lmb-freiburg/orion) and extract the `.tar` file.
* Set your `MODELNET_PATH` environment variable to the parent folder of the downloaded `.zip` files.
```
export MODELNET_PATH=/path/to/downloaded_zips
```
Consider adding this to your `~/.bashrc`.

See examples:
* This repo: [`modelnet/example`](https://github.com/jackd/modelnet/tree/master/example)
* Dataset interface: [`dids/example`](https://github.com/jackd/dids/tree/master/example).
* 3D utility functions []

This repository is under active development. Breaking changes will be occuring frequently.
