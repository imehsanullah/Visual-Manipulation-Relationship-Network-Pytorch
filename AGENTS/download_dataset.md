# VMRD Dataset Download Notes

Date recorded: 2026-04-25

This project expects the Visual Manipulation Relationship Dataset at:

```text
data/VMRD/vmrdcompv1
```

The OpenDataLab VMRD page points to the Xi'an Jiaotong University dataset page:

```text
https://gr.xjtu.edu.cn/zh/web/zeuslan/dataset
```

That page lists two Dropbox downloads:

```text
V1, 5185 images without grasps:
https://www.dropbox.com/s/9y8920dav3dtqme/VMRD.tar.gz?dl=0

V2, 4683 images with grasps:
https://www.dropbox.com/s/ff0f4bqw4s1pxa2/VMRD%20V2%20fixed.tar.gz?dl=0
```

For this repository, use V2 because the loader and models use grasp annotations from the `Grasps` directory.
The V1 Dropbox link was observed as deleted in the browser, while the V2 link was available.

## Successful Download

Run from the repository root:

```bash
mkdir -p data/VMRD
curl -fL --retry 5 --retry-delay 5 -C - \
  -o data/VMRD/VMRD_V2_fixed.tar.gz \
  "https://www.dropbox.com/s/ff0f4bqw4s1pxa2/VMRD%20V2%20fixed.tar.gz?dl=1"
```

The `dl=1` query parameter is important because it downloads the actual tarball instead of the Dropbox preview page.

Successful archive check:

```bash
ls -lh data/VMRD/VMRD_V2_fixed.tar.gz
file data/VMRD/VMRD_V2_fixed.tar.gz
```

Observed result:

```text
data/VMRD/VMRD_V2_fixed.tar.gz: 315M
gzip compressed data
```

## Archive Layout

The archive contains dataset folders directly at the archive root:

```text
Annotations/
Grasps/
ImageSets/
JPEGImages/
```

Because `datasets/vmrd.py` hard-codes:

```text
data/VMRD/vmrdcompv1
```

extract the archive into that directory:

```bash
mkdir -p data/VMRD/vmrdcompv1
tar -xzf data/VMRD/VMRD_V2_fixed.tar.gz -C data/VMRD/vmrdcompv1
```

## Validation

Check the extracted structure:

```bash
find data/VMRD/vmrdcompv1 -maxdepth 2 -type d | sort
find data/VMRD/vmrdcompv1/ImageSets/Main -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected folders:

```text
data/VMRD/vmrdcompv1
data/VMRD/vmrdcompv1/Annotations
data/VMRD/vmrdcompv1/Grasps
data/VMRD/vmrdcompv1/ImageSets
data/VMRD/vmrdcompv1/ImageSets/Main
data/VMRD/vmrdcompv1/JPEGImages
```

Expected split files:

```text
test.txt
trainval.txt
```

Observed counts:

```bash
find data/VMRD/vmrdcompv1/JPEGImages -type f -name '*.jpg' | wc -l
find data/VMRD/vmrdcompv1/Annotations -type f -name '*.xml' | wc -l
find data/VMRD/vmrdcompv1/Grasps -type f -name '*.txt' | wc -l
```

Observed output:

```text
4683
4683
4683
```

Observed disk usage:

```text
315M data/VMRD/VMRD_V2_fixed.tar.gz
359M data/VMRD/vmrdcompv1
```

## Environment Dependencies Used

The environment used for validation was:

```text
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env
```

Install the project dependencies into that exact environment:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -m pip install \
  --only-binary=:all: \
  numpy==1.19.5 \
  scipy==1.5.4 \
  opencv-python==4.5.5.64 \
  Cython==0.29.36 \
  msgpack==1.0.5 \
  easydict==1.13 \
  matplotlib==3.3.4 \
  PyYAML==6.0.1 \
  tensorboardX==2.5.1 \
  protobuf==3.19.6 \
  pillow==8.4.0
```

The loader imports `torch`, so install a Python 3.6 compatible PyTorch pair:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -m pip install \
  --only-binary=:all: \
  torch==1.10.1 \
  torchvision==0.11.2
```

Avoid installing unpinned `opencv-python` in this Python 3.6 environment. Pip selected `opencv-python 4.12` from source, which triggered a long CMake build. Pinning `opencv-python==4.5.5.64` used a compatible wheel.

## Local Extension Build

The dataset import requires `model.utils.cython_bbox`. Build it from `model/utils/bbox.pyx`:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; setup(name='cython_bbox', ext_modules=cythonize([Extension('model.utils.cython_bbox', ['model/utils/bbox.pyx'], include_dirs=[numpy.get_include()])]), script_args=['build_ext', '--inplace'])"
```

This creates an ignored compiled extension under `model/utils/`.

## Repo Import Fix

`datasets/imdb.py` used `PIL.Image.open` but only imported `PIL`.
The successful validation required adding:

```python
import PIL.Image
```

## Loader Sanity Checks

Test split validation:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -c "from datasets.vmrd import vmrd; d=vmrd('test'); print('num_images', d.num_images); print('first_index', d.image_index[0]); print('first_image', d.image_path_at(0)); g=d._load_grasp_annotation(d.image_index[0]); print('first_grasps', g['grasps'].shape, g['grasp_inds'].shape)"
```

Observed output:

```text
num_images 450
first_index 00035
first_image /home/user/ehsanullahm1/thesis/Visual-Manipulation-Relationship-Network-Pytorch/data/VMRD/vmrdcompv1/JPEGImages/00035.jpg
first_grasps (31, 8) (31,)
```

Trainval split validation:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -c "from datasets.vmrd import vmrd; d=vmrd('trainval'); print('num_images_augmented', d.num_images); print('original_num_img', d._original_num_img); print('first_index', d.image_index[0]); print('first_image', d.image_path_at(0))"
```

Observed output:

```text
Initialize image widths and heights...
num_images_augmented 16932
original_num_img 4233
first_index 00022
first_image /home/user/ehsanullahm1/thesis/Visual-Manipulation-Relationship-Network-Pytorch/data/VMRD/vmrdcompv1/JPEGImages/00022.jpg
```
