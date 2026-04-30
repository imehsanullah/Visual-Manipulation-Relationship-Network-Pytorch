# Running VMRD Training on CPU

Date recorded: 2026-04-26

This note records the successful CPU training smoke run for this repository after the VMRD V2 dataset was downloaded to:

```text
data/VMRD/vmrdcompv1
```

The environment used was:

```text
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env
```

## Result Summary

CPU training was able to start on the downloaded VMRD dataset.

The smoke run reached real optimizer iterations and printed losses:

```text
[session 1][epoch  1][iter    1/8466]
loss: 13.3146

[session 1][epoch  1][iter    2/8466]
loss: 11.6528

[session 1][epoch  1][iter    3/8466]
loss: 11.4276
```

The run was stopped after confirming training works. A full CPU epoch has `8466` iterations and will take many hours.

## CPU Training Command

Run from the repository root:

```bash
PYTHONPATH=$PWD/model:$PWD \
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python main.py \
  --dataset vmrdcompv1 \
  --frame all_in_one \
  --net res101 \
  --epochs 1 \
  --disp_interval 1 \
  --bs 2 \
  --nw 0
```

`PYTHONPATH=$PWD/model:$PWD` is required because parts of this old codebase still use legacy imports.

## Installed Runtime Packages

The environment passed:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -m pip check
```

Observed output:

```text
No broken requirements found.
```

Important package versions:

```text
torch 1.10.1+cu102
torchvision 0.11.2+cu102
numpy 1.19.5
scipy 1.5.4
opencv-python 4.5.5.64
Pillow 8.4.0
tensorboardX 2.5.1
protobuf 3.19.6
scikit-image 0.17.2
scikit-learn 0.24.2
```

Packages installed during CPU training setup:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -m pip install \
  --only-binary=:all: \
  scikit-image==0.17.2 \
  imageio==2.15.0 \
  networkx==2.5.1 \
  PyWavelets==1.1.1 \
  tifffile==2020.9.3

/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -m pip install \
  --only-binary=:all: \
  scikit-learn==0.24.2 \
  joblib==1.1.1 \
  threadpoolctl==3.1.0
```

Earlier dependency setup also installed:

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

/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -m pip install \
  --only-binary=:all: \
  torch==1.10.1 \
  torchvision==0.11.2
```

## Built Local Extensions

### `model.utils.cython_bbox`

The VMRD loader imports `model.utils.cython_bbox`.

Build command:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; setup(name='cython_bbox', ext_modules=cythonize([Extension('model.utils.cython_bbox', ['model/utils/bbox.pyx'], include_dirs=[numpy.get_include()])]), script_args=['build_ext', '--inplace'])"
```

Expected artifact:

```text
model/utils/cython_bbox.cpython-36m-x86_64-linux-gnu.so
```

### `pycocotools._mask`

The bundled `pycocotools/_mask.so` was incompatible with Python 3.6 and failed with:

```text
undefined symbol: _Py_ZeroStruct
```

Fix applied:

```text
pycocotools/_mask.pyx
```

Removed the stale distutils source directive:

```text
# distutils: sources = ../MatlabAPI/private/maskApi.c
```

Rebuild command:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; setup(name='pycocotools', ext_modules=cythonize([Extension('pycocotools._mask', ['pycocotools/_mask.pyx', 'pycocotools/maskApi.c'], include_dirs=[numpy.get_include(), 'pycocotools'], extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'])]), script_args=['build_ext', '--inplace'])"
```

Expected artifact:

```text
pycocotools/_mask.cpython-36m-x86_64-linux-gnu.so
```

### `model.roi_layers.C_ROIPooling`

The project imports `model.roi_layers.C_ROIPooling`.

The normal build failed because the machine has CUDA 12.9 while PyTorch was compiled with CUDA 10.2:

```text
The detected CUDA version (12.9) mismatches the version that was used to compile PyTorch (10.2).
```

CPU-only build command:

```bash
CUDA_VISIBLE_DEVICES= \
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python setup.py build_ext --inplace
```

Run that command from:

```text
model/roi_layers
```

Expected artifact:

```text
model/roi_layers/C_ROIPooling.cpython-36m-x86_64-linux-gnu.so
```

## Code Fixes Needed for CPU Training

These fixes were needed to get the old codebase running with Python 3.6 and modern package behavior.

### `datasets/imdb.py`

Problem:

```text
AttributeError: module 'PIL' has no attribute 'Image'
```

Fix:

```python
import PIL.Image
```

### `pycocotools/refcoco.py`

Problem:

```text
SyntaxError: Missing parentheses in call to 'print'
```

Fixes:

```text
cPickle -> pickle
Python 2 print statements -> print(...)
unicode fallback added for Python 3
len(seg) / 2 -> len(seg) // 2
```

### `datasets/bdds.py`

Problem:

```text
ModuleNotFoundError: No module named 'pascal_voc'
```

Fix:

```python
from .pascal_voc import pascal_voc
from .imdb import imdb
```

### `model/utils/config.py`

Problem:

```text
TypeError: load() missing 1 required positional argument: 'Loader'
```

Fix:

```python
yaml.safe_load(f)
```

Problem:

```text
ValueError: Type mismatch (<class 'tuple'> vs. <class 'list'>) for config key: SCALES
```

Fix:

```python
elif isinstance(b[k], tuple) and isinstance(v, list):
    v = tuple(v)
```

### `datasets/vmrd.py`

Problem:

```text
TypeError: unsupported operand type(s) for +: 'dict_items' and 'dict_items'
```

Fix:

```python
dict(list(self._load_vmrd_annotation(index).items()) +
     list(self._load_grasp_annotation(index).items()))
```

### `cfgs/vmrdcompv1_all_in_one_res101.yml`

Problem:

```text
FileNotFoundError: data/pretrained_model/resnet101_caffe.pth
```

Available pretrained file:

```text
data/pretrained_model/resnet101-5d3b4d8f.pth
```

Fix:

```yaml
PRETRAIN_TYPE: "pytorch"
```

CPU OP2L pooling fix:

```yaml
VMRN:
  OP2L_POOLING_MODE: align
```

`pool` mode failed on CPU with:

```text
RuntimeError: Not implemented on the CPU
```

### `model/MGN.py`

Problem:

```text
TypeError: empty() received an invalid combination of arguments
```

Cause:

```python
self.dout_base_model / 4
```

Python 3 returns a float for `/`, but PyTorch channel counts must be integers.

Fix:

```python
self.dout_base_model // 4
```

### `roi_data_layer/roibatchLoader.py`

Problem:

```text
TypeError: 'range' object does not support item assignment
```

Cause:

```python
np.random.shuffle(range(...))
```

Fix:

```python
np.random.shuffle(list(range(...)))
```

### Model Config Imports

Problem:

The training script updated `model.utils.config.cfg`, but several model files imported `utils.config.cfg` through legacy `PYTHONPATH`, creating duplicate config modules. This caused stale/default config values during forward passes.

Fix:

Normalize imports from:

```python
from utils.config import cfg
from utils.net_utils import ...
```

to:

```python
from model.utils.config import cfg
from model.utils.net_utils import ...
```

Files updated:

```text
model/AllinOne.py
model/Detectors.py
model/FPN.py
model/FasterRCNN.py
model/FasterRCNN_VMRN.py
```

### `model/roi_layers/roi_align.py`

Problem:

The custom CPU ROIAlign forward worked, but backward failed:

```text
RuntimeError: Not implemented on the CPU
```

Fix:

Use `torchvision.ops.roi_align`, which supports CPU autograd:

```python
from torchvision.ops import roi_align as torchvision_roi_align
```

Then call `torchvision_roi_align(...)` in `ROIAlign`, `RoIAlignAvg`, and `RoIAlignMax`.

## Validation Commands

Core import and dataset validation:

```bash
PYTHONPATH=$PWD/model:$PWD \
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env/bin/python -c "import torch, torchvision, cv2, scipy, sklearn, skimage, pycocotools._mask; import model.utils.cython_bbox; from model.roi_layers import ROIAlign; from datasets.vmrd import vmrd; print('imports_ok'); print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'torch_cuda', torch.version.cuda); print('torchvision', torchvision.__version__); print('sklearn', sklearn.__version__, 'skimage', skimage.__version__); d=vmrd('test'); print('vmrd_test_images', d.num_images)"
```

Observed output:

```text
imports_ok
torch 1.10.1+cu102 cuda_available True torch_cuda 10.2
torchvision 0.11.2+cu102
sklearn 0.24.2 skimage 0.17.2
vmrd_test_images 450
```

## GPU Limitation

Do not expect GPU training to work in this exact `vmrn_env` on the current machine.

Observed GPU:

```text
NVIDIA GeForce RTX 4080 SUPER
CUDA capability sm_89
```

Installed PyTorch:

```text
torch 1.10.1+cu102
```

CUDA test failed with:

```text
NVIDIA GeForce RTX 4080 SUPER with CUDA capability sm_89 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Reason:

```text
Python 3.6 cannot install modern PyTorch builds that support RTX 4080 / sm_89.
```

Practical options:

```text
1. Use CPU training in vmrn_env for compatibility checks only.
2. Create a newer Python/PyTorch CUDA 12 environment for practical RTX 4080 training.
3. Use an older GPU compatible with the current torch 1.10.1+cu102 wheel.
```

## Generated Files

The CPU smoke run generated the VMRD roidb cache:

```text
data/cache/vmrd_compv1_trainval_gt_roidb.pkl
```

Observed size:

```text
35520407 bytes
```

Compiled extension artifacts expected after setup:

```text
model/roi_layers/C_ROIPooling.cpython-36m-x86_64-linux-gnu.so
model/utils/cython_bbox.cpython-36m-x86_64-linux-gnu.so
pycocotools/_mask.cpython-36m-x86_64-linux-gnu.so
```
