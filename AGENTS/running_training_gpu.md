# Running VMRD Training on GPU

Date recorded: 2026-04-30

This note records the successful GPU training smoke run for this repository using a new Python 3.10 environment. The original CPU-compatible environment was left untouched:

```text
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env
```

The new GPU environment is:

```text
/home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310
```

## Result Summary

GPU training works in `vmrn_gpu_py310` on the downloaded VMRD dataset.

The smoke run reached real CUDA optimizer iterations:

```text
[session 1][epoch  1][iter    1/8466]
loss: 14.2829
time cost: 0.553993

[session 1][epoch  1][iter    2/8466]
loss: 13.1911
time cost: 0.226369

[session 1][epoch  1][iter    3/8466]
loss: 11.0997
time cost: 0.200764

[session 1][epoch  1][iter   61/8466]
loss: 7.6773
time cost: 0.224207
```

The process was stopped after confirming GPU training works. It did not complete a full epoch.

## Environment Creation

Create the env:

```bash
/home/user/ehsanullahm1/miniconda3/bin/conda create -y \
  -n vmrn_gpu_py310 \
  python=3.10 \
  pip
```

Upgrade pip tooling:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310/bin/python -m pip install \
  --upgrade pip setuptools wheel
```

## PyTorch CUDA Install

Install PyTorch CUDA 12.4 wheels:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310/bin/python -m pip install \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu124
```

Observed versions:

```text
torch 2.6.0+cu124
torchvision 0.21.0+cu124
```

## Project Dependencies

Install project dependencies, pinning NumPy below 2:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310/bin/python -m pip install \
  "numpy==1.26.4" \
  scipy \
  opencv-python \
  "Cython==0.29.37" \
  msgpack \
  easydict \
  matplotlib \
  PyYAML \
  tensorboardX \
  "protobuf<5" \
  scikit-image \
  scikit-learn
```

`Cython==0.29.37` is important. Cython 3 failed to compile the repo’s old `pycocotools/_mask.pyx`.

The environment passed:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310/bin/python -m pip check
```

Observed output:

```text
No broken requirements found.
```

## CUDA Verification

Run:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310/bin/python -c "import torch, torchvision, numpy; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('torchvision', torchvision.__version__); print('numpy', numpy.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0)); x=torch.randn(1024,1024,device='cuda'); y=x @ x; torch.cuda.synchronize(); print('cuda_mm_ok', float(y[0,0]))"
```

Observed output:

```text
torch 2.6.0+cu124 cuda 12.4
torchvision 0.21.0+cu124
numpy 1.26.4
cuda_available True
device NVIDIA GeForce RTX 4080 SUPER capability (8, 9)
cuda_mm_ok ...
```

## Built Local Extensions

### `pycocotools._mask`

Problem with the old bundled artifact:

```text
ImportError: pycocotools/_mask.so: undefined symbol: _Py_ZeroStruct
```

Also, Cython 3 failed with strict compile errors. Use Cython 0.29.37.

Build command:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310/bin/python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; setup(name='pycocotools', ext_modules=cythonize([Extension('pycocotools._mask', ['pycocotools/_mask.pyx', 'pycocotools/maskApi.c'], include_dirs=[numpy.get_include(), 'pycocotools'], extra_compile_args=['-Wno-cpp', '-Wno-unused-function', '-std=c99'])]), script_args=['build_ext', '--inplace'])"
```

Expected artifact:

```text
pycocotools/_mask.cpython-310-x86_64-linux-gnu.so
```

### `model.utils.cython_bbox`

Build command:

```bash
/home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310/bin/python -c "from setuptools import setup, Extension; from Cython.Build import cythonize; import numpy; setup(name='cython_bbox', ext_modules=cythonize([Extension('model.utils.cython_bbox', ['model/utils/bbox.pyx'], include_dirs=[numpy.get_include()])]), script_args=['build_ext', '--inplace'])"
```

Expected artifact:

```text
model/utils/cython_bbox.cpython-310-x86_64-linux-gnu.so
```

## Import Validation

Run from the repository root:

```bash
PYTHONPATH=$PWD/model:$PWD \
/home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310/bin/python -c "import torch, torchvision, numpy; import pycocotools._mask; import model.utils.cython_bbox; from model.roi_layers import ROIAlign, ROIPool, nms; from datasets.vmrd import vmrd; print('imports_ok'); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); print('torchvision', torchvision.__version__, 'numpy', numpy.__version__); print('vmrd_test_images', vmrd('test').num_images)"
```

Observed output:

```text
imports_ok
torch 2.6.0+cu124 cuda 12.4 cuda_available True
device NVIDIA GeForce RTX 4080 SUPER (8, 9)
torchvision 0.21.0+cu124 numpy 1.26.4
vmrd_test_images 450
```

## GPU Training Command

Run from the repository root:

```bash
PYTHONPATH=$PWD/model:$PWD \
/home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310/bin/python main.py \
  --dataset vmrdcompv1 \
  --frame all_in_one \
  --net res101 \
  --cuda \
  --epochs 1 \
  --disp_interval 1 \
  --bs 2 \
  --nw 0
```

For a normal run, remove or increase `--disp_interval 1`; printing every iteration is useful for smoke tests but noisy.

## Snapshot Note

The current config has:

```text
SNAPSHOT_ITERS: 10000
```

One epoch has:

```text
8466 iterations
```

Therefore a 1-epoch run may not save a checkpoint unless `SNAPSHOT_ITERS` is lowered or more than one epoch is run.

## Compatibility Fixes Applied

These repo changes were needed for Python 3.10 / PyTorch 2.6 GPU training.

### `model/roi_layers/nms.py`

Problem:

```text
RuntimeError: unsupported torch version. Supported: 0.4.0 (recommended) and 1.x
```

Fix:

Allow Torch 2.x and use `torchvision.ops.nms`.

### `model/roi_layers/__init__.py`

Problem:

Torch 2.x was rejected by the ROI layer import gate.

Fix:

Allow Torch major version `"2"` in the same path as Torch 1.x.

### `model/roi_layers/roi_align.py`

Problem:

The custom ROIAlign extension was unnecessary and problematic across modern environments.

Fix:

Use:

```python
from torchvision.ops import roi_align as torchvision_roi_align
```

Then call `torchvision_roi_align(...)` in `ROIAlign`, `RoIAlignAvg`, and `RoIAlignMax`.

### `model/roi_layers/roi_pool.py`

Problem:

The custom ROI pool extension is old and not needed with modern Torchvision.

Fix:

Use:

```python
from torchvision.ops import roi_pool as torchvision_roi_pool
```

Then call `torchvision_roi_pool(...)` in `ROIPool.forward`.

### `torch.load(..., weights_only=False)`

Problem:

PyTorch 2.6 changed the default `torch.load` behavior to `weights_only=True`, which failed on legacy pretrained `.pth` files:

```text
RuntimeError: Cannot use weights_only=True with files saved in the legacy .tar format.
```

Fix:

Pass `weights_only=False` for trusted local project checkpoints/pretrained files.

Files updated:

```text
main.py
model/basenet/resnet.py
model/basenet/vgg.py
model/MGN.py
model/AllinOne.py
model/FasterRCNN_VMRN.py
model/VAM.py
```

### `model/utils/bbox.pyx`

Problem:

NumPy 1.26 removed `np.float`:

```text
AttributeError: module 'numpy' has no attribute 'float'
```

Fix:

```python
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
```

### `datasets/imdb.py`

Problem:

Uses removed NumPy alias:

```python
boxes.astype(np.float)
gt_boxes.astype(np.float)
```

Fix:

```python
boxes.astype(np.float64)
gt_boxes.astype(np.float64)
```

## Previously Required Python 3 Fixes

The GPU run also depends on the earlier Python 3 fixes made for CPU training:

```text
datasets/imdb.py: import PIL.Image
pycocotools/refcoco.py: Python 3 print/pickle/unicode/division fixes
datasets/bdds.py: package-relative imports
model/utils/config.py: yaml.safe_load and list-to-tuple config conversion
datasets/vmrd.py: dict_items list conversion
cfgs/vmrdcompv1_all_in_one_res101.yml: PRETRAIN_TYPE: "pytorch"
cfgs/vmrdcompv1_all_in_one_res101.yml: VMRN.OP2L_POOLING_MODE: align
model/MGN.py: integer division for channel counts
roi_data_layer/roibatchLoader.py: shuffle mutable lists instead of range objects
model imports normalized from utils.* to model.utils.*
```

## Important Difference From CPU Env

The old CPU env remains:

```text
/home/user/ehsanullahm1/miniconda3/envs/vmrn_env
```

The new GPU env is separate:

```text
/home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310
```

Use `vmrn_gpu_py310` for RTX 4080 GPU training. Use `vmrn_env` only for the older CPU-compatible setup.
