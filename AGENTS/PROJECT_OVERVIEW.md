# Project Overview

This repository is a PyTorch research codebase for robotic visual manipulation.
Given an image of stacked or overlapping objects, it can predict object
detections, manipulation relationships between objects, and robot grasp
rectangles.

The main entry point is:

```bash
main.py
```

The project is based on Faster R-CNN style object detection code and extends it
with visual manipulation relationship prediction and grasp detection.

## What The Project Produces

Depending on the selected `--frame`, the project can produce:

- Object detections: object class, bounding box, and confidence score.
- Manipulation relationships: pairwise object relations labeled as `FATHER`,
  `CHILD`, or `NOREL`.
- Grasp detections: oriented grasp rectangles for robot grippers.
- Combined predictions in the `all_in_one` model: object boxes, object classes,
  relationships, and grasps from one network.

The main reported metrics are:

- `mAP`: mean Average Precision for object detection.
- `mAP-G`: mean Average Precision with grasp detection.
- `Rel-IA`: relationship image accuracy.
- Object relationship recall and precision.
- Grasp accuracy or grasp mAP, depending on dataset and model.

The upstream README reports these reference results:

```text
Faster R-CNN ResNet-101 on VOC: mAP 71.5
FPN ResNet-101 on VOC: mAP 73.9
ROI-GD ResNet-101 on VMRD: mAP 94.5, mAP-G 75.7
F-VMRN ResNet-101 on VMRD: mAP 95.6, Rel-IA 64.7
F-VMRN VGG-16 on VMRD: mAP 95.0, Rel-IA 68.7
```

These are upstream reported numbers. In this local workspace, `output/` does not
currently contain saved checkpoints or full evaluation results.

## Local Dataset State

The VMRD dataset is expected at:

```text
data/VMRD/vmrdcompv1
```

This workspace currently contains:

```text
4683 JPEG images
4683 XML object/relationship annotations
4683 grasp annotation text files
```

The VMRD split files are:

```text
data/VMRD/vmrdcompv1/ImageSets/Main/trainval.txt
data/VMRD/vmrdcompv1/ImageSets/Main/test.txt
```

The dataset parser is:

```text
datasets/vmrd.py
```

## Input Annotation Format

Each VMRD image has an XML annotation in:

```text
data/VMRD/vmrdcompv1/Annotations
```

Each object annotation contains:

- object class name
- bounding box
- object index
- parent object indices
- child object indices

The parent/child annotations define the manipulation relationship graph.
For example, if object A is listed as the father of object B, the model treats
that as a directional relationship between the two objects.

Each image also has a grasp file in:

```text
data/VMRD/vmrdcompv1/Grasps
```

Each grasp line stores:

- 8 coordinates describing a rotated grasp rectangle
- the object index that owns the grasp
- a trailing label field

The 8 coordinates are the four corner points of the gripper rectangle.

## Supported Model Frames

`main.py` supports these frame names:

```text
faster_rcnn
ssd
fpn
faster_rcnn_vmrn
ssd_vmrn
all_in_one
fcgn
mgn
vam
```

Important frames:

- `faster_rcnn`: object detection only.
- `faster_rcnn_vmrn`: Faster R-CNN plus manipulation relationship prediction.
- `fcgn`: grasp detection only.
- `mgn`: object detection plus ROI-based grasp detection.
- `all_in_one`: object detection, relationship prediction, and grasp detection
  together.

## End-To-End Pipeline

### 1. Configuration And Dataset Selection

The CLI arguments are parsed in:

```text
model/utils/config.py
```

For example:

```bash
python main.py --dataset vmrdcompv1 --frame all_in_one --net res101 --cuda
```

For `vmrdcompv1`, the code maps:

```text
training set: vmrd_compv1_trainval
test set:     vmrd_compv1_test
```

The matching config file is selected from `cfgs/`, for example:

```text
cfgs/vmrdcompv1_all_in_one_res101.yml
```

### 2. ROIDB Construction

The dataset is converted into a region-of-interest database by:

```text
roi_data_layer/roidb.py
```

The ROIDB stores per-image metadata:

- image path
- image width and height
- object boxes
- object class labels
- grasp rectangles
- grasp owner indices
- parent/child relationship lists
- overlap metadata used during training

For VMRD trainval, the dataset class also creates rotated training variants.

### 3. Batch Loading And Preprocessing

Batch loaders live in:

```text
roi_data_layer/roibatchLoader.py
```

They resize and normalize images, apply augmentations, crop or pad images for
batching, and convert annotations into tensors.

For `all_in_one`, one training sample contains:

```text
image tensor
image info
ground-truth object boxes
ground-truth grasp rectangles
number of object boxes
number of grasps
relationship matrix
grasp owner indices
```

The relationship matrix is generated from XML parent/child lists.

### 4. Feature Extraction

The model uses a backbone such as ResNet-101 or VGG-16 to convert the image into
feature maps.

Backbone code is under:

```text
model/basenet
```

### 5. Object Detection Branch

For Faster R-CNN based frames:

1. The RPN proposes object regions.
2. ROI Align or ROI Pool extracts per-object features.
3. A classifier predicts object class probabilities.
4. A box regressor refines bounding boxes.

The object detection outputs are:

```text
rois
class probabilities
bounding box predictions
RPN classification loss
RPN box loss
RCNN classification loss
RCNN box loss
```

### 6. Relationship Branch: VMRN

Relationship logic is implemented in:

```text
model/Detectors.py
model/FasterRCNN_VMRN.py
```

For every object pair, the model builds features for:

- object 1
- object 2
- the union region covering both objects

These features are passed through a relationship classifier.

Relationship labels are defined in `model/utils/config.py`:

```text
FATHER = 1
CHILD  = 2
NOREL  = 3
```

At inference time, relationship probabilities are converted into a full
object-by-object relationship matrix.

### 7. Grasp Branch: MGN / FCGN

Grasp logic is implemented in:

```text
model/MGN.py
model/FCGN.py
model/fcgn
```

The grasp branch works inside object ROIs. It predicts oriented grasp boxes
using anchor-like grasp candidates.

For each object ROI, it predicts:

- grasp anchor confidence
- grasp box regression offsets
- final rotated grasp rectangles

The default all-in-one VMRD config uses grasp anchor angles:

```text
[-67.5, -22.5, 22.5, 67.5]
```

### 8. All-In-One Model

The combined model is:

```text
model/AllinOne.py
```

It combines:

- Faster R-CNN object detection
- VMRN relationship prediction
- ROI-based grasp detection

Its training loss combines:

- RPN classification loss
- RPN box regression loss
- RCNN classification loss
- RCNN box regression loss
- relationship classification loss
- relationship regularization loss
- grasp classification loss
- grasp box regression loss

## Inference And Evaluation

Evaluation is handled by:

```text
evaluate_model() in main.py
```

During testing, the code loops over the test images and collects:

```text
all_boxes
all_grasp
all_rel
```

Post-processing functions are in:

```text
model/utils/net_utils.py
```

Important functions:

- `objdet_inference`: decodes object boxes, applies score thresholding and NMS.
- `grasp_inference`: decodes grasp rectangles.
- `objgrasp_inference`: combines object and grasp predictions.
- `rel_prob_to_mat`: converts relationship probabilities into a relationship
  matrix.

Evaluation then computes object detection, grasp, and relationship metrics.

## Optional Visualizations

If `--vis` is used, the project writes visualization images under:

```text
output/<dataset>/data_vis/train
output/<dataset>/data_vis/test
```

The visualization code can draw:

- object boxes
- grasp rectangles
- grasp ownership
- manipulation relationship trees

## Checkpoints And Outputs

Training checkpoints are saved under:

```text
output/<dataset>/<net>
```

For example:

```text
output/vmrdcompv1/res101/all_in_one_1_1_10000.pth
```

Detection results may also be written as:

```text
det_res.pkl
```

Per-class evaluation files are written under the dataset result directory, such
as:

```text
data/VMRD/results/vmrdcompv1/Main
```

## Example Commands

Train the all-in-one model:

```bash
python main.py --dataset vmrdcompv1 --frame all_in_one --net res101 --cuda
```

Test a checkpoint:

```bash
python main.py \
  --test \
  --dataset vmrdcompv1 \
  --frame all_in_one \
  --net res101 \
  --cuda \
  --checkpoint 1000 \
  --checkepoch 1
```

A local CPU smoke run has previously reached real optimizer iterations with
losses around:

```text
iter 1: loss 13.3146
iter 2: loss 11.6528
iter 3: loss 11.4276
```

A full CPU run is very slow. The intended path is GPU training with `--cuda`.

## Short Summary

This project takes an image of multiple objects, detects each object, predicts
the manipulation order or stacking relationship between object pairs, and
predicts robot grasp rectangles. The `all_in_one` frame is the most complete
pipeline because it performs object detection, relationship prediction, and
grasp detection in one model.
