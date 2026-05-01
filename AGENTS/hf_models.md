# Hugging Face Model Backup

This project's completed VMRD Faster R-CNN detection run was backed up to Hugging Face.

## Repository

- Hugging Face repo: `iamehsanullah/vmrd-faster-rcnn-res101-detection`
- URL: `https://huggingface.co/iamehsanullah/vmrd-faster-rcnn-res101-detection`
- Repo type: model
- Visibility at creation time: private

## Uploaded Artifacts

```text
README.md
checkpoints/faster_rcnn_1_20_4146.pth
configs/vmrdcompv1_faster_rcnn_res101.yml
docs/download_dataset.md
docs/running_training_cpu.md
docs/running_training_gpu.md
logs/eval_1_20_4146.log
logs/train.log
results/det_res.pkl
```

## Model Record

- Task: object detection only
- Model frame: `faster_rcnn`
- Backbone: `res101`
- Dataset argument: `vmrdcompv1`
- Training split: `vmrd_compv1_trainval`
- Evaluation split: `vmrd_compv1_test`
- Test images: `450`
- Checkpoint evaluated: `faster_rcnn_1_20_4146.pth`
- Evaluation metric: VOC07 Python evaluation from this repo
- Mean AP: `0.9395`

This result does not evaluate grasp prediction or manipulation relationship prediction.

## Local Source Files

The uploaded checkpoint came from:

```text
output/faster_rcnn_vmrdcompv1_res101_gpu/vmrdcompv1/res101/faster_rcnn_1_20_4146.pth
```

The final evaluation log came from:

```text
output/faster_rcnn_vmrdcompv1_res101_gpu/eval_1_20_4146.log
```

The full training log came from:

```text
output/faster_rcnn_vmrdcompv1_res101_gpu/train.log
```

## Commands Used

Repository creation:

```bash
/home/user/ehsanullahm1/miniconda3/bin/huggingface-cli repo create \
  vmrd-faster-rcnn-res101-detection \
  --type model \
  --private \
  --yes
```

Upload:

```bash
/home/user/ehsanullahm1/miniconda3/bin/huggingface-cli upload \
  iamehsanullah/vmrd-faster-rcnn-res101-detection \
  hf_backup_vmrd_faster_rcnn_res101_detection \
  . \
  --repo-type model \
  --commit-message "Backup VMRD Faster R-CNN detection checkpoint"
```

Verification:

```bash
/home/user/ehsanullahm1/miniconda3/bin/python -c \
  "from huggingface_hub import list_repo_files; files=list_repo_files('iamehsanullah/vmrd-faster-rcnn-res101-detection', repo_type='model'); print('\n'.join(files))"
```

## Notes

- The VMRD dataset itself was not uploaded.
- The pretrained base ResNet weights were not uploaded.
- The local staging folder `hf_backup_vmrd_faster_rcnn_res101_detection/` was used for upload and can be removed after confirming the remote backup.
- The uploaded checkpoint depends on this repo's code and the compatibility patches made for the current Python/PyTorch/CUDA environment.
