# Running Long VMRD Jobs With tmux

Use `tmux` for long-running jobs such as training so the process keeps running after the terminal disconnects.

## Pattern

Start a detached session:

```bash
tmux new-session -d -s SESSION_NAME "COMMAND"
```

Attach to watch it interactively:

```bash
tmux attach -t SESSION_NAME
```

Detach without stopping the job:

```text
Ctrl-b then d
```

List running sessions:

```bash
tmux ls
```

Stop a session:

```bash
tmux kill-session -t SESSION_NAME
```

## Training Example

This is the pattern used to start a VMRD `all_in_one` GPU training run for this repository:

```bash
mkdir -p /home/user/ehsanullahm1/thesis/upstream_research_repositories/scene_graph_related_research_papers/Visual-Manipulation-Relationship-Network-Pytorch/output/vmrdcompv1/res101

tmux new-session -d -s vmrn_all_in_one_gpu0 \
  "cd /home/user/ehsanullahm1/thesis/upstream_research_repositories/scene_graph_related_research_papers/Visual-Manipulation-Relationship-Network-Pytorch && \
   CUDA_VISIBLE_DEVICES=0 \
   PYTHONPATH=\$PWD/model:\$PWD \
   /home/user/ehsanullahm1/miniconda3/envs/vmrn_gpu_py310/bin/python main.py \
     --dataset vmrdcompv1 \
     --frame all_in_one \
     --net res101 \
     --cuda \
     --epochs 1 \
     --disp_interval 20 \
     --bs 2 \
     --nw 0 \
     2>&1 | tee -a ./output/vmrdcompv1/res101/train.log"
```

Important details:

- `CUDA_VISIBLE_DEVICES=0` restricts the job to physical GPU `0`. Change this if you want a different GPU.
- `PYTHONPATH=$PWD/model:$PWD` is required because parts of this old codebase still use legacy imports.
- `--dataset vmrdcompv1 --frame all_in_one --net res101` selects the VMRD combined detection, relationship, and grasp model.
- `--cuda` enables CUDA training in `main.py`.
- `--disp_interval 20` prints progress every 20 iterations. Use `--disp_interval 1` for short smoke tests.
- `--bs 2 --nw 0` matches the verified local GPU smoke run settings.
- `output/vmrdcompv1/res101` is where this project saves checkpoints by default for this dataset and backbone.
- `2>&1 | tee -a .../train.log` writes stdout and stderr to both the tmux window and a log file.
- Use the absolute Python path from the `vmrn_gpu_py310` conda environment so the job does not depend on interactive shell activation.

## Monitoring

Watch the log without attaching to tmux:

```bash
tail -f /home/user/ehsanullahm1/thesis/upstream_research_repositories/scene_graph_related_research_papers/Visual-Manipulation-Relationship-Network-Pytorch/output/vmrdcompv1/res101/train.log
```

Check GPU usage:

```bash
nvidia-smi
```

Check which processes are running:

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E 'main.py|vmrn_gpu_py310' | grep -v grep
```

## Notes

If `tmux` cannot be accessed from a sandboxed command, rerun the `tmux` command outside the sandbox.

For reproducibility, put each long run in its own named session and log file. If you want to keep checkpoints from separate experiments apart, pass a different `--save_dir`, for example:

```bash
--save_dir output/experiments/all_in_one_gpu0_run1
```

This project saves checkpoints under:

```text
SAVE_DIR/DATASET/NET
```

For the default command above, that is:

```text
output/vmrdcompv1/res101
```
