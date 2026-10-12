# TD-MPC2 for StackCube-v1 (ManiSkill3, CPU + expert warm-start)

Adapted from github.com/nicklashansen/tdmpc2 for ManiSkill3 StackCube-v1,
42-dim unified observation, and CPU-only training.

## What this is
- Expert warm-start (world model + policy on 889 demo trajectories), then online closed-loop TD-MPC2.
- CPU result (model_size=1, ~150k steps): first success ~100k, best 7% SR / reward 183
  with strong MPPI planning (512 samples, horizon 5). SR plateaus at 5-10% on CPU.
- This validates the task is learnable from scratch; it does NOT reproduce the paper 100%.

## Reproduce the paper result (GPU, recommended)
Official repo: github.com/nicklashansen/tdmpc2 (ManiSkill2). Or download the official
ManiSkill2 5-task checkpoint (includes StackCube) from https://www.tdmpc2.com/models and
evaluate directly. Paper setting: GPU, 4M steps, model_size>=5, 3 seeds -> ~100%.

## Run our CPU adaptation (Windows, conda env diffcl10)
cd tdmpc2
python warmstart.py --epochs 120 --out ..\checkpoints\warmstart.pt
python train_stackcube.py --steps 150000 --model-size 1 --num-samples 256 --horizon 3 ^
  --exp-name warmonline2 --warm-start ..\checkpoints\warmstart.pt
python eval_warmstart.py 50 mppi 512 5

Expert demos: ~/.maniskill/demos/StackCube-v1/rl/trajectory.state.pd_ee_delta_pos.physx_cpu.h5

## CPU adaptation notes
- Adam: removed capturable=True; WorldModel.to() skips meta tensors; cuda->cpu.
- Must run inside the tdmpc2/ package dir (avoid tdmpc2 import conflict).
- Windows: add conda env and its Library\bin to PATH.
