# Batch_D3PG
PyTorch implementation of batch D3PG, 
which uses n-step Bellman update and parallel experience sampling.
The Detail of this algorithm can be found in this paper:

Barth-Maron, Gabriel, et al. 
"Distributed distributional deterministic policy gradients." 
arXiv preprint arXiv:1804.08617 (2018).

## Requirements 

- pytorch 1.4.0
- tensorboard
- numpy
- tqdm 
- gym
- baselines
- pybullet (optional)

## Setup

You can use the provided `requirements.txt` file to install necessary dependencies.

```bash
$ pip install -r requirements.txt
```

## Training D3PG agents

For example, to train a d3pg agent using 12 processes for pybullet ant locomotion task as follows:

```bash
$ python train.py --task-id=AntBulletEnv-v0 --num-processes=12 --num-env-steps=5000000
```

You can also monitor the training process and perform hyper-parameters tuning using tensorboard:

```bash
$ tensorboard --logdir=log
```