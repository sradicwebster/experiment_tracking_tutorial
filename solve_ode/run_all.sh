#!/bin/bash
wandb online
python hydra_configs_wandb.py
python hydra_configs_wandb.py method=rk4
python hydra_configs_wandb.py method=exact
wandb offline