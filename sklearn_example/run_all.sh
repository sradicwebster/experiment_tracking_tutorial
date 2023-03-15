#!/bin/bash
wandb online
python train_sklearn.py
python train_sklearn.py model=svm
python train_sklearn.py model=rforest
wandb offline