#!/bin/bash
wandb online
python train_sklearn.py
python train_sklearn.py model=rforest
python train_sklearn.py dataset=iris task=classification model=logistic metric=accuracy
python train_sklearn.py dataset=iris task=classification model=svm +preprocessing=minmax metric=accuracy
wandb offline