# Improved Feature Distillation via Projector Ensemble 
This repository contains the code for Projector Ensemble Distillation (PED) accepted at NeurIPS'22.

Part of the code is modified from [CRD](https://github.com/HobbitLong/RepDistiller).

## Environment
Python==3.6, pytorch==1.8.0, torchvision==0.2.1

## Datasets
You need to manually download [ImageNet](https://www.image-net.org/download.php) dataset and save it in './data'.

## Download the pre-trained teacher networks
sh scripts/fetch_pretrained_teachers.sh

## Run on CIFAR-100
sh scripts/run_cifar.sh

## Run on ImageNet
sh scripts/run_imagenet.sh
