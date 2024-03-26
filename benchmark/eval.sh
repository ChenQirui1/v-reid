#!/bin/bash

# Path to Images
queryPath_veri="../dataset/VeRi/image_query/"
queryList_veri="./list/veri_query_list.txt"
galleryPath_veri="../dataset/VeRi/image_test/"
galleryList_veri="./list/veri_test_list.txt"


# Number of classes
num_veri=576

# CUDA_VISIBLE_DEVICES=0 python -u test_eval.py $queryPath_veri $queryList_veri $galleryPath_veri $galleryList_veri --dataset veri --backbone resnet50 --weights "./models/resnet50/Car_epoch_50.pth" --save_dir './results/veri/resnet50/' 

CUDA_VISIBLE_DEVICES=0 python -u eval_vehiclenet.py $queryPath_veri $queryList_veri $galleryPath_veri $galleryList_veri --dataset veri --backbone resnet50 --weights "./implementations/vehiclenet/model/ft_ResNet50/net_last.pth" --save_dir './results/veri/vehiclenet/' 