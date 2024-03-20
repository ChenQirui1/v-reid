#!/bin/bash

# Path to Images
ImagePath_veri="../dataset/VeRi/image_train/"
TrainList_veri="./list/veri_train_list.txt"
# Number of classes
num_veri=576

CUDA_VISIBLE_DEVICES=0 nohup python -u train.py $ImagePath_veri $TrainList_veri -n $num_veri --batch_size 64 --val_step 3 --write-out --start_epoch 0 --backbone resnet50 --save_dir './models/resnet50/' --epochs 10 > logs/resnet50.log 2>&1 &