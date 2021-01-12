#!/usr/bin/env bash
python train.py --dataroot ./ --name WSI_GAN_ki67_to_he --model cycle_gan --batch_size 2 --dataset_mode KI672HE