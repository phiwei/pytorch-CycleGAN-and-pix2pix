#!/usr/bin/env bash
python train.py --dataroot ./ --name WSI_GAN_he_to_ki67 --model cycle_gan --batch_size 2 --dataset_mode HE2KI67 --gpu_ids 1