#!/usr/bin/env bash
python predict_WSI.py --dataroot ./ --name WSI_GAN_he_to_ki67 --model cycle_gan --dataset_mode HE2KI67Prediction --no_dropout --model test --gpu_ids 0