#!/usr/bin/env bash
python predict_WSI.py --dataroot ./ --name WSI_GAN_ki67_to_he --model cycle_gan --dataset_mode KI672HEPrediction --no_dropout --model test