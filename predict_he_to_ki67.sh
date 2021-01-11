#!/usr/bin/env bash
python predict_WSI.py --dataroot ./ --name WSI_GAN_he_to_ki67 --model cycle_gan --dataset_mode WSIPrediction --no_dropout --model test --direction BtoA --gpu_ids 1