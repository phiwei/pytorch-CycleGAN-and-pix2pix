"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import os
import sys
import socket
import numpy as np
import pandas as pd

from tqdm import tqdm
from accimage import Image
from torchvision import transforms
from data.base_dataset import BaseDataset, get_transform


class KI672HEPredictionDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(load_size=500, crop_size=500, preprocess='none', no_flip=True)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)

        # Set paths
        if socket.gethostname().endswith('1'):
            path_dfs_base = '/mnt/hdd8tb/phiwei/projects/breast_ki67/dataframes/'
            path_imgs_base = '/mnt/ssd2tb/data/breast_ki67/'
        elif socket.gethostname().endswith('6'):
            path_dfs_base = '/mnt/hdd16tb/data/breast_ki67/dataframes/'
            path_imgs_base = '/mnt/ssd3tb/data/breast_ki67/'

        path_df_match = os.path.join(path_dfs_base, 'df_match_clinical.csv')
        path_df_tile_HE = os.path.join(path_dfs_base, 'df_tile_he_mpp_0.45366_ts_500_str_500_labeled.pkl')
        path_df_tile_KI67 = os.path.join(path_dfs_base, 'df_tile_ki67_mpp_0.45366_ts_500_str_500_labeled.pkl')

        self.path_base_HE = os.path.join(path_imgs_base, 'tiles_he_mpp_0.45366_ts_500_str_500')
        self.path_base_KI67 = os.path.join(path_imgs_base, 'tiles_ki67_mpp_0.45366_ts_500_str_500')

        # Load dfs
        df_match = pd.read_csv(path_df_match)
        df_match.set_index('PAD', inplace=True)

        df_tile_HE = pd.read_pickle(path_df_tile_HE)
        df_tile_KI67 = pd.read_pickle(path_df_tile_KI67)

        # Only select cancer tiles
        df_tile_HE = df_tile_HE.loc[df_tile_HE['label'] > 0].reset_index(drop=True)
        df_tile_KI67 = df_tile_KI67.loc[df_tile_KI67['label'] > 0].reset_index(drop=True)

        # Drop blury tiles
        df_tile_HE = df_tile_HE.loc[df_tile_HE['blur'] > 250].reset_index(drop=True)
        df_tile_KI67 = df_tile_KI67.loc[df_tile_KI67['blur'] > 250].reset_index(drop=True)

        # Draw random subset
        # df_tile_HE = df_tile_HE.loc[np.random.choice(len(df_tile_HE), size=500)]
        # df_tile_KI67 = df_tile_KI67.loc[np.random.choice(len(df_tile_KI67), size=500)]

        # Construct lists of tile paths
        self.tiles_he = [os.path.join(self.path_base_HE, 
                                      row['slide_name'], 
                                      row['tile_name'])
                         for _, row in tqdm(df_tile_HE.iterrows(), total=len(df_tile_HE))]
        self.tiles_ki67 = [os.path.join(self.path_base_KI67, 
                                        row['slide_name'], 
                                        row['tile_name'])
                           for _, row in tqdm(df_tile_KI67.iterrows(), total=len(df_tile_KI67))]

        # Set up transforms
        self.transform = transforms.ToTensor()

        # Set up length of dataset depending on direction of transform
        if opt.direction == 'AtoB':
            self.len = len(self.tiles_ki67)
        elif opt.direction == 'BtoA':
            self.len = len(self.tiles_he)
        else:
            raise ValueError('Unknown direction.')

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        
        # Obtain data for selected patient
        index_he = np.remainder(index, len(self.tiles_he))
        index_ki67 = np.remainder(index, len(self.tiles_ki67))

        # Make file paths, load images
        path_tile_HE = self.tiles_he[index_he]
        path_tile_KI67 = self.tiles_ki67[index_ki67]

        assert os.path.isfile(path_tile_HE), 'HE image path does not exist: {}'.format(path_tile_HE)
        assert os.path.isfile(path_tile_KI67), 'KI67 image path does not exist: {}'.format(path_tile_KI67)

        img_HE = Image(path_tile_HE)
        img_KI67 = Image(path_tile_KI67)

        # Transform images
        img_HE = self.transform(img_HE)
        img_KI67 = self.transform(img_KI67)

        return {'A': img_KI67, 'B': img_HE, 'A_paths': path_tile_KI67, 'B_paths': path_tile_HE}

    def __len__(self):
        """Return the total number of images."""
        return self.len
