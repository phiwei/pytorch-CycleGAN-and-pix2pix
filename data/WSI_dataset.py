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
# from data.image_folder import make_dataset
# from PIL import Image

sys.path.append('/mnt/hdd8tb/phiwei/repos/chimetorch')
from chimetorch import RotateMirror


class WSIDataset(BaseDataset):
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
        path_df_tile_HE = os.path.join(path_dfs_base, 'df_tile_0.45366.pkl')
        path_df_tile_KI67 = os.path.join(path_dfs_base, 'df_tile_ki67_mpp_0.45366_ts_500_str_500.pkl')

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

        # Iterate through all patient IDs and initialise dataset
        patient_list = list()
        for ID in tqdm(df_match.index):
            
            # Get slidenames and lists of tilenames
            slidename_HE = df_match.loc[ID, 'HE_slidename'][:-len('.ndpi')]
            slidename_KI67 = df_match.loc[ID, 'KI67_slidename'][:-len('.ndpi')]

            tiles_HE = df_tile_HE['tile_name'].loc[df_tile_HE['slide_name'] == slidename_HE].values.tolist()
            tiles_KI67 = df_tile_KI67['tile_name'].loc[df_tile_KI67['slide_name'] == slidename_KI67].values.tolist()

            # Write to dict, append
            dict_curr = dict()
            dict_curr['slidename_HE'] = slidename_HE
            dict_curr['slidename_KI67'] = slidename_KI67
            dict_curr['tiles_HE'] = tiles_HE
            dict_curr['tiles_KI67'] = tiles_KI67
            patient_list.append(dict_curr)

        self.patient_list = patient_list


        self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             RotateMirror()])

        self.len = 10000
        self.n_patients = len(df_match)

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
        index = np.remainder(index, self.n_patients)
        dict_tmp = self.patient_list[index]
        slidename_HE = dict_tmp['slidename_HE']
        slidename_KI67 = dict_tmp['slidename_KI67']

        # Randomly draw a tile from H&E and Ki67
        tilename_HE = np.random.choice(dict_tmp['tiles_HE'])
        tilename_KI67 = np.random.choice(dict_tmp['tiles_KI67'])

        # Make file paths, load images
        path_tile_HE = os.path.join(self.path_base_HE, slidename_HE, tilename_HE)
        path_tile_KI67 = os.path.join(self.path_base_KI67, slidename_KI67, tilename_KI67)

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
