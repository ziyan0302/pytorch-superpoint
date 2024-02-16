import numpy as np
# import tensorflow as tf
import torch
from pathlib import Path
import torch.utils.data as data

# from .base_dataset import BaseDataset
from settings import DATA_PATH, EXPER_PATH
from utils.tools import dict_update
import cv2
from utils.utils import homography_scaling_torch as homography_scaling
from utils.utils import filter_points
import h5py
import pdb


class Mvsec(data.Dataset):
    default_config = {
        'labels': None,
        'cache_in_memory': False,
        'validation_size': 100,
        'truncate': None,
        'preprocessing': {
            'resize': [240, 320]
        },
        'num_parallel_calls': 10,
        'augmentation': {
            'photometric': {
                'enable': False,
                'primitives': 'all',
                'params': {},
                'random_order': True,
            },
            'homographic': {
                'enable': False,
                'params': {},
                'valid_border_margin': 0,
            },
        },
        'warped_pair': {
            'enable': False,
            'params': {},
            'valid_border_margin': 0,
        },
        'homography_adaptation': {
            'enable': False
        }
    }

    def __init__(self, export=False, transform=None, task='train', **config):

        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)

        self.transforms = transform
        self.action = 'train' if task == 'train' else 'val'

        # get files
        file_path = Path(DATA_PATH, 'mvsec_dataset/outdoor_day1_data.hdf5')  # Update with your HDF5 file path
        with h5py.File(file_path, 'r') as hf:
            self.data = hf['davis']['left']['image_raw']
            self.labels = hf['davis']['left']['image_raw_ts'] if config['labels'] else None
    
        pdb.set_trace()
        sequence_set = []
        # prepare seqence_set as in coco
        # for iData in range(len(self.data)):
        #     sample = {'image': self.data[iData], 'name': name, 'points': str(p)}
        #     sequence_set.append(sample)
                    
    def _get_data(self, split_name, **config):
        has_keypoints = True if self.labels else False
        is_training = split_name == 'training'

        def _preprocess(image):
            image = tf.image.rgb_to_grayscale(image)
            if config['preprocessing']['resize']:
                image = pipeline.ratio_preserving_resize(image,
                                                         **config['preprocessing'])
            return image

        # Python function to read labels
        def _read_labels(idx):
            return self.labels[idx] if self.labels else None

        # Create TensorFlow dataset
        dataset = tf.data.Dataset.range(len(self.data))
        dataset = dataset.map(lambda idx: (self.data[idx], _read_labels(idx)))

        # Preprocess images
        dataset = dataset.map(lambda data, label: (_preprocess(data), label))

        # Keep only the first elements for validation
        if split_name == 'validation':
            dataset = dataset.take(config['validation_size'])

        # Cache to avoid always reading from disk
        if config['cache_in_memory']:
            tf.logging.info('Caching data, first access will take some time.')
            dataset = dataset.cache()

        return dataset