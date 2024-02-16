import numpy as np
import tensorflow as tf
from pathlib import Path

from .base_dataset import BaseDataset
from .utils import pipeline
from superpoint.settings import DATA_PATH, EXPER_PATH
import h5py


class MVSEC(BaseDataset):
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
    }

    def _init_dataset(self, **config):
        file_path = Path(DATA_PATH, 'outdoor_day1_data.hdf5')  # Update with your HDF5 file path
        with h5py.File(file_path, 'r') as hf:
            self.data = hf['davis']['left']['image_raw']
            self.labels = hf['davis']['left']['image_raw_ts'] if config['labels'] else None
    
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