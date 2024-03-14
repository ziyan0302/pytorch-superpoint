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
import sys
sys.path.append("./datasets")
from event_utils import gen_discretized_event_volume


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

        # temporary fixed config
        self.flow_time_bins = 9
        # Update config
        self.config = self.default_config
        self.config = dict_update(self.config, config)

        self.transforms = transform
        self.action = 'train' if task == 'train' else 'val'

        # get files
        self.file_path = Path(DATA_PATH, 'mvsec_dataset/outdoor_day1_data.hdf5')  # Update with your HDF5 file path
        with h5py.File(self.file_path, 'r') as hf:
            self.image_raw_ts = np.array(hf['davis']['left']['image_raw_ts'])
            self.len = len(self.image_raw_ts)

        if (0):
            with h5py.File(self.file_path, 'r') as hf:
                pdb.set_trace()
                self.len
                self.events_raw = np.array(hf['davis']['left']['events'][:1000])
                tmp_events = hf['davis']['left']['events']
                indices = (self.image_raw_ts[1] < tmp_events[:,2]) &  (tmp_events[:,2] < self.image_raw_ts[2])
                events_within_bounds = tmp_events[indices]
                events_within_bounds.shape
                print("init")

        
        self.init_var()
        pass

    def init_var(self):
        torch.set_default_tensor_type(torch.FloatTensor)
        from utils.homographies import sample_homography_np as sample_homography
        from utils.utils import compute_valid_mask
        from utils.photometric import ImgAugTransform, customizedTransform
        from utils.utils import inv_warp_image, inv_warp_image_batch, warp_points, inv_warp_image_batch_mvsec
        self.sample_homography = sample_homography
        self.inv_warp_image = inv_warp_image_batch_mvsec
        self.inv_warp_image_batch = inv_warp_image_batch
        self.compute_valid_mask = compute_valid_mask
        self.ImgAugTransform = ImgAugTransform
        self.customizedTransform = customizedTransform
        self.warp_points = warp_points

        self.enable_homo_train = self.config['augmentation']['homographic']['enable']
        self.enable_homo_val = False

        self.cell_size = 8
        if self.config['preprocessing']['resize']:
            self.sizer = self.config['preprocessing']['resize']

        self.gaussian_label = False
        if self.config['gaussian_label']['enable']:
            self.gaussian_label = True
            y, x = self.sizer

        pass


    def __getitem__(self, index):
        '''

        :param index:
        :return:
            image: tensor (H, W, channel=1)
        '''
        def _read_image(input_image):
            input_image = np.transpose(input_image, (1, 2, 0))
            input_image = cv2.resize(input_image, (self.sizer[1], self.sizer[0]),
                                     interpolation=cv2.INTER_AREA)
            H, W = input_image.shape[0], input_image.shape[1]
            # H = H//cell*cell
            # W = W//cell*cell
            # input_image = input_image[:H,:W,:]
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

            input_image = input_image.astype('float32') / 255.0
            return input_image

        to_floatTensor = lambda x: torch.tensor(x).type(torch.FloatTensor)

        with h5py.File(self.file_path, 'r') as hf:
            events_raw = hf['davis']['left']['events']
            if (index < self.len - 1):
                indices = (self.image_raw_ts[index] < events_raw[:,2]) &  (events_raw[:,2] < self.image_raw_ts[index+1])
                events_within_bounds = events_raw[indices]
                imageN, imageH, imageW = hf['davis']['left']['image_raw'].shape
                event_volume = gen_discretized_event_volume(torch.tensor(events_within_bounds).cpu(),
                                                        [self.flow_time_bins*2,
                                                        imageH,
                                                        imageW])
                
                self.data = event_volume
        
                sequence_set = []
                # prepare seqence_set as in coco
                if (0):
                    self.data.shape
                    tmp_event = hf['davis']['left']['events']
                    tmp_event.shape
                    events_raw.shape
                    event_volume.shape
                for iData in range(len(self.data)):
                    sample = {'image': self.data[iData], 'name': iData}
                    sequence_set.append(sample)


        if (0):
            from numpy.linalg import inv
            sample = self.samples[index]
            input  = {}
            input.update(sample)
            # image
            img_o = _read_image(sample['image'])
            # img_o = sample['image']
            H, W = img_o.shape[0], img_o.shape[1]
            # print(f"image: {image.shape}")
            img_aug = img_o.copy()

            img_aug = torch.tensor(img_aug, dtype=torch.float32).view(-1, H, W)
            valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))
            input.update({'image': img_aug})
            input.update({'valid_mask': valid_mask})

            name = sample['name']

            input.update({'name': name, 'scene_name': "./"}) # dummy scene name
        if (1):
            img_aug = self.data
            valid_mask = self.compute_valid_mask(torch.tensor([self.sizer[1], self.sizer[0]]), inv_homography=torch.eye(3))
            input  = {'image': img_aug, 'name': index}
            input.update({'valid_mask': valid_mask})
            
        if self.config['homography_adaptation']['enable']:
            # img_aug = torch.tensor(img_aug)
            homoAdapt_iter = self.config['homography_adaptation']['num']
            homographies = np.stack([self.sample_homography(np.array([2, 2]), shift=-1,
                           **self.config['homography_adaptation']['homographies']['params'])
                           for i in range(homoAdapt_iter)])
            ##### use inverse from the sample homography
            homographies = np.stack([inv(homography) for homography in homographies])
            homographies[0,:,:] = np.identity(3)
            # homographies_id = np.stack([homographies_id, homographies])[:-1,...]

            ######

            homographies = torch.tensor(homographies, dtype=torch.float32)
            inv_homographies = torch.stack([torch.inverse(homographies[i, :, :]) for i in range(homoAdapt_iter)])

            # images
            warped_img = self.inv_warp_image_batch(img_aug.squeeze().repeat(homoAdapt_iter,1,1,1), inv_homographies, mode='bilinear').unsqueeze(0)
            warped_img = warped_img.squeeze()
            # masks
            valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homographies,
                                                 erosion_radius=self.config['augmentation']['homographic'][
                                                     'valid_border_margin'])
            input.update({'image': warped_img, 'valid_mask': valid_mask, 'image_2D':img_aug})
            input.update({'homographies': homographies, 'inv_homographies': inv_homographies})

        if self.config['warped_pair']['enable']:
            homography = self.sample_homography(np.array([2, 2]), shift=-1,
                                        **self.config['warped_pair']['params'])

            ##### use inverse from the sample homography
            homography = np.linalg.inv(homography)
            #####
            inv_homography = np.linalg.inv(homography)

            homography = torch.tensor(homography).type(torch.FloatTensor)
            inv_homography = torch.tensor(inv_homography).type(torch.FloatTensor)

            # photometric augmentation from original image

            # warp original image
            warped_img = torch.tensor(self.data, dtype=torch.float32)
            warped_img = self.inv_warp_image(warped_img.squeeze(), inv_homography, mode='bilinear').unsqueeze(0) 
            # print("warped_img: ", warped_img.shape)
            # print("==========================================")
            H = warped_img.shape[3]
            W = warped_img.shape[4]
            warped_img = warped_img.view(-1, H, W)

            input.update({'warped_img': warped_img})

            # print('erosion_radius', self.config['warped_pair']['valid_border_margin'])
            valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homography,
                        erosion_radius=self.config['warped_pair']['valid_border_margin'])  # can set to other value
            input.update({'warped_valid_mask': valid_mask})
            input.update({'homographies': homography, 'inv_homographies': inv_homography})

        input.update({'name': index, 'scene_name': "./"})
 
        return input

    def __len__(self):
            return self.len



   