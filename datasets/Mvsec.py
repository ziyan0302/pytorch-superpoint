import numpy as np
# import tensorflow as tf
import torch
from pathlib import Path
import torch.utils.data as data
import os

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
from event_utils import gen_discretized_event_volume, normalize_event_volume, gen_event_images
import random

def concat(list_of_names):
    all_datasets = []
    for name in list_of_names:
        one_dataset = Mvsec(name)
        all_datasets.append(one_dataset)
    dataset = torch.utils.data.ConcatDataset(all_datasets)
    return dataset

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

        self.init_var()
        # get files
        dataset_info_path = self.config['dataset_info']
        self.dataset_start_dict = {}
        # Open the dataset_info text file and store dataset info
        with open(dataset_info_path, 'r') as file:
            for line in file:
                path, start_time = line.strip().split()
                datasetName = path.split('/')[-1].split('.')[0]
                self.dataset_start_dict[datasetName] = {"path" : path, "start_time" : float(start_time) }
        
        self.dataset_root = Path(DATA_PATH, 'mvsec_dataset')
        self.dataset_collection = {}
        self.len = 0
        startIdx = 0
        self.dataset_names = []
        self.dataset_startIdx = []
        self.requiredEvents = 10000
        for datasetName, data_info in self.dataset_start_dict.items():
            self.dataset_startIdx.append(startIdx + self.len)
            self.dataset_names.append(datasetName)
            file_path = Path(self.dataset_root, data_info['path'])  # Update with your HDF5 file path
            print("=========================")
            print("file_file: ", file_path)
            with h5py.File(file_path, 'r') as hf:
                image_raw_ts = np.array(hf['davis']['left']['image_raw_ts'])
                start_frame = int(data_info['start_time'] * (1/0.02189))
                self.dataset_collection[datasetName] = {'start_frame' : start_frame, 'len' : (len(image_raw_ts) - start_frame)}
                self.len += self.dataset_collection[datasetName]['len']
            
            img2events_folder = os.path.join(self.dataset_root, "mvsec_cache")
            img2events_file = os.path.join(img2events_folder, f"{datasetName}.npy")
            self.dataset_collection[datasetName]['cache_path'] = img2events_file
            # img2events_file = os.path.join(DATA_PATH, "img2events.npy")
            if not os.path.exists(img2events_file):
                with h5py.File(file_path, 'r') as hf:
                    # event_ts = hf['davis']['left']['events'][start_frame:, 2]   
                    event_ts = hf['davis']['left']['events'][:, 2]   
                    img2events = np.searchsorted(event_ts, image_raw_ts[start_frame:])
                np.save(img2events_file, img2events)

            else:
                self.img2events = np.load(img2events_file)
        


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

        dataset_idx = -1
        for startIdx in self.dataset_startIdx:
            if index < startIdx:
                break
            dataset_idx +=1 
        self.dataset_startIdx
        datasetName = self.dataset_names[dataset_idx]
        data_info = self.dataset_start_dict[datasetName]
        file_path = data_info['path']
        start_frame = self.dataset_collection[datasetName]['start_frame']
        img2events_file = self.dataset_collection[datasetName]['cache_path']
        index = index - self.dataset_startIdx[dataset_idx]

        with h5py.File(Path(self.dataset_root,file_path), 'r') as hf:
            events_raw = hf['davis']['left']['events']
            self.img2events = np.load(img2events_file)

            if (0): # claude's modification
                start = self.img2events[index]
                end = self.img2events[index + 1]
                new_start = np.random.randint(start, end - self.requiredEvents)
                new_end = new_start + self.requiredEvents
                events_within_bounds = events_raw[new_start:new_end, :]

            if (1): # ziyan's modification
                event_start = self.img2events[index]-1
                event_end = event_start + self.requiredEvents
                events_within_bounds = events_raw[event_start:event_end, :]
                ##TODO: can not avoid too few events with in bound at the end of each dataset
                imageN, imageH, imageW = hf['davis']['left']['image_raw'].shape
                self.raw_image = hf['davis']['left']['image_raw'][index+start_frame]
                event_volume = gen_discretized_event_volume(torch.tensor(events_within_bounds).cpu(),
                                                        [self.flow_time_bins*2,
                                                        imageH,
                                                        imageW])
                # print("events_within_bounds: ", events_within_bounds.shape)
                # pdb.set_trace()
                
                event_volume = normalize_event_volume(event_volume)
                self.data = event_volume
                # pdb.set_trace()
                if (0):
                    event_volume.shape
                    event_img = gen_event_images(event_volume.unsqueeze(0).type(torch.float32), prefix=None, device='cpu').squeeze()
                    import cv2
                    image_raw_ts = np.array(hf['davis']['left']['image_raw_ts'])
                    image_raw_ts[index+start_frame]
                    event_ts = hf['davis']['left']['events'][:, 2]   
                    image_raw_ts[index+start_frame] - event_ts[event_start-2]
                    len(hf['davis']['left']['image_raw'])
                    cv2.imwrite("tmp.jpg", hf['davis']['left']['image_raw'][index+start_frame])
                    cv2.imwrite("tmp.jpg", (event_img*255).numpy())

        
        

                

                

            if (0):
                tmp = [self.img2events[1:] - self.img2events[:-1]]
                tmp
                events_raw.shape
                event_ts = hf['davis']['left']['events'][:, 2]
                len(hf['davis']['left']['events'])
                image_raw_ts[0] - event_ts[self.img2events[0]+1]
                event_ts[59] - event_ts[0]
                image_raw_ts[1] - image_raw_ts[0]
                event_start = self.img2events[-1]
                len(event_ts) - self.img2events[-1]
                event_end = event_start + self.requiredEvents
                event_end = start + self.requiredEvents
                len(events_raw)
                events_within_bounds = events_raw[event_start:event_end, :]
                events_within_bounds.shape

                image_raw_ts = np.array(hf['davis']['left']['image_raw_ts'])[137:]
                image_raw_ts.shape


            if (0):

                if not (index < (self.dataset_collection[datasetName]['len'] - self.requiredEvents)):
                    index = random.randint(0, (self.dataset_collection[datasetName]['len'] - self.requiredEvents))
                # indices = (self.image_raw_ts[index] < events_raw[:,2]) &  (events_raw[:,2] < self.image_raw_ts[index+1])
                # events_within_bounds = events_raw[indices]
            
                self.img2events = np.load(img2events_file)


                start = self.img2events[index]
                # end = self.img2events[index + 1]
                end = start + self.requiredEvents
                events_within_bounds = events_raw[start:end, :]

                imageN, imageH, imageW = hf['davis']['left']['image_raw'].shape
                self.raw_image = hf['davis']['left']['image_raw'][index]
                event_volume = gen_discretized_event_volume(torch.tensor(events_within_bounds).cpu(),
                                                        [self.flow_time_bins*2,
                                                        imageH,
                                                        imageW])
                # print("events_within_bounds: ", events_within_bounds.shape)
                # pdb.set_trace()
                
                event_volume = normalize_event_volume(event_volume)
                self.data = event_volume
        
                sequence_set = []
                # prepare seqence_set as in coco
                if (0):
                    self.data.shape
                    tmp_event = hf['davis']['left']['events']
                    tmp_event.shape
                    events_raw.shape
                    event_volume.shape
                # for iData in range(len(self.data)):
                #     sample = {'image': self.data[iData], 'name': iData, 'raw_image': raw_image}
                #     sequence_set.append(sample)


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
            input  = {'image': img_aug, 'name': index, 'raw_image': self.raw_image}
            input.update({'valid_mask': valid_mask})
            
        if self.config['homography_adaptation']['enable']: # disable
            # img_aug = torch.tensor(img_aug)
            homoAdapt_iter = self.config['homography_adaptation']['num']
            homographies = np.stack([self.sample_homography(np.array([1, 1]), shift=0,
                           **self.config['homography_adaptation']['homographies']['params'])
                           for i in range(homoAdapt_iter)])
            print("homographies: ", homographies)
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

        if self.config['warped_pair']['enable']: # enable
            homography = self.sample_homography(np.array([1, 1]), shift=0,
                                        **self.config['warped_pair']['params'])

            ##### use inverse from the sample homography
            homography = np.linalg.inv(homography)
            #####
            inv_homography = np.linalg.inv(homography)

            homography = torch.tensor(homography).type(torch.FloatTensor)
            inv_homography = torch.tensor(inv_homography).type(torch.FloatTensor)

            # homographically transform raw image
            warped_rawImg = torch.tensor(input['raw_image'], dtype=torch.float32)
            warped_rawImg = self.inv_warp_image(warped_rawImg.squeeze(), inv_homography, mode='bilinear').unsqueeze(0) 
            if (0):
                import matplotlib.pyplot as plt
                plt.figure()
                plt.imshow(warped_rawImg.numpy())
                plt.savefig("tmp.jpg")
            H = warped_rawImg.shape[3]
            W = warped_rawImg.shape[4]
            warped_rawImg = warped_rawImg.view(-1, H, W)
            input.update({'warped_rawImg': warped_rawImg})



            # photometric augmentation from original image

            # warp original image
            # warped_img = torch.tensor(self.data, dtype=torch.float32)
            warped_img = self.data.clone().detach().to(torch.float32)
            warped_img = self.inv_warp_image(warped_img.squeeze(), inv_homography, mode='bilinear').unsqueeze(0) 
            # print("warped_img: ", warped_img.shape)
            # print("==========================================")
            H = warped_img.shape[3]
            W = warped_img.shape[4]
            warped_img = warped_img.view(-1, H, W)

            input.update({'warped_img': warped_img})

            # print('erosion_radius', self.config['warped_pair']['valid_border_margin'])
            # pdb.set_trace()
            valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=inv_homography,
                        erosion_radius=self.config['warped_pair']['valid_border_margin'])  # can set to other value
            input.update({'warped_valid_mask': valid_mask})
            input.update({'homographies': homography, 'inv_homographies': inv_homography})

        input.update({'name': index, 'scene_name': "./"})
 
        return input

    def __len__(self):
            return self.len



   