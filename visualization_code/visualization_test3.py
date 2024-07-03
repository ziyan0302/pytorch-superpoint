"""
Visaulization code:
1. Find the corner points in image and warped image
2. Extract description for corner points on image and those on warped image
3. indexize the corner points on image
4. find the most similar point on warped image
"""
import pdb
import numpy as np
import torch
import torch.optim
import torch.nn.functional as F
import torch.utils.data

import sys
sys.path.append("../pytorch-superpoint")
from tqdm import tqdm
from utils.loader import dataLoader
import logging

from pathlib import Path
from models.model_wrap import SuperPointFrontend_torch
import cv2 
import matplotlib.pyplot as plt
import yaml
from utils.loader import dataLoader as dataLoader

import torch.nn.functional as F
import cv2
from utils.utils import normPts
from utils.utils import filter_points
from datasets.event_utils import gen_event_images
import visualization_utils as vutil
import time

np.random.seed(42)
torch.manual_seed(42) 


@torch.no_grad()
class Val_model_heatmap(SuperPointFrontend_torch):
    def __init__(self, config, device='cpu', verbose=False):
        self.config = config
        self.model = self.config['name']
        self.params = self.config['params']
        self.weights_path = self.config['pretrained']
        self.device=device

        ## other parameters

        # self.name = 'SuperPoint'
        # self.cuda = cuda
        self.nms_dist = self.config['nms']
        self.conf_thresh = self.config['detection_threshold']
        # self.nn_thresh = self.config['nn_thresh']  # L2 descriptor distance for good match.
        self.cell = 8  # deprecated
        self.cell_size = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.
        self.sparsemap = None
        self.heatmap = None # np[batch, 1, H, W]
        self.pts = None
        self.pts_subpixel = None
        ## new variables
        self.pts_nms_batch = None
        self.desc_sparse_batch = None
        self.patches = None
        pass


    def loadModel(self):
        # model = 'SuperPointNet'
        # params = self.config['model']['subpixel']['params']
        from utils.loader import modelLoader
        self.net = modelLoader(model=self.model, **self.params)

        checkpoint = torch.load(self.weights_path,
                                map_location=lambda storage, loc: storage)
        self.net.load_state_dict(checkpoint['model_state_dict'])

        self.net = self.net.to(self.device)
        logging.info('successfully load pretrained model from: %s', self.weights_path)
        pass
    
    def run(self, images):
        """
        input: 
            images: tensor[batch(1), 1, H, W]

        """
        from Train_model_heatmap import Train_model_heatmap
        from utils.var_dim import toNumpy
        train_agent = Train_model_heatmap

        with torch.no_grad():
            outs = self.net(images)
        return outs

filename = "/home/ziyan/02_research/pytorch-superpoint/configs/superpoint_mvsec_test_heatmap.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_tensor_type(torch.FloatTensor)
with open(filename, "r") as f:
    config = yaml.safe_load(f)


# data = dataLoader(config, dataset='hpatches')
task = config["data"]["dataset"]

data = dataLoader(config, dataset=task, warp_input=True, shuffle=False)
# test_set, test_loader = data['test_set'], data['test_loader']
train_loader, val_loader = data["train_loader"], data["val_loader"]

val_agent = Val_model_heatmap(config['model'], device="cuda")
val_agent.loadModel()


from tqdm import tqdm
# data_iter = iter(val_loader)

# Define the number of frames to skip
skip_steps = 50

data_iter = iter(val_loader)



# for index, (sample, nextSample) in tqdm(enumerate(zip(data_iter, data_iter))):
for index, sample in tqdm(enumerate(data_iter)):
    for _ in range(skip_steps-1):
        next(data_iter)
    nextSample = next(data_iter)
    

        

    img, warped_img, outs, coarse_desc = vutil.dataExtractor(sample, val_agent)
    nextImg, _, nextOuts, nextCoarse_desc = vutil.dataExtractor(nextSample, val_agent)
    image_a_pred = coarse_desc
    nextImage_a_pred = nextCoarse_desc

    # get the normed interest points and description
    uv_a, matches_a_descriptors = vutil.uv2descriptorExtractor(image_a_pred, img)
    pdb.set_trace()
    
    if(0):

        # Generate grid points
        x = np.linspace(-1, 1, 100)
        y = np.linspace(-1, 1, 100)

        # Create meshgrid
        X, Y = np.meshgrid(x, y)

        idxs = np.arange(0,uv_a.shape[0],11)
        tmp = uv_a[idxs]
        tmp.shape

    uv_aNext, matches_aNext_descriptors = vutil.uv2descriptorExtractor(nextImage_a_pred, nextImg)
    
    if (1): # randomly filter out some interest points
        idxs_a = np.arange(0,uv_a.shape[0],11)
        uv_a = uv_a[idxs_a]
        matches_a_descriptors = matches_a_descriptors[idxs_a]

        idxs_next = np.arange(0,uv_aNext.shape[0],11)
        uv_aNext = uv_aNext[idxs_next]
        matches_aNext_descriptors = matches_aNext_descriptors[idxs_next]


        

    Hc, Wc = image_a_pred.shape[2], image_a_pred.shape[3]
    img_shape = (Hc, Wc)

    homographies_H = sample["homographies"]
    homographies_H = vutil.scale_homography_torch(homographies_H, img_shape, shift=(-1, -1))
   
    print("matches_a_descriptors: ", matches_a_descriptors.shape)
    print("matches_aNext_descriptors: ", matches_aNext_descriptors.shape)

    # Compute distances between descriptors
    distances_matrix = vutil.compute_distances(matches_a_descriptors.cpu().numpy(), matches_aNext_descriptors.cpu().numpy())
    column_index_with_smallest_value = np.argmax(distances_matrix, axis=1)
    matches = []
    for row_index, col_index in enumerate(column_index_with_smallest_value):
        # Create a cv2.DMatch object with query index as row_index, train index as col_index,
        # and optionally set distance to 0
        dmatch = cv2.DMatch(row_index, col_index, 0)
        # Append the created dmatch to the list
        matches.append(dmatch)


    input_uv = vutil.uvListTransformer(uv_a, Wc, Hc)
    pred_uv = vutil.uvListTransformer(uv_aNext, Wc, Hc)
    

    img1 = sample['raw_image'].squeeze().cpu().numpy()
    img2 = nextSample['raw_image'].squeeze().cpu().numpy()
    # img2 = np.floor(sample['warped_rawImg'].squeeze().cpu().numpy()).astype(np.uint8)
    
    if (0): ## draw matched picture
        img3 = cv2.drawMatches(img1,  # img1
                            input_uv,  # keypoints1
                            img2,  # img2
                            pred_uv,  # keypoints2
                            matches,  # matches
                            None,  # outImg
                            matchColor=(0, 255, 0),  # matchColor (green)
                            singlePointColor=(255, 0, 0),  # singlePointColor (red)
                            matchesMask=None)  # matchesMask
                        


        cv2.imwrite("img3.jpg", img3)

    if (0): ## show event volume visualization
        event_img = gen_event_images(sample['image'].type(torch.float32), prefix=None, device='cpu').squeeze()
        cv2.imwrite("event.jpg", (event_img*255).numpy())
        time.sleep(0.1)

    if (1): # show interest points only
        out1 = cv2.drawKeypoints(img2, input_uv, None)
        cv2.imwrite("img1.jpg", out1)
         

    if (1): ## draw lines for matched points on the same image
        out1 = cv2.drawKeypoints(img2, input_uv, None)
        out2 = cv2.drawKeypoints(out1, pred_uv, None)
        points1 = np.float32([input_uv[match.queryIdx].pt for match in matches])
        points2 = np.float32([pred_uv[match.trainIdx].pt for match in matches])
        for pt1, pt2 in zip(points1, points2): 
            pt1 = tuple(map(int, pt1))  # Convert to integer coordinates
            pt2 = tuple(map(int, pt2))  # Convert to integer coordinates
            cv2.line(out2, pt1, pt2, (0, 255, 0), 2)  # Draw a line from pt1 to pt2
        cv2.imwrite("out2.jpg", out2)
        
    # pdb.set_trace()


print("end")