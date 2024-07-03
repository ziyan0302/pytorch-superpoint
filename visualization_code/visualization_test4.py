"""
Visaulization code:
show all the event images 
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



filename = "/home/ziyan/02_research/pytorch-superpoint/configs/superpoint_mvsec_test_heatmap.yaml"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(filename, "r") as f:
    config = yaml.safe_load(f)
task = config["data"]["dataset"]

data = dataLoader(config, dataset=task, warp_input=True, shuffle=False)
# test_set, test_loader = data['test_set'], data['test_loader']
train_loader, val_loader = data["train_loader"], data["val_loader"]

from tqdm import tqdm
data_iter = iter(val_loader)

# Define the number of frames to skip
skip_steps = 10

# for index, (sample, nextSample) in tqdm(enumerate(zip(data_iter, data_iter))):
for index, sample in tqdm(enumerate(data_iter)):
    if (0): ## show sequentual raw images, event volume visualization
        for _ in range(skip_steps-1):
            next(data_iter)
        nextSample = next(data_iter)

        img1 = sample['raw_image'].squeeze().cpu().numpy()
        img2 = nextSample['raw_image'].squeeze().cpu().numpy()
            
        event_img = gen_event_images(sample['image'].type(torch.float32), prefix=None, device='cpu').squeeze()
        nextEvent_img = gen_event_images(nextSample['image'].type(torch.float32), prefix=None, device='cpu').squeeze()
            
        event_img = (event_img*255).numpy()
        nextEvent_img = (nextEvent_img*255).numpy()
        # cv2.imwrite("event.jpg", (event_img*255).numpy())

        # Create a blank canvas to draw the images on
        canvas_height = max(event_img.shape[0] + img1.shape[0], nextEvent_img.shape[0] + img2.shape[0])
        canvas_width = max(event_img.shape[1] + nextEvent_img.shape[1], img1.shape[1] + img2.shape[1])
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

        # Draw the first image on the canvas
        canvas[0:event_img.shape[0], 0:event_img.shape[1]] = event_img

        # Draw the second image on the canvas
        canvas[0:nextEvent_img.shape[0], event_img.shape[1]:event_img.shape[1]+nextEvent_img.shape[1]] = nextEvent_img
        canvas[event_img.shape[0]:(event_img.shape[0]+img1.shape[0]), 0:img1.shape[1]] = img1
        canvas[nextEvent_img.shape[0]:(nextEvent_img.shape[0]+img2.shape[0]), img1.shape[1]:(img2.shape[1]+img2.shape[1])] = img2
        cv2.imwrite("subsequentEvents.jpg", canvas)

        time.sleep(0.5)
    if (0): ## show warped image pair and event pair
        for _ in range(skip_steps-1):
            next(data_iter)
        nextSample = next(data_iter)

        img1 = sample['raw_image'].squeeze().cpu().numpy()
        img2 = sample['warped_rawImg'].squeeze().cpu().numpy()
        img2.shape
        event_img = gen_event_images(sample['image'].type(torch.float32), prefix=None, device='cpu').squeeze()
        warpedEvent_img = gen_event_images(sample['warped_img'].type(torch.float32), prefix=None, device='cpu').squeeze()
            
        event_img = (event_img*255).numpy()
        warpedEvent_img = (warpedEvent_img*255).numpy()
        # cv2.imwrite("event.jpg", (event_img*255).numpy())

        # Create a blank canvas to draw the images on
        canvas_height = max(event_img.shape[0] + img1.shape[0], warpedEvent_img.shape[0] + img2.shape[0])
        canvas_width = max(event_img.shape[1] + warpedEvent_img.shape[1], img1.shape[1] + img2.shape[1])
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

        # Draw the first image on the canvas
        canvas[0:event_img.shape[0], 0:event_img.shape[1]] = event_img

        # Draw the second image on the canvas
        canvas[0:warpedEvent_img.shape[0], event_img.shape[1]:event_img.shape[1]+warpedEvent_img.shape[1]] = warpedEvent_img
        canvas[event_img.shape[0]:(event_img.shape[0]+img1.shape[0]), 0:img1.shape[1]] = img1
        canvas[warpedEvent_img.shape[0]:(warpedEvent_img.shape[0]+img2.shape[0]), img1.shape[1]:(img2.shape[1]+img2.shape[1])] = img2
        cv2.imwrite("subsequentEvents.jpg", canvas)

        time.sleep(0.5)
