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


def findInterestPoints(img, blockSize=2, ksize=1, k=0.1):
    img_mean = torch.mean(img, dim=1)
    result = torch.where(img_mean > 0, torch.tensor(1), torch.tensor(0)).numpy()[0]
    # Apply Harris corner detection
    # Adjust the parameter k to control sensitivity to corners
    # You may need to experiment with different values of k
    dst = cv2.cornerHarris(np.uint8(result), blockSize, ksize, k)
    dst.max()
    # Threshold the response to highlight corners
    # You may need to adjust the threshold value according to your image
    threshold = 0.1 * dst.max()
    # Find coordinates where dst exceeds the threshold
    coordinates = np.argwhere(dst > threshold)
    # make uv to be in [0,1]
    coordinates = coordinates/np.array([img.shape[-2], img.shape[-1]])
    return coordinates
def descriptor_reshape(descriptors, Hc, Wc):
    descriptors = descriptors.view(-1, Hc * Wc).transpose(0, 1)  # torch [D, H, W] --> [H*W, d]
    descriptors = descriptors.unsqueeze(0)  # torch [1, H*W, D]
    return descriptors
def normPts(pts, shape):
    """
    normalize pts to [-1, 1]
    :param pts:
        tensor (y, x)
    :param shape:
        tensor shape (y, x)
    :return:
    """
    pts = pts/shape*2 - 1
    return pts
def deNormPts(pts, shape):
    """
    normalize pts to [shape0, shape1]
    :param pts:
        tensor (y, x) [-1, 1]
    :param shape:
        tensor shape (y, x)
    :return:
    """
    pts = (pts + 1) / 2 * shape
    return pts
def sampleDescriptors(image_a_pred, matches_a, mode, norm=False):
    matches_a.unsqueeze_(0).unsqueeze_(1)
    matches_a_descriptors = F.grid_sample(image_a_pred, matches_a, mode=mode, align_corners=True)
    matches_a_descriptors = matches_a_descriptors.squeeze().transpose(0,1)
    if norm:
        dn = torch.norm(matches_a_descriptors, p=2, dim=1) # Compute the norm of b_descriptors
        matches_a_descriptors = matches_a_descriptors.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return matches_a_descriptors

def bfMatching(des1, des2):
    # create BFMatcher object
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    
    # Match descriptors.
    matches = bf.match(des1.cpu().numpy(),des2.cpu().numpy())
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def scale_homography(H, shape, shift=(-1,-1)):
    height, width = shape[0], shape[1]
    trans = np.array([[2./width, 0., shift[0]], [0., 2./height, shift[1]], [0., 0., 1.]])
    H_tf = np.linalg.inv(trans) @ H @ trans
    return H_tf

def scale_homography_torch(H, shape, shift=(-1,-1), dtype=torch.float32):
    height, width = shape[0], shape[1]
    trans = torch.tensor([[2./width, 0., shift[0]], [0., 2./height, shift[1]], [0., 0., 1.]], dtype=dtype)
    # print("torch.inverse(trans) ", torch.inverse(trans))
    # print("H: ", H)
    H_tf = torch.inverse(trans) @ H.cpu() @ trans
    return H_tf

def warp_coor_cells_with_homographies(coor_cells, homographies, uv=False, device='cpu'):
    from utils.utils import warp_points
    # warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
    # warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
    warped_coor_cells = coor_cells
    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:,1], warped_coor_cells[:,0]), dim=1) # (y, x) to (x, y)

    # print("homographies: ", homographies)
    warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)

    # shape_cell = torch.tensor([H//cell_size, W//cell_size]).type(torch.FloatTensor).to(device)
    # warped_coor_mask = denormPts(warped_coor_cells, shape_cell)

    return warped_coor_cells

def dataExtractor(sample, val_agent):
    img = sample['image'].type(torch.float32) # event volume
    warped_img = sample['warped_img'].type(torch.float32) # warped event volume
    outs = val_agent.run(img.to("cuda")) # prediction of models
    coarse_desc = outs['desc'] # description
    return img, warped_img, outs, coarse_desc


def uv2descriptorExtractor(image_a_pred, img):
    Hc, Wc = image_a_pred.shape[2], image_a_pred.shape[3]
    img_shape = (Hc, Wc)
    uv_a1 = findInterestPoints(img, blockSize=2, ksize=1, k=0.1)
    uv_a = uv_a1 * np.array([Hc-1, Wc-1])
    uv_a[:,0], uv_a[:,1] = uv_a[:,1].copy(), uv_a[:,0].copy()  # u point to right, v point to down
    uv_a = torch.from_numpy(uv_a).float()
    uv_a = normPts(uv_a, torch.tensor([Wc, Hc]).float()) # [u, v] # [-1,1]
    mode = 'bilinear' 
    matches_a_descriptors = sampleDescriptors(image_a_pred, uv_a.to("cuda"), mode, norm=False)
    return uv_a, matches_a_descriptors

def img2descAndWarpedDesc(image_a_pred, image_b_pred, img, homographies_H):
    Hc, Wc = image_a_pred.shape[2], image_a_pred.shape[3]
    img_shape = (Hc, Wc)
    uv_a1 = findInterestPoints(img, blockSize=2, ksize=1, k=0.1)
    uv_a = uv_a1 * np.array([Hc-1, Wc-1])
    uv_a[:,0], uv_a[:,1] = uv_a[:,1].copy(), uv_a[:,0].copy()  # u point to 
    uv_a = torch.from_numpy(uv_a).float()
    uv_b_matches = warp_coor_cells_with_homographies(uv_a, homographies_H.to('cpu'), uv=True, device='cpu')
    uv_b_matches = uv_b_matches.squeeze(0)
    uv_b_matches, mask = filter_points(uv_b_matches, torch.tensor([Wc, Hc]).to(device='cpu'), return_mask=True)
    uv_a = uv_a[mask] # then filter out uv_a (points before warpped)
    uv_a = normPts(uv_a, torch.tensor([Wc, Hc]).float()) # [u, v] # [-1,1]
    uv_b = normPts(uv_b_matches, torch.tensor([Wc, Hc]).float()) # [u, v] # [-1,1]
    mode = 'bilinear' 
    norm = False
    matches_a_descriptors = sampleDescriptors(image_a_pred, uv_a.to("cuda"), mode, norm=norm)
    matches_b_descriptors = sampleDescriptors(image_b_pred, uv_b.to("cuda"), mode, norm=norm)
    return uv_a, uv_b, matches_a_descriptors, matches_b_descriptors


    


def compute_distances(descriptors_a, descriptors_b):
    distances = np.dot(descriptors_a, descriptors_b.T)
    return distances

def uvListTransformer(uv_a, Wc, Hc):
    input_uv = []
    for i in range(uv_a.squeeze().shape[0]):
        pt = torch.floor(deNormPts(uv_a.squeeze()[i], torch.tensor([Wc*8, Hc*8]).float())).type(torch.int16)
        input_uv.append(cv2.KeyPoint(pt[0].item(), pt[1].item(), 0.1))
    return input_uv

def computeAccuracy(matches):
    accu = 0
    for i in range(len(matches)):
        if matches[i].queryIdx == matches[i].trainIdx:
            accu +=1

    accu /= len(matches)
    return accu

def filter_points(points, shape, return_mask=False):
    ### check!
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape-1)
    mask = (torch.prod(mask, dim=-1) == 1)
    if return_mask:
        return points[mask], mask
    return points [mask]


def warp_coor_cells_with_homographies(coor_cells, homographies, uv=False, device='cpu'):
    from utils.utils import warp_points
    # warped_coor_cells = warp_points(coor_cells.view([-1, 2]), homographies, device)
    # warped_coor_cells = normPts(coor_cells.view([-1, 2]), shape)
    warped_coor_cells = coor_cells
    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:,1], warped_coor_cells[:,0]), dim=1) # (y, x) to (x, y)

    # print("homographies: ", homographies)
    warped_coor_cells = warp_points(warped_coor_cells, homographies, device)

    if uv == False:
        warped_coor_cells = torch.stack((warped_coor_cells[:, :, 1], warped_coor_cells[:, :, 0]), dim=2)  # (batch, x, y) to (batch, y, x)

    # shape_cell = torch.tensor([H//cell_size, W//cell_size]).type(torch.FloatTensor).to(device)
    # warped_coor_mask = denormPts(warped_coor_cells, shape_cell)

    return warped_coor_cells
