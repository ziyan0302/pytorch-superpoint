"""
Visaulization code:
1. Find the corner points in image and warped image
2. Extract description for corner points on image and those on warped image
3. indexize the corner points on image
4. find the most similar point on warped image bi-directionally
5. Visualize the model's result on warped image and next image
(# show 1.raw RGB image w/ pts 2. next raw RGB image w/ pts
    # 3. warped RGB image w/ pts 4. correct pairing in warped data
    # 5. prediction pairing in next data 6. pts movement w/ estimated next homography
    # 7. prediction pairing in warped data 8. pts movement w/ estimated warped homography)
6. calculate the 4-corner error of the image
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
from utils.utils import filter_points_np
from datasets.event_utils import gen_event_images
import visualization_utils as vutil
import time

np.random.seed(42)
torch.manual_seed(42) 


def draw_CrossAndBoundary(out2):
    random_colIdx = np.random.randint(0, 10) //2
    cv2.line(out2, (out2.shape[1]//2,0), (out2.shape[1]//2, out2.shape[0]), (125 + random_colIdx*20, 125, 125 -  random_colIdx*20), thickness=2)
    cv2.line(out2, (0,out2.shape[0]//2), (out2.shape[1],out2.shape[0]//2), (125+ random_colIdx*20,125,125 - random_colIdx*20), thickness=2)
    cv2.line(out2, (0,0), (0, out2.shape[0]), (125 + random_colIdx*20, 125, 125 -  random_colIdx*20), thickness=2)
    cv2.line(out2, (0,0), (out2.shape[1],0), (125 + random_colIdx*20, 125, 125 -  random_colIdx*20), thickness=2)
    cv2.line(out2, (0, out2.shape[0]), (out2.shape[1], out2.shape[0]), (125 + random_colIdx*20, 125, 125 -  random_colIdx*20), thickness=2)
    cv2.line(out2, (out2.shape[1], 0), (out2.shape[1], out2.shape[0]), (125 + random_colIdx*20, 125, 125 -  random_colIdx*20), thickness=2)
    
    

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

pretrained_path = "/home/ziyan/02_research/pytorch-superpoint/logs/superpoint_mvsec/2024-06-20_19-15/checkpoints/superPointNet_8000_checkpoint.pth.tar"
# pretrained_path = "/home/ziyan/02_research/pytorch-superpoint/logs/superpoint_mvsec/2024-06-19_16-55/checkpoints/superPointNet_8000_checkpoint.pth.tar"
config['model']['pretrained'] = pretrained_path

val_agent = Val_model_heatmap(config['model'], device="cuda")
val_agent.loadModel()

def countParms(model):
    
    totalParms = sum(p.numel() for p in model.parameters())
    trainParms = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {totalParms:,}, Trainable params: {trainParms:,}")


countParms(val_agent.net)


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

    # show 1.raw RGB image w/ pts 2. next raw RGB image w/ pts
    # 3. warped RGB image w/ pts 4. correct pairing in warped data
    # 5. prediction pairing in next data 6. pts movement w/ estimated next homography
    # 7. prediction pairing in warped data 8. pts movement w/ estimated warped homography
    shown_imgs = np.zeros((sample['raw_image'].shape[1]*2, sample['raw_image'].shape[2]*4, 3))

    event_img = gen_event_images(sample['image'].type(torch.float32), prefix=None, device='cpu').squeeze()

    img, warped_img, outs, coarse_desc = vutil.dataExtractor(sample, val_agent)
    nextImg, _, nextOuts, nextCoarse_desc = vutil.dataExtractor(nextSample, val_agent)
    image_a_pred = coarse_desc
    nextImage_a_pred = nextCoarse_desc

    Hc, Wc = image_a_pred.shape[2], image_a_pred.shape[3]
    img_shape = (Hc, Wc)

    homographies_H = sample["homographies"]
    homographies_H = vutil.scale_homography_torch(homographies_H, img_shape, shift=(-1, -1))

    # get result on warped img
    warped_desc = val_agent.run(warped_img.to("cuda"))['desc']
    warpedImage_a_pred = warped_desc
    

    # get the normed interest points and description
    uv_a, matches_a_descriptors = vutil.uv2descriptorExtractor(image_a_pred, img)

    uv_aNext, matches_aNext_descriptors = vutil.uv2descriptorExtractor(nextImage_a_pred, nextImg)
    uv_aWarped, matches_aWarped_descriptors = vutil.uv2descriptorExtractor(warpedImage_a_pred, warped_img)

    # correct answer
    uv_a_befW, uv_a_aftW, matches_aBefW_descriptors, matches_aAftW_descriptors = \
        vutil.img2descAndWarpedDesc(image_a_pred, warpedImage_a_pred, img, homographies_H)

    if (1): # randomly filter out some interest points
        sample_rate = 21
        idxs_a = np.arange(0,uv_a.shape[0],sample_rate)
        uv_a = uv_a[idxs_a]
        matches_a_descriptors = matches_a_descriptors[idxs_a]

        idxs_next = np.arange(0,uv_aNext.shape[0],sample_rate)
        uv_aNext = uv_aNext[idxs_next]
        matches_aNext_descriptors = matches_aNext_descriptors[idxs_next]

        idxs_pred_w = np.arange(0,uv_aWarped.shape[0],sample_rate)
        uv_aWarped = uv_aWarped[idxs_pred_w]
        matches_aWarped_descriptors = matches_aWarped_descriptors[idxs_pred_w]

        idxs_warp = np.arange(0,uv_a_befW.shape[0],sample_rate)
        uv_a_befW = uv_a_befW[idxs_warp]
        uv_a_aftW = uv_a_aftW[idxs_warp]
        matches_aBefW_descriptors = matches_aBefW_descriptors[idxs_warp]
        matches_aAftW_descriptors = matches_aAftW_descriptors[idxs_warp]




    


    print("matches_a_descriptors: ", matches_a_descriptors.shape)
    print("matches_aNext_descriptors: ", matches_aNext_descriptors.shape)

    # Compute distances between descriptors
    matches = []
    if torch.any(torch.isnan(matches_a_descriptors)) or torch.any(torch.isnan(matches_aNext_descriptors)):
        print("nan happeningggggggggggggggggggggg")
    else:
        vutil.compute_matches(matches_a_descriptors, matches_aNext_descriptors, matches)

    # Compute distances between descriptors from before_warp and after_warp
    matchesWarp = []
    if torch.any(torch.isnan(matches_a_descriptors)) or torch.any(torch.isnan(matches_aWarped_descriptors)):
        print("nan happeningggggggggggggggggggggg")
    else:
        vutil.compute_matches(matches_a_descriptors, matches_aWarped_descriptors, matchesWarp)
        



    input_uv = vutil.uvListTransformer(uv_a, Wc, Hc) # pts on current events 
    pred_uv = vutil.uvListTransformer(uv_aNext, Wc, Hc) # pts on next events
    predWarp_uv = vutil.uvListTransformer(uv_aWarped, Wc, Hc) # pts on warped events
    input_uv_befW = vutil.uvListTransformer(uv_a_befW, Wc, Hc) # pts on current events (filtered by warped operation)
    correct_uv_afW = vutil.uvListTransformer(uv_a_aftW, Wc, Hc) # pts on warped events
    

    img1 = sample['raw_image'].squeeze().cpu().numpy()
    img2 = nextSample['raw_image'].squeeze().cpu().numpy()
    imgWarped = sample['warped_rawImg'].squeeze().cpu().numpy()

    out1 = cv2.drawKeypoints(img1, input_uv, None)
    draw_CrossAndBoundary(out1)
    out2 = cv2.drawKeypoints(img2, pred_uv, None)
    draw_CrossAndBoundary(out2)
    out2W = cv2.drawKeypoints(imgWarped.astype(np.uint8), predWarp_uv, None)
    draw_CrossAndBoundary(out2W)
    shown_imgs[:out1.shape[0],:out1.shape[1],:] = out1
    # shown_imgs[:out1.shape[0], event_img.shape[1]:event_img.shape[1]*2,:] = event_img
    shown_imgs[:out1.shape[0], out1.shape[1]:out2.shape[1]*2,:] = out2
    shown_imgs[:out1.shape[0], out2.shape[1]*2:out2.shape[1]*3,:] = out2W

    points1 = np.float32([uv.pt for uv in input_uv_befW])
    points2 = np.float32([uv.pt for uv in correct_uv_afW])
    vutil.draw_corresponding2(out2W, points1, points2, color=(255,0,255))
    shown_imgs[:out1.shape[0], out2.shape[1]*3:,:] = out2W # correct paring in warped img


    ## draw lines for matched points on the same image for next prediction
    out2 = vutil.draw_corresponding(img2, input_uv, pred_uv, matches)
    draw_CrossAndBoundary(out2)
    shown_imgs[out1.shape[0]:, :out2.shape[1]] = out2

    ## estimate homography between current and next img
    H, befH_pts, homo_pts = vutil.estimate_HandPts(input_uv, pred_uv, matches, Wc, Hc)
    keypointsBefH = [cv2.KeyPoint(pt[0].item(), pt[1].item(), 0.1) for pt in befH_pts]
    keypointsAftH = [cv2.KeyPoint(pt[0].item(), pt[1].item(), 0.1) for pt in homo_pts]
    out1 = cv2.drawKeypoints(img2, keypointsBefH, None)
    out2 = cv2.drawKeypoints(out1, keypointsAftH, None)
    # draw pairing with estimated H
    vutil.draw_corresponding2(out2, befH_pts, homo_pts)
    shown_imgs[out2.shape[0]:, out2.shape[1]:out2.shape[1]*2,:] = out2




    ## draw lines for matched points on the same image for warped pair
    out2 = vutil.draw_corresponding(imgWarped.astype(np.uint8), input_uv, predWarp_uv, matchesWarp, color=(255, 100, 255))
    
    shown_imgs[out1.shape[0]:,out2.shape[1]*2:out2.shape[1]*3, :] = out2

    ## estimate homography between current and warped img
    H, befH_pts, homo_pts = vutil.estimate_HandPts(input_uv, predWarp_uv, matchesWarp, Wc, Hc)
    keypointsBefH = [cv2.KeyPoint(pt[0].item(), pt[1].item(), 0.1) for pt in befH_pts]
    keypointsAftH = [cv2.KeyPoint(pt[0].item(), pt[1].item(), 0.1) for pt in homo_pts]
    out1 = cv2.drawKeypoints(img2, keypointsBefH, None)
    out2 = cv2.drawKeypoints(out1, keypointsAftH, None)
    # draw pairing with estimated H
    vutil.draw_corresponding2(out2, befH_pts, homo_pts, color=(255, 100, 255))
    shown_imgs[out2.shape[0]:, out2.shape[1]*3:,:] = out2

    ##  calculate the 4-corner error
    cornersOfImg = np.array(
        [[0,0],
         [0, out2.shape[0]],
         [out2.shape[1], out2.shape[0]],
         [out2.shape[1],0]], dtype=np.float64)
    cornersOfImg /= 8
    H
    homographies_H
    points = np.concatenate([cornersOfImg, np.ones(len(cornersOfImg))[:,np.newaxis]], axis=1)
    def get_pts_afterHomo(points, H, original_maxs):
        oriMaxX, oriMaxY = original_maxs[0], original_maxs[1]
        homo_pts = np.dot(points, H.T)
        homo_pts = homo_pts/homo_pts[:,2][:, np.newaxis]
        homo_pts = homo_pts[:,:2]
        minX = min(homo_pts[:,0])
        minX = min(minX, 0)
        minY = min(homo_pts[:,1])
        minY = min(minY, 0)
        maxX = max(homo_pts[:,0])
        maxX = max(maxX, oriMaxX)
        maxY = max(homo_pts[:,1])
        maxY = max(maxY, oriMaxY)
        return homo_pts, minX, maxX, minY, maxY
    homo_pts, minX, maxX, minY, maxY = get_pts_afterHomo(points, H, (out2.shape[0]/8, out2.shape[1]/8))  
    # corr_pts, minX_c, maxX_c, minY_c, maxY_c = get_pts_afterHomo(points, homographies_H, (out2.shape[0]/8, out2.shape[1]/8))  
    tmp_pts = np.copy(points)
    tmp_pts[:,1], tmp_pts[:,0] = points[:,0], points[:,1]
    corr_pts = vutil.warp_coor_cells_with_homographies(torch.from_numpy(points)[:,:2], homographies_H.to('cpu'), uv=True, device='cpu').squeeze().numpy()
    
    minX_c, maxX_c, minY_c, maxY_c = min(corr_pts[:,0]), max(corr_pts[:,0]), min(corr_pts[:,1]), max(corr_pts[:,1])


    

    minX, maxX, minY, maxY = min(minX, minX_c), max(maxX, maxX_c), min(minY, minY_c), max(maxY, maxY_c)

    homo_pts[:,0] -= minX
    homo_pts[:,1] -= minY
    corr_pts[:,0] -= minX
    corr_pts[:,1] -= minY

        

    Width = int(maxX - minX) + 1
    Height = int(maxY - minY) + 1
    comparison_img = np.zeros((Height*8, Width*8,3))
    original_corners = np.copy(cornersOfImg)
    original_corners[:,0] -= minX
    original_corners[:,1] -= minY


    original_corners *= 8
    homo_pts *= 8
    corr_pts *= 8

    comparison_img.shape
    for ptIdx1, ptIdx2 in zip([0,1,2,3], [1,2,3,0]):
        pt1 = tuple(map(int, original_corners[ptIdx1]))
        pt2 = tuple(map(int, original_corners[ptIdx2]))
        cv2.line(comparison_img, pt1, pt2, (255, 0, 0), thickness=2)

    comparison_img.shape
    imgStartX, imgStartY = np.int16(original_corners[0][0]), np.int16(original_corners[0][1])

    out2W.shape
    comparison_img[imgStartY:(imgStartY+out2W.shape[0]), imgStartX:(imgStartX+out2W.shape[1])] = out2W
        
    for ptIdx1, ptIdx2 in zip([0,1,2,3], [1,2,3,0]):
        pt1 = tuple(map(int, homo_pts[ptIdx1]))
        pt2 = tuple(map(int, homo_pts[ptIdx2]))
        cv2.line(comparison_img, pt1, pt2, (255, 255, 255), thickness=2)
    
    for ptIdx1, ptIdx2 in zip([0,1,2,3], [1,2,3,0]):
        pt1 = tuple(map(int, corr_pts[ptIdx1]))
        pt2 = tuple(map(int, corr_pts[ptIdx2]))
        cv2.line(comparison_img, pt1, pt2, (0, 255, 255), thickness=2)


    cv2.imwrite("out1.jpg", comparison_img)



    
    







    

    cv2.imwrite("out2.jpg", shown_imgs)
    # pdb.set_trace()
        
            


print("end")