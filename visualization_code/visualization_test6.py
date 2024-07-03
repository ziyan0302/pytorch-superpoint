"""
Visaulization code:
1. Find the corner points in image and warped image
2. Extract description for corner points on image and those on warped image
3. indexize the corner points on image
4. find the most similar point on warped image
5. Visualize the model's result on warped image and next image
6. use for multiple models
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

def countParms(model):
    
    totalParms = sum(p.numel() for p in model.parameters())
    trainParms = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total params: {totalParms:,}, Trainable params: {trainParms:,}")

pdb.set_trace()
val_agent.net
tmp = next(iter(train_loader))
tmp['image'].shape

countParms(val_agent.net)


from tqdm import tqdm
# data_iter = iter(val_loader)

# Define the number of frames to skip
skip_steps = 50

data_iter = iter(val_loader)


pretrained_list = ["/home/ziyan/02_research/pytorch-superpoint/logs/superpoint_mvsec/2024-06-17_11-28/checkpoints/superPointNet_8000_checkpoint.pth.tar","/home/ziyan/02_research/pytorch-superpoint/logs/superpoint_mvsec/2024-06-19_17-53/checkpoints/superPointNet_8000_checkpoint.pth.tar","/home/ziyan/02_research/pytorch-superpoint/logs/superpoint_mvsec/2024-06-19_16-55/checkpoints/superPointNet_8000_checkpoint.pth.tar","/home/ziyan/02_research/pytorch-superpoint/logs/superpoint_mvsec/2024-06-20_19-15/checkpoints/superPointNet_8000_checkpoint.pth.tar"]
# for index, (sample, nextSample) in tqdm(enumerate(zip(data_iter, data_iter))):
for index, sample in tqdm(enumerate(data_iter)):
    for _ in range(skip_steps-1):
        next(data_iter)
    nextSample = next(data_iter)

    shown_correspondings = np.zeros((sample['raw_image'].shape[1]*1, sample['raw_image'].shape[2]*len(pretrained_list), 3))
    shown_correspondingsWarped = np.zeros((sample['raw_image'].shape[1]*1, sample['raw_image'].shape[2]*len(pretrained_list), 3))
    

    if (0): ## show event volume visualization
        event_img = gen_event_images(sample['image'].type(torch.float32), prefix=None, device='cpu').squeeze()
        cv2.imwrite("event.jpg", (event_img*255).numpy())
        time.sleep(0.1)

    for iPath, pre_path in enumerate(pretrained_list):
        print("iPath:   ", iPath)
        config['model']['pretrained'] = pre_path
        val_agent = Val_model_heatmap(config['model'], device="cuda")
        val_agent.loadModel()

    

        

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
            distances_matrix = vutil.compute_distances(matches_a_descriptors.cpu().numpy(), matches_aNext_descriptors.cpu().numpy())
            # check bidirectionally and find the most similar pairs
            column_index_with_smallest_value, row_index_with_smallest_value = vutil.find_most_similar_pairs(distances_matrix)
            
            bidirectional_matches = vutil.find_bidirectional_matches(column_index_with_smallest_value, row_index_with_smallest_value)
            for row_index, col_index in bidirectional_matches:
                # Create a cv2.DMatch object with query index as row_index, train index as col_index,
                # and optionally set distance to 0
                dmatch = cv2.DMatch(row_index, col_index, 0)
                # Append the created dmatch to the list
                matches.append(dmatch)

        # Compute distances between descriptors from before_warp and after_warp
        matchesWarp = []
        if torch.any(torch.isnan(matches_a_descriptors)) or torch.any(torch.isnan(matches_aWarped_descriptors)):
            print("nan happeningggggggggggggggggggggg")
        else:
            distances_matrix_Warp = vutil.compute_distances(matches_a_descriptors.cpu().numpy(), matches_aWarped_descriptors.cpu().numpy())
            # check bidirectionally and find the most similar pairs
            column_index_with_smallest_value_Warp, row_index_with_smallest_value_Warp = vutil.find_most_similar_pairs(distances_matrix_Warp)

            bidirectional_matches_Warp = vutil.find_bidirectional_matches(column_index_with_smallest_value_Warp, row_index_with_smallest_value_Warp)
            for row_index, col_index in bidirectional_matches_Warp:
                # Create a cv2.DMatch object with query index as row_index, train index as col_index,
                # and optionally set distance to 0
                dmatch = cv2.DMatch(row_index, col_index, 0)
                # Append the created dmatch to the list
                matchesWarp.append(dmatch)



        input_uv = vutil.uvListTransformer(uv_a, Wc, Hc)
        pred_uv = vutil.uvListTransformer(uv_aNext, Wc, Hc)
        predWarp_uv = vutil.uvListTransformer(uv_aWarped, Wc, Hc)
        input_uv_befW = vutil.uvListTransformer(uv_a_befW, Wc, Hc)
        correct_uv_afW = vutil.uvListTransformer(uv_a_aftW, Wc, Hc)
        

        img1 = sample['raw_image'].squeeze().cpu().numpy()
        img2 = nextSample['raw_image'].squeeze().cpu().numpy()
        imgWarped = sample['warped_rawImg'].squeeze().cpu().numpy()
    


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
                # cv2.line(out2, pt1, pt2, (0, 255, 0), 2)  # Draw a line from pt1 to pt2
                line_length = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
                # tipLength = arrow_tip_length/line_length
                cv2.arrowedLine(out2, pt1, pt2, (255, 255, 0), thickness=2, tipLength=0.05)
            # pdb.set_trace()
        random_colIdx = np.random.randint(0, 10) //2
        cv2.line(out2, (out2.shape[1]//2,0), (out2.shape[1]//2, out2.shape[0]), (125 + random_colIdx*20, 125, 125 -  random_colIdx*20), thickness=2)
        cv2.line(out2, (0,out2.shape[0]//2), (out2.shape[1],out2.shape[0]//2), (125+ random_colIdx*20,125,125 - random_colIdx*20), thickness=2)
        
        shown_correspondings[:,iPath*img1.shape[1]:(iPath+1)*img1.shape[1], :] = out2


        if (1): ## draw lines for matched points on the same image for warped pair
            out1 = cv2.drawKeypoints(img1, input_uv, None)
            out2 = cv2.drawKeypoints(out1, predWarp_uv, None)
            points1 = np.float32([input_uv[match.queryIdx].pt for match in matchesWarp])
            points2 = np.float32([predWarp_uv[match.trainIdx].pt for match in matchesWarp])
            for pt1, pt2 in zip(points1, points2): 
                pt1 = tuple(map(int, pt1))  # Convert to integer coordinates
                pt2 = tuple(map(int, pt2))  # Convert to integer coordinates
                cv2.line(out2, pt1, pt2, (0, 255, 0), 2)  # Draw a line from pt1 to pt2
                cv2.arrowedLine(out2, pt1, pt2, (0, 255, 0), thickness=2, tipLength=0.05)

            # pdb.set_trace()
        random_colIdx = np.random.randint(0, 10) //2
        cv2.line(out2, (out2.shape[1]//2,0), (out2.shape[1]//2, out2.shape[0]), (125 + random_colIdx*20, 125, 125 -  random_colIdx*20), thickness=2)
        cv2.line(out2, (0,out2.shape[0]//2), (out2.shape[1],out2.shape[0]//2), (125+ random_colIdx*20,125,125 - random_colIdx*20), thickness=2)
        
        shown_correspondingsWarped[:,iPath*img1.shape[1]:(iPath+1)*img1.shape[1], :] = out2

    original_img_pair = np.zeros_like(shown_correspondings)
    pts2img1 = cv2.drawKeypoints(img1, input_uv, None)
    cv2.line(pts2img1, (pts2img1.shape[1]//2,0), (pts2img1.shape[1]//2, pts2img1.shape[0]), (255, 255, 255), thickness=2)
    cv2.line(pts2img1, (0,pts2img1.shape[0]//2), (pts2img1.shape[1],pts2img1.shape[0]//2), (255, 255, 255), thickness=2)
    pts2img2 = cv2.drawKeypoints(img2, pred_uv, None)
    cv2.line(pts2img2, (pts2img1.shape[1]//2,0), (pts2img1.shape[1]//2, pts2img1.shape[0]), (255, 0, 255), thickness=2)
    cv2.line(pts2img2, (0,pts2img1.shape[0]//2), (pts2img1.shape[1],pts2img1.shape[0]//2), (255, 0, 255), thickness=2)
    
    original_img_pair[:, :img1.shape[1], :] = pts2img1
    original_img_pair[:, img1.shape[1]: (img1.shape[1]+img2.shape[1]), :] = pts2img2

    # draw warp pair
    pts2imgbef = cv2.drawKeypoints(img1, input_uv_befW, None)
    cv2.line(pts2imgbef, (pts2imgbef.shape[1]//2,0), (pts2imgbef.shape[1]//2, pts2imgbef.shape[0]), (255, 255, 255), thickness=2)
    cv2.line(pts2imgbef, (0,pts2imgbef.shape[0]//2), (pts2imgbef.shape[1],pts2imgbef.shape[0]//2), (255, 255, 255), thickness=2)
    pts2imgAft = cv2.drawKeypoints(imgWarped.astype(np.uint8), correct_uv_afW, None)
    
    if len(pretrained_list) >2:
        imgStart = img1.shape[1]+img2.shape[1]
    else:
        imgStart = img1.shape[1]
    original_img_pair.shape
    original_img_pair[:, imgStart:(imgStart+img1.shape[1]), :].shape
    original_img_pair[:, imgStart:(imgStart+img1.shape[1]), :] = pts2imgAft

    points1 = np.float32([input_uv_befW[i].pt for i in range(len(input_uv_befW))])
    points2 = np.float32([correct_uv_afW[i].pt for i in range(len(correct_uv_afW))])
    for pt1, pt2 in zip(points1, points2): 
        pt1 = tuple(map(int, pt1))  # Convert to integer coordinates
        pt2 = tuple(map(int, pt2))  # Convert to integer coordinates
        # cv2.line(out2, pt1, pt2, (0, 255, 0), 2)  # Draw a line from pt1 to pt2
        line_length = int(np.linalg.norm(np.array(pt1) - np.array(pt2)))
        # tipLength = arrow_tip_length/line_length
        cv2.arrowedLine(pts2imgAft, pt1, pt2, (255, 255, 0), thickness=2, tipLength=0.05)
    cv2.line(pts2imgAft, (pts2imgAft.shape[1]//2,0), (pts2imgAft.shape[1]//2, pts2imgAft.shape[0]), (255, 0, 255), thickness=2)
    cv2.line(pts2imgAft, (0,pts2imgAft.shape[0]//2), (pts2imgAft.shape[1],pts2imgAft.shape[0]//2), (255, 0, 255), thickness=2)
    imgStart += img1.shape[1]
    original_img_pair.shape
    original_img_pair[:, imgStart:(imgStart+imgWarped.shape[1]), :] = pts2imgAft



    
    result = np.concatenate((shown_correspondings, shown_correspondingsWarped), axis=0)
    result = np.concatenate((result, original_img_pair), axis=0)
    cv2.imwrite("out2.jpg", result)
        
            


print("end")