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

filename = "/home/ziyan/02_research/pytorch-superpoint/configs/superpoint_mvsec_train_heatmap.yaml"

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
def descriptor_reshape(descriptors):
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


# for iImg in range(5):
for sample in val_loader:
    # sample = next(iter(train_loader))
    # sample_val = next(iter(val_loader))

    if (0):
        sample_val.keys()
        img1 = sample['raw_image'][1].squeeze().cpu().numpy()
        
        img1 = sample_val['raw_image'][0].squeeze().cpu().numpy()
        img1.shape
        sample_val = next(iter(val_loader))
        cv2.imwrite("img2.jpg", sample_val['raw_image'].squeeze().cpu().numpy())
        
        

    homographies_H = sample["homographies"]

    img = sample['image'].type(torch.float32) # event volume
    warped_img = sample['warped_img'].type(torch.float32) # warped event volume

    outs = val_agent.run(img.to("cuda"))
    coarse_desc = outs['desc']
    warped_outs = val_agent.run(warped_img.to("cuda"))
    coarse_desc_warp = warped_outs['desc']

    ####TODO : show the comparison of description of img and img_warp
    image_a_pred = coarse_desc
    image_b_pred = coarse_desc_warp
    Hc, Wc = image_a_pred.shape[2], image_a_pred.shape[3]
    img_shape = (Hc, Wc)

    uv_a1 = findInterestPoints(img, blockSize=2, ksize=1, k=0.1)
    # uv_b = findInterestPoints(warped_img, blockSize=2, ksize=1, k=0.1)
    uv_a = uv_a1 * np.array([Hc-1, Wc-1])
    # uv_b = uv_b * np.array([Hc-1, Wc-1])
    uv_a[:,0], uv_a[:,1] = uv_a[:,1].copy(), uv_a[:,0].copy()  # u point to right, v point to down
    # uv_b[:,0], uv_b[:,1] = uv_b[:,1].copy(), uv_b[:,0].copy()  

    uv_a = torch.from_numpy(uv_a).float()
    homographies_H = scale_homography_torch(homographies_H, img_shape, shift=(-1, -1))
    
    uv_b_matches = warp_coor_cells_with_homographies(uv_a, homographies_H.to('cpu'), uv=True, device='cpu')
    # pdb.set_trace()
    # uv_b_matches.round_() 
    uv_b_matches = uv_b_matches.squeeze(0)
    # filtering out of range points
    uv_b_matches, mask = filter_points(uv_b_matches, torch.tensor([Wc, Hc]).to(device='cpu'), return_mask=True)
    uv_a = uv_a[mask] # then filter out uv_a (points before warpped)
    uv_at = uv_a.clone()

    
    uv_a = normPts(uv_a, torch.tensor([Wc, Hc]).float()) # [u, v] # [-1,1]
    # uv_b = torch.from_numpy(uv_b).float()
    # uv_b = normPts(uv_b, torch.tensor([Wc, Hc]).float()) # [u, v] # [-1,1]
    uv_b = normPts(uv_b_matches, torch.tensor([Wc, Hc]).float()) # [u, v] # [-1,1]

    mode = 'bilinear' 

    norm = False
    matches_a_descriptors = sampleDescriptors(image_a_pred, uv_a.to("cuda"), mode, norm=norm)
    matches_b_descriptors = sampleDescriptors(image_b_pred, uv_b.to("cuda"), mode, norm=norm)


    # matches = bfMatching(matches_a_descriptors, matches_b_descriptors)

    print("matches_a_descriptors: ", matches_a_descriptors.shape)
    print("matches_b_descriptors: ", matches_b_descriptors.shape)
    # pdb.set_trace()

    def compute_distances(descriptors_a, descriptors_b):
        num_a = len(descriptors_a)
        num_b = len(descriptors_b)
        distances = np.zeros((num_a, num_b))  # Initialize matrix to store distances

        for i in range(num_a):
            for j in range(num_b):
                distances[i, j] = np.linalg.norm(descriptors_a[i] - descriptors_b[j])

        return distances
    # Compute distances between descriptors
    distances_matrix = compute_distances(matches_a_descriptors.cpu().numpy(), matches_b_descriptors.cpu().numpy())
    column_index_with_smallest_value = np.argmin(distances_matrix, axis=1)
    matches = []
    for row_index, col_index in enumerate(column_index_with_smallest_value):
        # Create a cv2.DMatch object with query index as row_index, train index as col_index,
        # and optionally set distance to 0
        dmatch = cv2.DMatch(row_index, col_index, 0)
        # Append the created dmatch to the list
        matches.append(dmatch)


    input_uv = []
    for i in range(uv_a.squeeze().shape[0]):
        pt = torch.floor(deNormPts(uv_a.squeeze()[i], torch.tensor([Wc*8, Hc*8]).float())).type(torch.int16)
        input_uv.append(cv2.KeyPoint(pt[0].item(), pt[1].item(), 0.1))
    pred_uv = []
    for i in range(uv_b.squeeze().shape[0]):
        pt = torch.floor(deNormPts(uv_b.squeeze()[i], torch.tensor([Wc*8, Hc*8]).float())).type(torch.int16)
        pred_uv.append(cv2.KeyPoint(pt[0].item(), pt[1].item(), 0.1))

    pdb.set_trace()
    if (0):
        img.shape
        warped_img.dtype
        uv_a.shape
        uv_a[mask].shape
        uv_b.shape
        uv_b_matches.shape
        matches_a_descriptors.shape
        matches_b_descriptors.shape
        distances_matrix.shape
        len(matches)
        distances_matrix[0][:100]
        len(input_uv)
        input_uv[0].pt
        len(pred_uv)
        testpt = torch.tensor([uv_a[0][0], uv_a[0][1], 1.0])
        deNormPts(uv_a.squeeze()[0], torch.tensor([Wc*8, Hc*8]))
        projpt = homographies_H.type(torch.float32) @ testpt.type(torch.float32).squeeze()
        projpt.shape
        projpt[:,0], projpt[:,1] = projpt[:,0]/projpt[:,2], projpt[:,1]/projpt[:,2]

        uv_at = torch.from_numpy(uv_a).float()
        homographies_H.shape
        torch.inverse(homographies_H)

        uv_b_matches = warp_coor_cells_with_homographies(uv_a, torch.inverse(homographies_H[0]).T.to('cpu'), uv=True, device='cpu')
        uv_b_matches = warp_coor_cells_with_homographies(uv_a, homographies_H[0].to('cpu'), uv=True, device='cpu')
        # uv_b_matches.round_() 
        # uv_b_matches = uv_b_matches.squeeze(0)
    
        uv_b_matches, mask = filter_points(uv_b_matches, torch.tensor([Wc, Hc]).to(device='cpu'), return_mask=True)
        uv_a = uv_a[mask] # then filter out uv_a (points before warpped)

        uv_b_matches.shape

    
        homoB = [uv_b_matches[matches[i].queryIdx] for i in range(len(matches))]
        len(homoB)
        homoB1 = [cv2.KeyPoint(y,x,0.1) for x,y in homoB]

        img1 = sample['raw_image'].squeeze().cpu().numpy()
        img2 = np.floor(sample['warped_rawImg'].squeeze().cpu().numpy()).astype(np.uint8)
        tmp_uv = findInterestPoints(torch.from_numpy(img2).float(), blockSize=2, ksize=1, k=0.1)
        sample['raw_image'].shape
        img2.shape

        uv_at = torch.floor(uv_a*8).type(torch.int16)
        uv_at = torch.floor(uv_at*8).type(torch.int16)
        # uv_at = uv_a[mask]*8
        uv_at[0][0]
        input_uv1 = [cv2.KeyPoint(int(u),int(v),0.1) for u,v in uv_at]
        input_uv1[0].pt
        input_uv[5].pt
        pred_uv[5].pt

        uv_b_test = torch.floor(uv_b_matches*8).type(torch.int16)
        homo_uv = [cv2.KeyPoint(int(u),int(v),0.1) for u,v in uv_b_test]



        out1 = cv2.drawKeypoints(img1, input_uv, None)
        out2 = cv2.drawKeypoints(img2, pred_uv, None)
        out2 = cv2.drawKeypoints(img2, homo_uv, None)
        out2 = cv2.drawKeypoints(img2, input_uv, None)
        out2 = cv2.drawKeypoints(out2, homoB1, None, color=(255,0,0))
        type(out1)
        img3 = cv2.drawMatches(out1,  input_uv,  out2,  pred_uv,  matches,  None,  matchColor=(0, 255, 0),  singlePointColor=(255, 0, 0),  matchesMask=None)
    
        cv2.imwrite("out1.jpg", out1)
        cv2.imwrite("out2.jpg", out2)
        cv2.imwrite("img3.jpg", img3)




    # uv_b_matches = torch.floor(uv_b_matches*8).type(torch.int16)
    # homo_uv = [cv2.KeyPoint(int(u),int(v),0.1) for u,v in uv_b_matches]

    ## compute matched accuracy

    if (0):
        accu = 0
        for i in range(len(matches)):
            if matches[i][0] == matches[i][1]:
                accu +=1

        accu /= len(matches)
        print("accuracy: ", accu)
    img1 = sample['raw_image'].squeeze().cpu().numpy()
    img2 = np.floor(sample['warped_rawImg'].squeeze().cpu().numpy()).astype(np.uint8)
    if (1):
        # ## draw matched picture
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
    if (0):
        out1 = cv2.drawKeypoints(img2, input_uv, None)
        out2 = cv2.drawKeypoints(img2, pred_uv, None)
        
        
    pdb.set_trace()
    # plt.title(f"dist[0]: {disMostSim[0]}, dist[1]:{disMostSim[1]}, accu: {accu}")
    # plt.savefig(f"result{iImg}.jpg")
    # print(f"save result{iImg}")



print("end")