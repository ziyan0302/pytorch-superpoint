import pdb
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from tqdm import tqdm
from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update

from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened

from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
from utils.utils import save_checkpoint

from pathlib import Path
from models.model_wrap import SuperPointFrontend_torch

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

if (0):
    config_loc = "/home/ziyan/02_research/pytorch-superpoint/configs/superpoint_mvsec_train_heatmap.yaml"
    with open(config_loc, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    # load dataset
    task = config['data']['dataset']
    data = dataLoader(config, dataset=task, warp_input=True)
    train_loader, val_loader = data['train_loader'], data['val_loader']

    sample = next(iter(train_loader))
    raw_image = sample['raw_image'].transpose(0,1).transpose(1,2)
filename = "/home/ziyan/02_research/pytorch-superpoint/configs/superpoint_mvsec_train_heatmap.yaml"
import yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_tensor_type(torch.FloatTensor)
with open(filename, "r") as f:
    config = yaml.safe_load(f)

from utils.loader import dataLoader as dataLoader

# data = dataLoader(config, dataset='hpatches')
task = config["data"]["dataset"]

data = dataLoader(config, dataset=task, warp_input=True)
# test_set, test_loader = data['test_set'], data['test_loader']
train_loader, val_loader = data["train_loader"], data["val_loader"]

val_agent = Val_model_heatmap(config['model'], device="cuda")
val_agent.loadModel()
sample = next(iter(train_loader))
img = sample['image']
warped_img = sample['warped_img']
outs = val_agent.run(img.to("cuda"))
coarse_desc = outs['desc']
warped_outs = val_agent.run(warped_img.to("cuda"))
coarse_desc_warp = warped_outs['desc']


####TODO : show the comparison of description of img and img_warp
image_a_pred = coarse_desc
image_b_pred = coarse_desc_warp
import torch.nn.functional as F
import cv2
from utils.utils import filter_points
from utils.utils import crop_or_pad_choice
from utils.utils import normPts
Hc, Wc = image_a_pred.shape[2], image_a_pred.shape[3]
img_shape = (Hc, Wc)
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
    threshold = 0.01 * dst.max()
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
# image_a_pred = descriptor_reshape(image_a_pred)  # torch [1, H*W, D]
# image_b_pred = descriptor_reshape(image_b_pred)  # torch [batch_size, H*W, D]

uv_a = findInterestPoints(img, blockSize=2, ksize=1, k=0.1)
uv_b = findInterestPoints(warped_img, blockSize=2, ksize=1, k=0.1)

uv_a = uv_a * np.array([Hc-1, Wc-1])
uv_b = uv_b * np.array([Hc-1, Wc-1])
uv_a[:,0], uv_a[:,1] = uv_a[:,1].copy(), uv_a[:,0].copy()  
uv_b[:,0], uv_b[:,1] = uv_b[:,1].copy(), uv_b[:,0].copy()  

uv_a = torch.from_numpy(uv_a).float()
uv_a = normPts(uv_a, torch.tensor([Wc, Hc]).float()) # [u, v] # [-1,1]
uv_b = torch.from_numpy(uv_b).float()
uv_b = normPts(uv_b, torch.tensor([Wc, Hc]).float()) # [u, v] # [-1,1]

uv_a = uv_a.to("cuda")
uv_b = uv_b.to("cuda")


mode = 'bilinear' 
def sampleDescriptors(image_a_pred, matches_a, mode, norm=False):
    # image_a_pred = image_a_pred.unsqueeze(0) # torch [1, D, H, W]
    # image_a_pred = image_a_pred.squeeze(0)
    matches_a.unsqueeze_(0).unsqueeze_(1)
    matches_a_descriptors = F.grid_sample(image_a_pred, matches_a, mode=mode, align_corners=True)
    matches_a_descriptors = matches_a_descriptors.squeeze().transpose(0,1)
    
    # print("image_a_pred: ", image_a_pred.shape)
    # print("matches_a: ", matches_a.shape)
    # print("matches_a: ", matches_a)
    # print("matches_a_descriptors: ", matches_a_descriptors)
    if norm:
        dn = torch.norm(matches_a_descriptors, p=2, dim=1) # Compute the norm of b_descriptors
        matches_a_descriptors = matches_a_descriptors.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    return matches_a_descriptors

# image_b_pred = image_b_pred.unsqueeze(0) # torch [1, D, H, W]
# matches_b.unsqueeze_(0).unsqueeze_(2)
# matches_b_descriptors = F.grid_sample(image_b_pred, matches_b, mode=mode)
# matches_a_descriptors = matches_a_descriptors.squeeze().transpose(0,1)
norm = False
uv_a.shape
matches_a_descriptors = sampleDescriptors(image_a_pred, uv_a, mode, norm=norm)
matches_b_descriptors = sampleDescriptors(image_b_pred, uv_b, mode, norm=norm)

def find_most_similar_index(descriptor_a, descriptors_b):
    # Calculate cosine similarity between descriptor_a and all descriptors in descriptors_b
    similarities = F.cosine_similarity(descriptor_a.unsqueeze(0), descriptors_b, dim=1)
    
    # Find the index of the most similar descriptor
    most_similar_index = torch.argmax(similarities)
    
    return most_similar_index

matches_a_descriptors.shape
matches_b_descriptors.shape
uva = []
pred_uvb = []

if len(matches_a_descriptors) >10:
    for _ in range(10):
        index_a = int(torch.floor(torch.rand(1) * len(matches_a_descriptors)))
        descriptor_a = matches_a_descriptors[index_a]
        uva.append(torch.floor(deNormPts(uv_a.squeeze()[index_a].to('cpu'), torch.tensor([Wc*8, Hc*8]).float())))
        # Find the index in matches_b_descriptors that is the most similar to descriptor_a
        most_similar_index_b = find_most_similar_index(descriptor_a, matches_b_descriptors)
        pred_uvb.append(torch.floor(deNormPts(uv_b.squeeze()[most_similar_index_b].to('cpu'), torch.tensor([Wc*8, Hc*8]).float())))

input_uv = [uv for uv in zip(*uva)]
pred_uv = [uv for uv in zip(*pred_uvb)]

pdb.set_trace()
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2)
axes[0,0].imshow(sample['raw_image'].squeeze(), cmap='gray')
axes[0,0].scatter(input_uv[0], input_uv[1])
for i in range(len(uva)):    axes[0,0].annotate(i, (input_uv[0][i], input_uv[1][i]), color = 'r')
print("finish 1")
axes[1,0].imshow(sample['warped_rawImg'].squeeze(), cmap='gray')
axes[1,0].scatter(pred_uv[0], pred_uv[1])
for i in range(len(pred_uvb)): axes[1,0].annotate(i, (pred_uv[0][i], pred_uv[1][i]), color = 'r')
axes[0,1].imshow(sample['raw_image'].squeeze(), cmap='gray')
axes[1,1].imshow(sample['warped_rawImg'].squeeze(), cmap='gray')

plt.savefig("tmp.jpg")


pdb.set_trace()

print("end")