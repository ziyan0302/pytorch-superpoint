"""This is the main training interface using heatmap trick

Author: You-Yi Jau, Rui Zhu
Date: 2019/12/12
"""

import numpy as np
import torch
# from torch.autograd import Variable
# import torch.backends.cudnn as cudnn
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
# from tqdm import tqdm
# from utils.loader import dataLoader, modelLoader, pretrainedLoader
import logging

from utils.tools import dict_update

# from utils.utils import labels2Dto3D, flattenDetection, labels2Dto3D_flattened
# from utils.utils import pltImshow, saveImg
from utils.utils import precisionRecall_torch
# from utils.utils import save_checkpoint

from pathlib import Path
from Train_model_frontend import Train_model_frontend
import pdb
from pytorch_lightning import LightningModule, Trainer
import pytorch_lightning as pl
from datasets.event_utils import gen_event_images
import visualization_code.visualization_utils as vutil
import cv2




def thd_img(img, thd=0.015):
    img[img < thd] = 0
    img[img >= thd] = 1
    return img


def toNumpy(tensor):
    return tensor.detach().cpu().numpy()


def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


class Train_model_heatmap_mvsecImg(Train_model_frontend):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    """
    * SuperPointFrontend_torch:
    ** note: the input, output is different from that of SuperPointFrontend
    heatmap: torch (batch_size, H, W, 1)
    dense_desc: torch (batch_size, H, W, 256)
    pts: [batch_size, np (N, 3)]
    desc: [batch_size, np(256, N)]
    """
    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
        "data": {"gaussian_label": {"enable": False}},
    }

    def __init__(self, config, save_path=Path("."), device="cpu", verbose=False):
        # config
        # Update config
        print("Load Train_model_heatmap_mvsecImg!!")

        self.config = self.default_config
        self.config = dict_update(self.config, config)
        print("check config!!", self.config)

        # init parameters
        self.device = device
        self.save_path = save_path
        self._train = True
        self._eval = True
        self.cell_size = 8
        self.subpixel = False

        self.max_iter = config["train_iter"]

        self.gaussian = False
        if self.config["data"]["gaussian_label"]["enable"]:
            self.gaussian = True

        if self.config["model"]["dense_loss"]["enable"]:
            print("use dense_loss!")
            from utils.utils import descriptor_loss
            self.desc_params = self.config["model"]["dense_loss"]["params"]
            self.descriptor_loss = descriptor_loss
            self.desc_loss_type = "dense"
        elif self.config["model"]["sparse_loss"]["enable"]:
            print("use sparse_loss!")
            self.desc_params = self.config["model"]["sparse_loss"]["params"]
            from utils.loss_functions.sparse_loss import batch_descriptor_loss_sparse

            self.descriptor_loss = batch_descriptor_loss_sparse
            self.desc_loss_type = "sparse"

        # load model
        # self.net = self.loadModel(*config['model'])
        self.printImportantConfig()
        pass


    def detector_loss(self, input, target, mask=None, loss_type="softmax"):
        """
        # apply loss on detectors, default is softmax
        :param input: prediction
            tensor [batch_size, 65, Hc, Wc]
        :param target: constructed from labels
            tensor [batch_size, 65, Hc, Wc]
        :param mask: valid region in an image
            tensor [batch_size, 1, Hc, Wc]
        :param loss_type:
            str (l2 or softmax)
            softmax is used in original paper
        :return: normalized loss
            tensor
        """
        if loss_type == "l2":
            loss_func = nn.MSELoss(reduction="mean")
            loss = loss_func(input, target)
        elif loss_type == "softmax":
            loss_func_BCE = nn.BCELoss(reduction='none').cuda()
            loss = loss_func_BCE(nn.functional.softmax(input, dim=1), target)
            loss = (loss.sum(dim=1) * mask).sum()
            loss = loss / (mask.sum() + 1e-10)
        return loss

    def train_val_sample(self, sample, n_iter=0, train=False):
        """
        # key function
        :param sample:
        :param n_iter:
        :param train:
        :return:
        """

        task = "train" if train else "val"
        tb_interval = self.config["tensorboard_interval"]
        if_warp = self.config['data']['warped_pair']['enable']

        self.scalar_dict, self.images_dict, self.hist_dict = {}, {}, {}
        ## get the inputs
        # logging.info('get input img and label')
        img, mask_2D = (
            sample["image"],
            sample["valid_mask"],
        )

        # variables
        batch_size, H, W = img.shape[0], img.shape[2], img.shape[3]
        self.batch_size = batch_size
        # print("batch_size: ", batch_size)
        Hc = H // self.cell_size
        Wc = W // self.cell_size

        # warped images
        # img_warp, labels_warp_2D, mask_warp_2D = sample['warped_img'].to(self.device), \
        #     sample['warped_labels'].to(self.device), \
        #     sample['warped_valid_mask'].to(self.device)
        if if_warp:
            img_warp, mask_warp_2D = (
                sample["warped_img"],
                sample["warped_valid_mask"],
            )

        # homographies
        # mat_H, mat_H_inv = \
        # sample['homographies'].to(self.device), sample['inv_homographies'].to(self.device)
        if if_warp:
            mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        if train:
            # print("img: ", img.shape, ", img_warp: ", img_warp.shape)
            outs = self.net(img.to(self.device))
            semi, coarse_desc = outs["semi"], outs["desc"] # coarse_desc: [1,256,32,43]
            if if_warp:
                outs_warp = self.net(img_warp.to(self.device))
                semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"] # coarse_desc_warp: [1,256,32,43]
        else:
            with torch.no_grad():
                outs = self.net(img.to(self.device))
                semi, coarse_desc = outs["semi"], outs["desc"]
                if if_warp:
                    outs_warp = self.net(img_warp.to(self.device))
                    semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]
                pass

        # detector loss
        from utils.utils import labels2Dto3D

        mask_3D_flattened = self.getMasks(mask_2D, self.cell_size, device=self.device)
        # mask_3D_flattened = self.getMasks(mask_warp_2D, self.cell_size, device=self.device)


        mask_desc = mask_3D_flattened.unsqueeze(1) # [1,1,40,30]
        lambda_loss = self.config["model"]["lambda_loss"]


        
        # descriptor loss
        if lambda_loss > 0:
            assert if_warp == True, "need a pair of images"
            loss_desc, mask, positive_dist, negative_dist = self.descriptor_loss(
                coarse_desc,
                coarse_desc_warp,
                mat_H,
                img,
                mask_valid=mask_desc,
                device=self.device,
                **self.desc_params
            )
        else:
            ze = torch.tensor([0]).to(self.device)
            loss_desc, positive_dist, negative_dist = ze, ze, ze
        if None in [loss_desc, positive_dist, negative_dist]:
            print("loss_desc, positive_dist, negative_dist: ", loss_desc, positive_dist, negative_dist)
            print("skipping loss backward!!!!!!!!!!!!!!!11")
            return None

        loss_det_warp = torch.tensor([0]).float().to(self.device)
        # loss = loss_det + loss_det_warp
        loss = loss_det_warp
        if lambda_loss > 0:
            loss += lambda_loss * loss_desc

        self.loss = loss


        self.scalar_dict.update(
            {
                "loss": loss,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )


        self.input_to_imgDict(sample, self.images_dict)

        if train:
            loss.backward()
            self.optimizer.step()

        if n_iter % tb_interval == 0 or task == "val":
            logging.info(
                "current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval
            )

            self.printLosses(self.scalar_dict, task)


            ## draw predicted corresponding pair of interest points  
            image_a_pred = coarse_desc[0].unsqueeze(0)
            warped_a_pred = coarse_desc_warp[0].unsqueeze(0)
            Hc, Wc = image_a_pred.shape[2], image_a_pred.shape[3]
            img_shape = (Hc, Wc)

            homographies_H = sample["homographies"][0]
            homographies_H = vutil.scale_homography_torch(homographies_H, img_shape, shift=(-1, -1))

            # get uva and warped uvb, the predicted description got by uva and predicted description got by uvb
            # the length of uva and uvb is the same 
            uv_a, uv_b, matches_a_descriptors, matches_b_descriptors = vutil.img2descAndWarpedDesc(image_a_pred, warped_a_pred, img, homographies_H)

            # turn the pts to pixel coordinate
            input_uv = vutil.uvListTransformer(uv_a, Wc, Hc)
            pred_uv = vutil.uvListTransformer(uv_b, Wc, Hc)

            # Compute distances between descriptions
            # for each query idx, find the target idx with largest similarity
            distances_matrix = vutil.compute_distances(matches_a_descriptors.cpu().detach().numpy(), matches_b_descriptors.cpu().detach().numpy())
            column_index_with_smallest_value = np.argmax(distances_matrix, axis=1)
            matches = []
            import cv2
            for row_index, col_index in enumerate(column_index_with_smallest_value):
                # Create a cv2.DMatch object with query index as row_index, train index as col_index,
                # and optionally set distance to 0
                dmatch = cv2.DMatch(row_index, col_index, 0)
                # Append the created dmatch to the list
                matches.append(dmatch)

            img1 = sample['raw_image'][0].squeeze().cpu().numpy()
            img2 = sample['warped_rawImg'][0].squeeze().cpu().numpy()
            out1 = cv2.drawKeypoints(img1, input_uv, None)
            out2 = cv2.drawKeypoints(out1, pred_uv, None)
        
            points1 = np.float32([input_uv[match.queryIdx].pt for match in matches])
            points2 = np.float32([pred_uv[match.trainIdx].pt for match in matches])
            for pt1, pt2 in zip(points1, points2): 
                pt1 = tuple(map(int, pt1))  # Convert to integer coordinates
                pt2 = tuple(map(int, pt2))  # Convert to integer coordinates
                # cv2.line(out2, pt1, pt2, (0, 255, 0), 2)  # Draw a line from pt1 to pt2
                cv2.arrowedLine(out2, pt1, pt2, (0, 255, 0), thickness=2, tipLength=0.05)

            out2_numpy = cv2.cvtColor(out2, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            out2_numpy = np.transpose(out2_numpy, (2, 0, 1))  # Convert HWC to CHW
            # Convert NumPy array to tensor
            out2_tensor = torch.tensor(out2_numpy, dtype=torch.uint8)
            self.images_dict['matched_result'] = out2_tensor.unsqueeze(0)
            self.images_dict['raw_image'] = torch.tensor(img1, dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
            self.images_dict['raw_warpedimage'] = torch.tensor(img2, dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
    
            ## add event_img to tf
            event_img = sample['image']
            self.images_dict['event_volume'] = event_img
            self.tb_images_dict(task, self.images_dict, max_img=2)
            self.tb_hist_dict(task, self.hist_dict)

        self.tb_scalar_dict(self.scalar_dict, task)

        return loss.item()



if __name__ == "__main__":
    # load config
    # filename = "configs/superpoint_coco_train_heatmap.yaml"
    filename = "configs/superpoint_mvsec_train_heatmap.yaml"
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

    model_fe = Train_model_frontend(config)
    print('==> Successfully loaded pre-trained network.')

    train_agent = Train_model_heatmap_mvsecImg(config, device=device)

    train_agent.train_loader = train_loader
    # train_agent.val_loader = val_loader

    train_agent.loadModel()
    train_agent.dataParallel()
    train_agent.train()

    
    # epoch += 1
    try:
        model_fe.train()

    # catch exception
    except KeyboardInterrupt:
        logging.info("ctrl + c is pressed. save model")
    # is_best = True
