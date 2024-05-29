"""latest version of SuperpointNet. Use it!

"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *
import numpy as np
from pytorch_lightning import LightningModule
import pytorch_lightning as pl
from pathlib import Path
from utils.tools import dict_update
import logging
from Train_model_frontend import Train_model_frontend
import pdb



# from models.SubpixelNet import SubpixelNet
class SuperPointNet_gauss2_pl(pl.LightningModule):
    """ Pytorch definition of SuperPoint Network. """
    default_config = {
        "train_iter": 170000,
        "save_interval": 2000,
        "tensorboard_interval": 200,
        "model": {"subpixel": {"enable": False}},
    }

    def __init__(self, config=None):
        super(SuperPointNet_gauss2_pl, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        # self.inc = inconv(1, c1)
        self.inc = inconv(18, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        # self.down4 = down(c4, 512)
        # self.up1 = up(c4+c3, c2)
        # self.up2 = up(c2+c2, c1)
        # self.up3 = up(c1+c1, c1)
        # self.outc = outconv(c1, subpixel_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.outc = outconv(64, n_classes)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)
        self.output = None


        # config for pl
        # config
        # Update config
        print("Load Train_model_heatmap_mvsec!!")

        self.config = self.default_config
        self.config = dict_update(self.config, config)
        print("check config!!", self.config)

        # init parameters
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
        # self.printImportantConfig()
        pass





    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1) # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
        output = {'semi': semi, 'desc': desc}
        self.output = output

        return output
    
    def training_step(self, sample, n_iter = 0, train=False):
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
        
        # warped images
        if if_warp:
            img_warp, mask_warp_2D = (
                sample["warped_img"],
                sample["warped_valid_mask"],
            )

        # homographies
        if if_warp:
            mat_H, mat_H_inv = sample["homographies"], sample["inv_homographies"]


        # forward + backward + optimize
        if train:
            # print("img: ", img.shape, ", img_warp: ", img_warp.shape)
            outs = self(img.to(self.device))
            semi, coarse_desc = outs["semi"], outs["desc"] # coarse_desc: [1,256,32,43]
            if if_warp:
                outs_warp = self.net(img_warp.to(self.device))
                semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"] # coarse_desc_warp: [1,256,32,43]
        else:
            with torch.no_grad():
                outs = self(img.to(self.device))
                semi, coarse_desc = outs["semi"], outs["desc"]
                if if_warp:
                    outs_warp = self(img_warp.to(self.device))
                    semi_warp, coarse_desc_warp = outs_warp["semi"], outs_warp["desc"]
                pass
        mask_3D_flattened = Train_model_frontend.getMasksStatic(mask_2D, self.cell_size, device=self.device)
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

        # Ensure that loss_desc, positive_dist, and negative_dist require gradients
        # and are part of the computation graph
        loss_desc = loss_desc.requires_grad_()
        positive_dist = positive_dist.requires_grad_()
        negative_dist = negative_dist.requires_grad_()


        # loss = loss_det + loss_det_warp
        loss = loss_det_warp
        if lambda_loss > 0:
            loss += lambda_loss * loss_desc

        # Ensure that loss requires gradients and is part of the computation graph
        loss = loss.requires_grad_()


        self.loss = loss


        self.scalar_dict.update(
            {
                "loss": loss,
                "positive_dist": positive_dist,
                "negative_dist": negative_dist,
            }
        )


        Train_model_frontend.input_to_imgDict(sample, self.images_dict)


        if n_iter % tb_interval == 0 or task == "val":
            logging.info(
                "current iteration: %d, tensorboard_interval: %d", n_iter, tb_interval
            )

            Train_model_frontend.printLossesStatic(self.scalar_dict, task)
            Train_model_frontend.tb_images_dictStatic(task, self.images_dict, max_img=2)
        print("loss ready to return!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return loss

    def configure_optimizers(self):
        optimizer = Train_model_frontend.adamOptimStatic(self, lr=self.config["model"]["learning_rate"])
        return optimizer



    def process_output(self, sp_processer):
        """
        input:
          N: number of points
        return: -- type: tensorFloat
          pts: tensor [batch, N, 2] (no grad)  (x, y)
          pts_offset: tensor [batch, N, 2] (grad) (x, y)
          pts_desc: tensor [batch, N, 256] (grad)
        """
        from utils.utils import flattenDetection
        # from models.model_utils import pred_soft_argmax, sample_desc_from_points
        output = self.output
        semi = output['semi']
        desc = output['desc']
        # flatten
        heatmap = flattenDetection(semi) # [batch_size, 1, H, W]
        # nms
        heatmap_nms_batch = sp_processer.heatmap_to_nms(heatmap, tensor=True)
        # extract offsets
        outs = sp_processer.pred_soft_argmax(heatmap_nms_batch, heatmap)
        residual = outs['pred']
        # extract points
        outs = sp_processer.batch_extract_features(desc, heatmap_nms_batch, residual)

        # output.update({'heatmap': heatmap, 'heatmap_nms': heatmap_nms, 'descriptors': descriptors})
        output.update(outs)
        self.output = output
        return output


def get_matches(deses_SP):
    from models.model_wrap import PointTracker
    tracker = PointTracker(max_length=2, nn_thresh=1.2)
    f = lambda x: x.cpu().detach().numpy()
    # tracker = PointTracker(max_length=2, nn_thresh=1.2)
    # print("deses_SP[1]: ", deses_SP[1].shape)
    matching_mask = tracker.nn_match_two_way(f(deses_SP[0]).T, f(deses_SP[1]).T, nn_thresh=1.2)
    return matching_mask

    # print("matching_mask: ", matching_mask.shape)
    # f_mask = lambda pts, maks: pts[]
    # pts_m = []
    # pts_m_res = []
    # for i in range(2):
    #     idx = xs_SP[i][matching_mask[i, :].astype(int), :]
    #     res = reses_SP[i][matching_mask[i, :].astype(int), :]
    #     print("idx: ", idx.shape)
    #     print("res: ", idx.shape)
    #     pts_m.append(idx)
    #     pts_m_res.append(res)
    #     pass

    # pts_m = torch.cat((pts_m[0], pts_m[1]), dim=1)
    # matches_test = toNumpy(pts_m)
    # print("pts_m: ", pts_m.shape)

    # pts_m_res = torch.cat((pts_m_res[0], pts_m_res[1]), dim=1)
    # # pts_m_res = toNumpy(pts_m_res)
    # print("pts_m_res: ", pts_m_res.shape)
    # # print("pts_m_res: ", pts_m_res)
        
    # pts_idx_res = torch.cat((pts_m, pts_m_res), dim=1)
    # print("pts_idx_res: ", pts_idx_res.shape)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SuperPointNet_gauss2_pl()
    model = model.to(device)


    # check keras-like model summary using torchsummary
    from torchsummary import summary
    summary(model, input_size=(1, 240, 320))

    ## test
    image = torch.zeros((2,1,120, 160))
    outs = model(image.to(device))
    print("outs: ", list(outs))

    from utils.print_tool import print_dict_attr
    print_dict_attr(outs, 'shape')

    from models.model_utils import SuperPointNet_process 
    params = {
        'out_num_points': 500,
        'patch_size': 5,
        'device': device,
        'nms_dist': 4,
        'conf_thresh': 0.015
    }

    sp_processer = SuperPointNet_process(**params)
    outs = model.process_output(sp_processer)
    print("outs: ", list(outs))
    print_dict_attr(outs, 'shape')

    # timer
    import time
    from tqdm import tqdm
    iter_max = 50

    start = time.time()
    print("Start timer!")
    for i in tqdm(range(iter_max)):
        outs = model(image.to(device))
    end = time.time()
    print("forward only: ", iter_max/(end - start), " iter/s")

    start = time.time()
    print("Start timer!")
    xs_SP, deses_SP, reses_SP = [], [], []
    for i in tqdm(range(iter_max)):
        outs = model(image.to(device))
        outs = model.process_output(sp_processer)
        xs_SP.append(outs['pts_int'].squeeze())
        deses_SP.append(outs['pts_desc'].squeeze())
        reses_SP.append(outs['pts_offset'].squeeze())
    end = time.time()
    print("forward + process output: ", iter_max/(end - start), " iter/s")

    start = time.time()
    print("Start timer!")
    for i in tqdm(range(len(xs_SP))):
        get_matches([deses_SP[i][0], deses_SP[i][1]])
    end = time.time()
    print("nn matches: ", iter_max/(end - start), " iters/s")


if __name__ == '__main__':
    main()



