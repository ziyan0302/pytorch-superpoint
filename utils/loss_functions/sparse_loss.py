import utils.correspondence_tools.correspondence_finder as correspondence_finder
import numpy as np
import torch

from utils.homographies import scale_homography_torch
from utils.loss_functions.pixelwise_contrastive_loss import PixelwiseContrastiveLoss
import cv2
import pdb

def get_coor_cells(Hc, Wc, cell_size, device='cpu', uv=False):
    coor_cells = torch.stack(torch.meshgrid(torch.arange(Hc), torch.arange(Wc)), dim=2)
    coor_cells = coor_cells.type(torch.FloatTensor).to(device)
    coor_cells = coor_cells.view(-1, 2)
    # change vu to uv
    if uv:
        coor_cells = torch.stack((coor_cells[:,1], coor_cells[:,0]), dim=1) # (y, x) to (x, y)

    return coor_cells.to(device)

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


def create_non_matches(uv_a, uv_b_non_matches, multiplier):
    """
    Simple wrapper for repeated code
    :param uv_a:
    :type uv_a:
    :param uv_b_non_matches:
    :type uv_b_non_matches:
    :param multiplier:
    :type multiplier:
    :return:
    :rtype:
    """
    uv_a_long = (torch.t(uv_a[0].repeat(multiplier, 1)).contiguous().view(-1, 1),
                 torch.t(uv_a[1].repeat(multiplier, 1)).contiguous().view(-1, 1)) # ((100000,1), (100000,1))

    uv_b_non_matches_long = (uv_b_non_matches[0].view(-1, 1), uv_b_non_matches[1].view(-1, 1))

    return uv_a_long, uv_b_non_matches_long


def descriptor_loss_sparse(descriptors, descriptors_warped, homographies, img, mask_valid=None,
                           cell_size=8, device='cpu', descriptor_dist=4, lamda_d=250,
                           num_matching_attempts=1000, num_masked_non_matches_per_match=10, 
                           dist='cos', method='1d', 
                           pos_margin=1.0, neg_margin=0.2,
                           pts_sample_rate = 11,
                             **config):
    """
    consider batches of descriptors
    :param descriptors:
        Output from descriptor head
        tensor [descriptors, Hc, Wc]
    :param descriptors_warped:
        Output from descriptor head of warped image
        tensor [descriptors, Hc, Wc]
    """

    def uv_to_tuple(uv):
        return (uv[:, 0], uv[:, 1])

    def tuple_to_uv(uv_tuple):
        return torch.stack([uv_tuple[0], uv_tuple[1]])

    def tuple_to_1d(uv_tuple, W, uv=True):
        if uv:
            return uv_tuple[0] + uv_tuple[1]*W
        else:
            return uv_tuple[0]*W + uv_tuple[1]


    def uv_to_1d(points, W, uv=True):
        # assert points.dim == 2
        #     print("points: ", points[0])
        #     print("H: ", H)
        if uv:
            return points[..., 0] + points[..., 1]*W
        else:
            return points[..., 0]*W + points[..., 1]

    ## calculate matches loss
    def get_match_loss(image_a_pred, image_b_pred, matches_a, matches_b, dist='cos', method='1d'):
        match_loss, matches_a_descriptors, matches_b_descriptors = \
            PixelwiseContrastiveLoss.match_loss(image_a_pred, image_b_pred, 
                matches_a, matches_b, dist=dist, method=method, M = pos_margin)
        return match_loss

    def get_non_matches_corr(img_b_shape, uv_a, uv_b_matches, num_masked_non_matches_per_match=10, device='cpu'):
        ## sample non matches
        uv_b_matches = uv_b_matches.squeeze()
        uv_b_matches_tuple = uv_to_tuple(uv_b_matches) # (u: (1000,), v: (1000,))
        uv_b_non_matches_tuple = correspondence_finder.create_non_correspondences(uv_b_matches_tuple,
                                        img_b_shape, num_non_matches_per_match=num_masked_non_matches_per_match,
                                        img_b_mask=None) # find 100 pts to every pt # (u:(1000,100),v:(1000,100))
        # pdb.set_trace()
        ## create_non_correspondences
        #     print("img_b_shape ", img_b_shape)
        #     print("uv_b_matches ", uv_b_matches.shape)
        # print("uv_a: ", uv_to_tuple(uv_a))
        # print("uv_b_non_matches: ", uv_b_non_matches)
        #     print("uv_b_non_matches: ", tensorUv2tuple(uv_b_non_matches))
        # uv_b_non_matches_tuple[0].shape
        # uv_b_matches.shape
        # uv_a.shape
        tmp = uv_to_tuple(uv_a)
        tmp[0].shape
        uv_a_long = (torch.t(tmp[0].repeat(100, 1)).contiguous().view(-1, 1), torch.t(tmp[1].repeat(100, 1)).contiguous().view(-1, 1))
        uv_a_long[0].shape
        uv_a_tuple, uv_b_non_matches_tuple = \
            create_non_matches(uv_to_tuple(uv_a), uv_b_non_matches_tuple, num_masked_non_matches_per_match)
        return uv_a_tuple, uv_b_non_matches_tuple

    def get_non_match_loss(image_a_pred, image_b_pred, non_matches_a, non_matches_b, dist='cos'):
        ## non matches loss
        non_match_loss, num_hard_negatives, non_matches_a_descriptors, non_matches_b_descriptors = \
            PixelwiseContrastiveLoss.non_match_descriptor_loss(image_a_pred, image_b_pred,
                                                               non_matches_a.long().squeeze(),
                                                               non_matches_b.long().squeeze(),
                                                               M=neg_margin, invert=True, dist=dist)
        non_match_loss = non_match_loss.sum()/(num_hard_negatives + 1)
        return non_match_loss

    def findInterestPoints(img, blockSize=2, ksize=1, k=0.1):
        img_mean = torch.mean(img, dim=1)
        result = torch.where(img_mean > 0, torch.tensor(1), torch.tensor(0)).cpu().numpy()[0]
        # Apply Harris corner detection
        # Adjust the parameter k to control sensitivity to corners
        # You may need to experiment with different values of k
        dst = cv2.cornerHarris(np.uint8(result), blockSize, ksize, k)
        dst.max()
        # Threshold the response to highlight corners
        # You may need to adjust the threshold value according to your image
        threshold = 0.02 * dst.max()
        # Find coordinates where dst exceeds the threshold
        coordinates = np.argwhere(dst > threshold)
        # make uv to be in [0,1]
        coordinates = coordinates/np.array([img.shape[-2], img.shape[-1]])
        return coordinates

    from utils.utils import filter_points
    from utils.utils import crop_or_pad_choice
    from utils.utils import normPts

    Hc, Wc = descriptors.shape[1], descriptors.shape[2]
    img_shape = (Hc, Wc)

    def descriptor_reshape(descriptors):
        descriptors = descriptors.view(-1, Hc * Wc).transpose(0, 1)  # torch [D, H, W] --> [H*W, d]
        descriptors = descriptors.unsqueeze(0)  # torch [1, H*W, D]
        return descriptors

    image_a_pred = descriptor_reshape(descriptors)  # torch [1, H*W, D]
    image_b_pred = descriptor_reshape(descriptors_warped)  # torch [batch_size, H*W, D]

    # matches
    # uv_a = get_coor_cells(Hc, Wc, cell_size, uv=True, device='cpu')
    if (1):
        # finding interest points on event img
        uv_a = findInterestPoints(img, blockSize=2, ksize=1, k=0.1)
        idxs_a = np.arange(0,uv_a.shape[0],pts_sample_rate)
        uv_a = uv_a[idxs_a]
        uv_a = uv_a * np.array([Hc-1, Wc-1])

        uv_a[:,0], uv_a[:,1] = uv_a[:,1].copy(), uv_a[:,0].copy()  
        # pdb.set_trace()
        uv_a = torch.from_numpy(uv_a).float()
        

    homographies_H = scale_homography_torch(homographies, img_shape, shift=(-1, -1))


    uv_b_matches = warp_coor_cells_with_homographies(uv_a, homographies_H.to('cpu'), uv=True, device='cpu')
    uv_b_matches.round_() 
    uv_b_matches = uv_b_matches.squeeze(0)

    # filtering out of range points


    uv_b_matches, mask = filter_points(uv_b_matches, torch.tensor([Wc, Hc]).to(device='cpu'), return_mask=True)
    # filter out the points that would be out of image shape after warpped
    # first filter out uv_b_matches
    
    if (0):
        torch.count_nonzero(mask)
        uv_b_matches

    uv_a = uv_a[mask] # then filter out uv_a (points before warpped)
    if uv_a.shape[0] == 0:
        print("uv_a return None!!!!!!!!!!!!!!!!!!")
        return None, None, None 
    
    # crop to the same length
    shuffle = True
    if not shuffle: print("shuffle: ", shuffle)
    choice = crop_or_pad_choice(uv_b_matches.shape[0], num_matching_attempts, shuffle=shuffle)
    # resample the points to be a index array with fixed amount=1000
    choice = torch.tensor(choice).type(torch.int64)
    choice.shape
    uv_a = uv_a[choice]
    uv_b_matches = uv_b_matches[choice]

    if method == '2d':
        matches_a = normPts(uv_a, torch.tensor([Wc, Hc]).float()) # [u, v] # [-1,1]
        matches_b = normPts(uv_b_matches, torch.tensor([Wc, Hc]).float()) # [-1,1]
    else:
        matches_a = uv_to_1d(uv_a, Wc)
        matches_b = uv_to_1d(uv_b_matches, Wc)


    # important loss
    if method == '2d':
        match_loss = get_match_loss(descriptors, descriptors_warped, matches_a.to(device), 
            matches_b.to(device), dist=dist, method='2d')
    else:
        match_loss = get_match_loss(image_a_pred, image_b_pred, 
            matches_a.long().to(device), matches_b.long().to(device), dist=dist)

    # non matches

    # get non matches correspondence
    uv_a_tuple, uv_b_non_matches_tuple = get_non_matches_corr(img_shape,
                                            uv_a, uv_b_matches,
                                            num_masked_non_matches_per_match=num_masked_non_matches_per_match)
  
    non_matches_a = tuple_to_1d(uv_a_tuple, Wc)
    non_matches_b = tuple_to_1d(uv_b_non_matches_tuple, Wc)

    non_match_loss = get_non_match_loss(image_a_pred, image_b_pred, non_matches_a.to(device),
                                        non_matches_b.to(device), dist=dist)

    # TODO: experiment on only match_loss
    loss = lamda_d * match_loss + non_match_loss
    # print("loss: ", loss, " match_loss: ", match_loss, " nonMatLoss: ", non_match_loss)
    return loss, lamda_d * match_loss, non_match_loss
    pass

"""
img[uv_b_matches.long()[:,1],uv_b_matches.long()[:,0]] = 1
from utils.utils import pltImshow
pltImshow(img.numpy())

"""

def batch_descriptor_loss_sparse(descriptors, descriptors_warped, homographies, img, **options):
    loss = []
    pos_loss = []
    neg_loss = []
    batch_size = descriptors.shape[0]
    for i in range(batch_size):
        losses = descriptor_loss_sparse(descriptors[i], descriptors_warped[i],
                    # torch.tensor(homographies[i], dtype=torch.float32), **options)
                    homographies[i].type(torch.float32), img, **options)
        if (losses[0] == None or losses[1] == None or losses[2] == None):
           print("return nones!!!!!!!!!!!!")
           return None, None, None, None 
        loss.append(losses[0])
        pos_loss.append(losses[1])
        neg_loss.append(losses[2])
    import pdb
    # pdb.set_trace()
    loss, pos_loss, neg_loss = torch.stack(loss), torch.stack(pos_loss), torch.stack(neg_loss)
    return loss.mean(), None, pos_loss.mean(), neg_loss.mean()

if __name__ == '__main__':
    # config
    H, W = 240, 320
    cell_size = 8
    Hc, Wc = H // cell_size, W // cell_size

    D = 3
    torch.manual_seed(0)
    np.random.seed(0)

    batch_size = 2
    device = 'cpu'
    method = '2d'

    num_matching_attempts = 1000
    num_masked_non_matches_per_match = 200
    lamda_d = 1

    homographies = np.identity(3)[np.newaxis, :, :]
    homographies = np.tile(homographies, [batch_size, 1, 1])

    def randomDescriptor():
        descriptors = torch.tensor(np.random.rand(2, D, Hc, Wc)-0.5, dtype=torch.float32)
        dn = torch.norm(descriptors, p=2, dim=1)  # Compute the norm.
        descriptors = descriptors.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.
        return descriptors

    # descriptors = torch.tensor(np.random.rand(2, D, Hc, Wc), dtype=torch.float32)
    # dn = torch.norm(descriptors, p=2, dim=1) # Compute the norm.
    # desc = descriptors.div(torch.unsqueeze(dn, 1)) # Divide by norm to normalize.
    descriptors = randomDescriptor()
    # print("descriptors: ", descriptors.shape)
    # descriptors_warped = torch.tensor(np.random.rand(2, D, Hc, Wc), dtype=torch.float32)
    descriptors_warped = randomDescriptor()
    descriptor_loss = descriptor_loss_sparse(descriptors[0], descriptors_warped[0],
                                             torch.tensor(homographies[0], dtype=torch.float32),
                                             method=method)
    print("descriptor_loss: ", descriptor_loss)

    # loss = batch_descriptor_loss_sparse(descriptors, descriptors_warped,
    #                                     torch.tensor(homographies, dtype=torch.float32),
    #                                     num_matching_attempts = num_matching_attempts,
    #                                     num_masked_non_matches_per_match = num_masked_non_matches_per_match,
    #                                     device=device,
    #                                     lamda_d = lamda_d, 
    #                                     method=method)
    # print("batch descriptor_loss: ", loss)

    loss = batch_descriptor_loss_sparse(descriptors, descriptors,
                                        torch.tensor(homographies, dtype=torch.float32),
                                        num_matching_attempts = num_matching_attempts,
                                        num_masked_non_matches_per_match = num_masked_non_matches_per_match,
                                        device=device,
                                        lamda_d = lamda_d,
                                        method=method)
    print("same descriptor_loss (pos should be 0): ", loss)

