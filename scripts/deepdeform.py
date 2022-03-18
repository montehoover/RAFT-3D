import sys
sys.path.append('.')

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from lietorch import SE3
import raft3d.projective_ops as pops
from data_readers import frame_utils
from utils import show_image, normalize_image


DEPTH_SCALE = 0.2

def prepare_images_and_depths(image1, image2, depth1, depth2):
    """ padding, normalization, and scaling """

    # assert image1.shape == image2.shape and image1.shape == depth1.shape and image1.shape == depth2.shape, \
    #     "All RGB and depth images must have matching dimensions."
    batch_size, ch, ht, wd = image1.shape
    pad = ht % 8 

    image1 = F.pad(image1, [0,0,0,pad], mode='replicate')
    image2 = F.pad(image2, [0,0,0,pad], mode='replicate')
    depth1 = F.pad(depth1[:,None], [0,0,0,pad], mode='replicate')[:,0]
    depth2 = F.pad(depth2[:,None], [0,0,0,pad], mode='replicate')[:,0]

    depth1 = (DEPTH_SCALE * depth1).float()
    depth2 = (DEPTH_SCALE * depth2).float()
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)

    return image1, image2, depth1, depth2


def display(img, tau, phi):
    """ display se3 fields """
    fig, (ax1, ax2, ax3) = plt.subplots(1,3)
    ax1.imshow(img[:, :, ::-1] / 255.0)

    tau_img = np.clip(tau, -0.1, 0.1)
    tau_img = (tau_img + 0.1) / 0.2

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    ax2.imshow(tau_img)
    ax3.imshow(phi_img)
    plt.show()

def display2(img1, img2, tau, phi):
    """ display se3 fields """
    fig, axs = plt.subplots(2,2)
    axs[0, 0].imshow(img1[:, :, ::-1] / 255.0)
    axs[0, 1].imshow(img2[:, :, ::-1] / 255.0)


    tau_img = np.clip(tau, -0.1, 0.1)
    tau_img = (tau_img + 0.1) / 0.2

    phi_img = np.clip(phi, -0.1, 0.1)
    phi_img = (phi_img + 0.1) / 0.2

    axs[1, 0].imshow(tau_img)
    axs[1, 1].imshow(phi_img)
    plt.show()


@torch.no_grad()
def demo(args):
    import importlib
    RAFT3D = importlib.import_module(args.network).RAFT3D
    model = torch.nn.DataParallel(RAFT3D(args))
    model.load_state_dict(torch.load(args.model), strict=False)

    model.eval()
    model.cuda()
    
    fx, fy, cx, cy = (5.745410000000000537e+02, 5.775839999999999463e+02, 3.225230000000000246e+02, 2.385589999999999975e+02)
    img1 = cv2.imread('assets/deepdeform/train/seq070/color/000030.jpg')
    img2 = cv2.imread('assets/deepdeform/train/seq070/color/000035.jpg')
    disp1 = cv2.imread('assets/deepdeform/train/seq070/depth/000030.png', cv2.IMREAD_UNCHANGED)
    disp2 = cv2.imread('assets/deepdeform/train/seq070/depth/000035.png', cv2.IMREAD_UNCHANGED)
    disp1 = np.where(disp1 == 0, disp1.mean(), disp1)
    disp2 = np.where(disp2 == 0, disp2.mean(), disp2)

    depth1 = torch.from_numpy(1/4 * disp1).float().cuda().unsqueeze(0)
    depth2 = torch.from_numpy(1/4 * disp2).float().cuda().unsqueeze(0)
    image1 = torch.from_numpy(img1).permute(2,0,1).float().cuda().unsqueeze(0)
    image2 = torch.from_numpy(img2).permute(2,0,1).float().cuda().unsqueeze(0)
    intrinsics_2 = torch.as_tensor([fx, fy, cx, cy]).cuda().unsqueeze(0)

    image1_2, image2_2, depth1_2, depth2_2 = prepare_images_and_depths(image1, image2, depth1, depth2)

    # Ts = model(image1_1, image2_1, depth1_1, depth2_1, intrinsics_1, image1_2, image2_2, depth1_2, depth2_2, intrinsics_2, iters=16)
    Ts_1 = model(image1_1, image2_1, depth1_1, depth2_1, intrinsics_1, iters=16)
    Ts_2 = model(image1_2, image2_2, depth1_2, depth2_2, intrinsics_2, iters=16)

    
    # compute 2d and 3d from from SE3 field (Ts)
    # flow2d_1, flow3d_1, _ = pops.induced_flow(Ts_1, depth1_1, intrinsics_1)
    # flow2d_2, flow3d_2, _ = pops.induced_flow(Ts_2, depth1_2, intrinsics_2)

    # extract rotational and translational components of Ts
    # tau, phi = Ts_1.log().split([3,3], dim=-1)
    # tau = tau[0].cpu().numpy()
    # phi = phi[0].cpu().numpy()

    tau_2, phi_2 = Ts_2.log().split([3,3], dim=-1)
    tau_2 = tau_2[0].cpu().numpy()
    phi_2 = phi_2[0].cpu().numpy()

    # undo depth scaling
    # flow3d_1 = flow3d_1 / DEPTH_SCALE
    # flow3d_2 = flow3d_2 / DEPTH_SCALE

    # display2(img1, img2, tau, phi)
    display2(img1, img2, tau_2, phi_2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft3d.pth', help='checkpoint to restore')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    args = parser.parse_args()

    demo(args)

    


