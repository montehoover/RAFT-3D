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

def prepare_image_and_depth(image, depth):
    """ padding, normalization, and scaling """

    batch_size, ch, ht, wd = image.shape
    pad = ht % 8 
    image = F.pad(image, [0,0,0,pad], mode='replicate')
    depth = F.pad(depth[:,None], [0,0,0,pad], mode='replicate')[:,0]
    image = normalize_image(image)
    depth = (DEPTH_SCALE * depth).float()
    return image, depth

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

def display(disp_set):
    """ display se3 fields """

    rows = len(disp_set)
    cols = len(disp_set[1])
    fig, axs = plt.subplots(rows, cols)
    for i in range(rows):
        for j in range(cols):
            img = disp_set[i][j]
            # Always treat first row as rbg images
            if i == 0:
                img = img[:, :, ::-1] / 255.0
            else:
                img = np.clip(img, -0.1, 0.1)
                img = (img + 0.1) / 0.2
            axs[i, j].imshow(img)
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
    intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda().unsqueeze(0)

    img_nums = ['315', '330', '345', '360', '375', '390', '405', '420', '435']
    img_jpgs = []
    imgs = []
    depths = []
    for img_num in img_nums:
        print(f"Calculating SE3 motion for image {img_num}...")
        img_jpg = cv2.imread(f'assets/kinect/puma_narrow_fov/color/{img_num}.jpg')
        img = torch.from_numpy(img_jpg).permute(2,0,1).float().cuda().unsqueeze(0)
        depth = cv2.imread(f'assets/kinect/puma_narrow_fov/depth/{img_num}.png', cv2.IMREAD_UNCHANGED)
        depth = np.where(depth == 0, depth.mean(), depth)
        depth = torch.from_numpy(1/4 * depth).float().cuda().unsqueeze(0)
        img, depth = prepare_image_and_depth(img, depth)
        img_jpgs.append(img_jpg)
        imgs.append(img)
        depths.append(depth)
    
    Ts = []
    taus = []
    phis = []
    for i in range(len(imgs) - 1):
        T = model(imgs[i], imgs[i+1], depths[i], depths[i+1], intrinsics, iters=16)
        tau, phi = T.log().split([3,3], dim=-1)
        tau = tau[0].cpu().numpy()
        phi = phi[0].cpu().numpy()
        Ts.append(T)
        taus.append(tau)
        phis.append(phi)

    display_set = [img_jpgs, taus, phis]

    display(display_set)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft3d.pth', help='checkpoint to restore')
    parser.add_argument('--network', default='raft3d.raft3d', help='network architecture')
    args = parser.parse_args()

    demo(args)

    


