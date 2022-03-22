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
    
    
    # img1 = cv2.imread(f'assets/deepdeform/train/seq070/color/000{imgs[0]}.jpg')
    # img2 = cv2.imread(f'assets/deepdeform/train/seq070/color/000{imgs[1]}.jpg')
    # img3 = cv2.imread(f'assets/deepdeform/train/seq070/color/000{imgs[2]}.jpg')
    # img4 = cv2.imread(f'assets/deepdeform/train/seq070/color/000{imgs[3]}.jpg')
    # img5 = cv2.imread(f'assets/deepdeform/train/seq070/color/000{imgs[4]}.jpg')
    # img6 = cv2.imread(f'assets/deepdeform/train/seq070/color/000{imgs[5]}.jpg')
    # img7 = cv2.imread(f'assets/deepdeform/train/seq070/color/000{imgs[6]}.jpg')
    # img8 = cv2.imread(f'assets/deepdeform/train/seq070/color/000{imgs[7]}.jpg')
    # disp1 = cv2.imread(f'assets/deepdeform/train/seq070/depth/000{imgs[0]}.png', cv2.IMREAD_UNCHANGED)
    # disp2 = cv2.imread(f'assets/deepdeform/train/seq070/depth/000{imgs[1]}.png', cv2.IMREAD_UNCHANGED)
    # disp3 = cv2.imread(f'assets/deepdeform/train/seq070/depth/000{imgs[2]}.png', cv2.IMREAD_UNCHANGED)
    # disp4 = cv2.imread(f'assets/deepdeform/train/seq070/depth/000{imgs[3]}.png', cv2.IMREAD_UNCHANGED)
    # disp5 = cv2.imread(f'assets/deepdeform/train/seq070/depth/000{imgs[4]}.png', cv2.IMREAD_UNCHANGED)
    # disp6 = cv2.imread(f'assets/deepdeform/train/seq070/depth/000{imgs[5]}.png', cv2.IMREAD_UNCHANGED)
    # disp7 = cv2.imread(f'assets/deepdeform/train/seq070/depth/000{imgs[6]}.png', cv2.IMREAD_UNCHANGED)
    # disp8 = cv2.imread(f'assets/deepdeform/train/seq070/depth/000{imgs[7]}.png', cv2.IMREAD_UNCHANGED)
    # disp1 = np.where(disp1 == 0, disp1.mean(), disp1)
    # disp2 = np.where(disp2 == 0, disp2.mean(), disp2)
    # disp3 = np.where(disp3 == 0, disp3.mean(), disp3)
    # disp4 = np.where(disp4 == 0, disp4.mean(), disp4)
    # disp5 = np.where(disp5 == 0, disp5.mean(), disp5)
    # disp6 = np.where(disp6 == 0, disp6.mean(), disp6)
    # disp7 = np.where(disp7 == 0, disp7.mean(), disp7)
    # disp8 = np.where(disp8 == 0, disp8.mean(), disp8)

    # image1 = torch.from_numpy(img1).permute(2,0,1).float().cuda().unsqueeze(0)
    # image2 = torch.from_numpy(img2).permute(2,0,1).float().cuda().unsqueeze(0)
    # image3 = torch.from_numpy(img3).permute(2,0,1).float().cuda().unsqueeze(0)
    # image4 = torch.from_numpy(img4).permute(2,0,1).float().cuda().unsqueeze(0)
    # image5 = torch.from_numpy(img5).permute(2,0,1).float().cuda().unsqueeze(0)
    # image6 = torch.from_numpy(img6).permute(2,0,1).float().cuda().unsqueeze(0)
    # image7 = torch.from_numpy(img7).permute(2,0,1).float().cuda().unsqueeze(0)
    # image8 = torch.from_numpy(img8).permute(2,0,1).float().cuda().unsqueeze(0)
    # depth1 = torch.from_numpy(1/4 * disp1).float().cuda().unsqueeze(0)
    # depth2 = torch.from_numpy(1/4 * disp2).float().cuda().unsqueeze(0)
    # depth3 = torch.from_numpy(1/4 * disp3).float().cuda().unsqueeze(0)
    # depth4 = torch.from_numpy(1/4 * disp4).float().cuda().unsqueeze(0)
    # depth5 = torch.from_numpy(1/4 * disp5).float().cuda().unsqueeze(0)
    # depth6 = torch.from_numpy(1/4 * disp6).float().cuda().unsqueeze(0)
    # depth7 = torch.from_numpy(1/4 * disp7).float().cuda().unsqueeze(0)
    # depth8 = torch.from_numpy(1/4 * disp8).float().cuda().unsqueeze(0)
    
    # image1, image2, depth1, depth2 = prepare_images_and_depths(image1, image2, depth1, depth2)
    # image3, image4, depth3, depth4 = prepare_images_and_depths(image3, image4, depth3, depth4)
    # image5, image6, depth5, depth6 = prepare_images_and_depths(image5, image6, depth5, depth6)
    # image7, image8, depth7, depth8 = prepare_images_and_depths(image7, image8, depth7, depth8)

    # print("Calculating SE3 motion...")
    # Ts1 = model(image1, image2, depth1, depth2, intrinsics, iters=16)
    # print("Calculating SE3 motion...")
    # Ts2 = model(image2, image3, depth2, depth3, intrinsics, iters=16)
    # print("Calculating SE3 motion...")
    # Ts3 = model(image3, image4, depth3, depth4, intrinsics, iters=16)
    # print("Calculating SE3 motion...")
    # Ts4 = model(image4, image5, depth4, depth5, intrinsics, iters=16)
    # print("Calculating SE3 motion...")
    # Ts5 = model(image5, image6, depth5, depth6, intrinsics, iters=16)
    # print("Calculating SE3 motion...")
    # Ts6 = model(image6, image7, depth6, depth7, intrinsics, iters=16)
    # print("Calculating SE3 motion...")
    # Ts7 = model(image7, image8, depth7, depth8, intrinsics, iters=16)

    # tau1, phi1 = Ts1.log().split([3,3], dim=-1)
    # tau1, phi2 = Ts2.log().split([3,3], dim=-1)
    # tau1, phi3 = Ts3.log().split([3,3], dim=-1)
    # tau1, phi4 = Ts4.log().split([3,3], dim=-1)
    # tau1, phi5 = Ts5.log().split([3,3], dim=-1)
    # tau1, phi6 = Ts6.log().split([3,3], dim=-1)
    # tau1, phi7 = Ts7.log().split([3,3], dim=-1)

    # phi1 = phi1[0].cpu().numpy()
    # phi2 = phi2[0].cpu().numpy()
    # phi3 = phi3[0].cpu().numpy()
    # phi4 = phi4[0].cpu().numpy()
    # phi5 = phi5[0].cpu().numpy()
    # phi6 = phi6[0].cpu().numpy()
    # phi7 = phi7[0].cpu().numpy()

    # fig, axs = plt.subplots(2,7)
    # axs[0, 0].imshow(img1[:, :, ::-1] / 255.0)
    # axs[0, 1].imshow(img2[:, :, ::-1] / 255.0)
    # axs[0, 2].imshow(img3[:, :, ::-1] / 255.0)
    # axs[0, 3].imshow(img4[:, :, ::-1] / 255.0)
    # axs[0, 4].imshow(img5[:, :, ::-1] / 255.0)
    # axs[0, 5].imshow(img6[:, :, ::-1] / 255.0)
    # axs[0, 6].imshow(img7[:, :, ::-1] / 255.0)

    # phi_img1 = np.clip(phi1, -0.1, 0.1)
    # phi_img2 = np.clip(phi2, -0.1, 0.1)
    # phi_img3 = np.clip(phi3, -0.1, 0.1)
    # phi_img4 = np.clip(phi4, -0.1, 0.1)
    # phi_img5 = np.clip(phi5, -0.1, 0.1)
    # phi_img6 = np.clip(phi6, -0.1, 0.1)
    # phi_img7 = np.clip(phi7, -0.1, 0.1)
    # phi_img1 = (phi_img1 + 0.1) / 0.2
    # phi_img2 = (phi_img2 + 0.1) / 0.2
    # phi_img3 = (phi_img3 + 0.1) / 0.2
    # phi_img4 = (phi_img4 + 0.1) / 0.2
    # phi_img5 = (phi_img5 + 0.1) / 0.2
    # phi_img6 = (phi_img6 + 0.1) / 0.2
    # phi_img7 = (phi_img7 + 0.1) / 0.2

    # axs[1, 0].imshow(phi_img1)
    # axs[1, 1].imshow(phi_img2)
    # axs[1, 2].imshow(phi_img3)
    # axs[1, 3].imshow(phi_img4)
    # axs[1, 4].imshow(phi_img5)
    # axs[1, 5].imshow(phi_img6)
    # axs[1, 6].imshow(phi_img7)

    # plt.show()

    fx, fy, cx, cy = (5.745410000000000537e+02, 5.775839999999999463e+02, 3.225230000000000246e+02, 2.385589999999999975e+02)
    intrinsics = torch.as_tensor([fx, fy, cx, cy]).cuda().unsqueeze(0)

    img_nums = ['055', '075', '095', '115', '135', '145', '155', '165']
    img_jpgs = []
    imgs = []
    depths = []
    for img_num in img_nums:
        print(f"Calculating SE3 motion for image {img_num}...")
        img_jpg = cv2.imread(f'assets/deepdeform/train/seq070/color/000{img_num}.jpg')
        img = torch.from_numpy(img_jpg).permute(2,0,1).float().cuda().unsqueeze(0)
        depth = cv2.imread(f'assets/deepdeform/train/seq070/depth/000{img_num}.png', cv2.IMREAD_UNCHANGED)
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

    


