import torch
import torch.nn as nn
import torch.nn.functional as F

# lietorch for tangent space backpropogation
from lietorch import SE3

from .blocks.extractor import BasicEncoder
from .blocks.resnet import FPN
from .blocks.corr import CorrBlock
from .blocks.gru import ConvGRU
from .sampler_ops import bilinear_sampler, depth_sampler

from . import projective_ops as pops
from . import se3_field

import point_cloud_utils as pcu

GRAD_CLIP = .01

class GradClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_x):
        o = torch.zeros_like(grad_x)
        grad_x = torch.where(grad_x.abs()>GRAD_CLIP, o, grad_x)
        grad_x = torch.where(torch.isnan(grad_x), o, grad_x)
        return grad_x

class GradientClip(nn.Module):
    def __init__(self):
        super(GradientClip, self).__init__()

    def forward(self, x):
        return GradClip.apply(x)


class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.gru = ConvGRU(hidden_dim)

        self.corr_enc = nn.Sequential(
            nn.Conv2d(196, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3*128, 1, padding=0))

        self.flow_enc = nn.Sequential(
            nn.Conv2d(9, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3*128, 1, padding=0))

        self.ae = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 32, 1, padding=0),
            GradientClip())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            GradientClip())

        self.weight = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1, padding=0),
            nn.Sigmoid(),
            GradientClip())

        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0),
            GradientClip())


    def forward(self, net, inp, corr, flow, twist, dz, upsample=True):
        motion_info = torch.cat([flow, 10*dz, 10*twist], dim=-1)
        motion_info = motion_info.clamp(-50.0, 50.0).permute(0,3,1,2)

        mot = self.flow_enc(motion_info)
        cor = self.corr_enc(corr)

        # net is the hidden state
        net = self.gru(net, inp, cor, mot)

        ae = self.ae(net)
        mask = self.mask(net)
        delta = self.delta(net)
        weight = self.weight(net)

        # Return: hidden state (net), _, affinity embedding map V (ae), revision map rx,ry,rz (delta), and confidence maps wx,wy,wz (weight)
        return net, mask, ae, delta, weight


class RAFT3D(nn.Module):
    def __init__(self, args):
        super(RAFT3D, self).__init__()

        self.args = args
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        self.corr_levels = 4
        self.corr_radius = 3

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = FPN(output_dim=hdim+3*hdim)
        self.update_block = BasicUpdateBlock(args, hidden_dim=hdim)

    def initializer(self, image1):
        """ Initialize coords and transformation maps """

        batch_size, ch, ht, wd = image1.shape
        device = image1.device

        y0, x0 = torch.meshgrid(torch.arange(ht//8), torch.arange(wd//8))
        coords0 = torch.stack([x0, y0], dim=-1).float()
        coords0 = coords0[None].repeat(batch_size, 1, 1, 1).to(device)

        Ts = SE3.Identity(batch_size, ht//8, wd//8, device=device)
        return Ts, coords0
        
    def features_and_correlation(self, image1, image2):
        # extract features and build correlation volume
        fmap1, fmap2 = self.fnet([image1, image2])

        corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # extract context features using Resnet50
        net_inp = self.cnet(image1)
        net, inp = net_inp.split([128, 128*3], dim=1)

        net = torch.tanh(net)
        inp = torch.relu(inp)

        return corr_fn, net, inp

    def forward(self, image1, image2, depth1, depth2, intrinsics, iters=12, train_mode=False):
        """ Estimate optical flow between pair of frames """

        Ts, coords0 = self.initializer(image1)
        corr_fn, net, inp = self.features_and_correlation(image1, image2)

        # intrinsics and depth at 1/8 resolution
        intrinsics_r8 = intrinsics / 8.0
        depth1_r8 = depth1[:,3::8,3::8]
        depth2_r8 = depth2[:,3::8,3::8]

        flow_est_list = []
        flow_rev_list = []

        print("Iteration metrics:")
        print("              Chamfer:       | Depth residual: ")
        print("              --------------- -----------------")
        for itr in range(iters):
            Ts = Ts.detach()

            # Transformed prediction for x,y,d
            # This is really x,y,d, with x,y being image plane space (or u,v) and d = 1 / z
            # so z here is zinv because it is built into pops.projective_transform()
            coords1_xyz, _ = pops.projective_transform(Ts, depth1_r8, intrinsics_r8)
            
            # Predicted 2D correspondences x,y and predicted depths
            # x' in section 3.2 of the paper is coords1
            # d' in section 3.2 of the paper is zinv_proj
            coords1, zinv_proj = coords1_xyz.split([2,1], dim=-1)
            # The depths from frame 2 corresponding to the predicted x,y correspondences.
            # We can compare the directly predicted depths with the depth associated with predicted x,y to get a partial data term. For example,
            # if the prediction for x,y (x') and d (d') is exactly correct, the depth in frame 2 (d-') associated with x' will exactly match d'.
            # The data term (d' - d-') tells us how close we are in general but doesn't tell us how close in any dimension. That is probably okay.
            # d-' ("d bar prime") in section 3.2 of the paper is zinv
            zinv, _ = depth_sampler(1.0/depth2_r8, coords1)

            # Correlation features extracted from the neighborhood of the correlation volume
            # L_C(x') in section 3.2 of the paper is corr
            corr = corr_fn(coords1.permute(0,3,1,2).contiguous())
            # Predicted 2D optical flow
            # (x' - x) in section 3.2 of the paper is flow
            flow = coords1 - coords0

            # Depth residual data term (note: in the paper they have d' - d-' but here it is reversed. Doesn't make any difference.)
            # (d-' - d') in section 3.2 of the paper is dz
            dz = zinv.unsqueeze(-1) - zinv_proj
            # Twist field log_SE3(T)
            twist = Ts.log()

            # Add Chamfer distance data term here for comparison:
            X1 = pops.inv_project(depth1_r8, intrinsics_r8)
            X2 = pops.inv_project(depth2_r8, intrinsics_r8)
            X2_pred  = Ts * X1
            X2 = X2.squeeze()
            flat_dims = X2.shape[0] * X2.shape[1], 3
            X2 = X2.cpu().numpy().reshape(*flat_dims)
            X2_pred = X2_pred.cpu().numpy().reshape(*flat_dims)
            chamf = pcu.chamfer_distance(X2, X2_pred)
            print(f"Iteration {itr+1:2}: {chamf:4E}   | {torch.sum(abs(dz)):4E}")

            # GRU inputs: net->hidden state (initialized from context resnet), inp->context features (static features from context resnet), corr->correlation features, flow/dz/twist->predicted motion features
            # GRU outputs: net->hidden state, mask->? (used for upsampling back to full resolution), ae->affinities, delta->revision map, weight->weights (confidences)
            net, mask, ae, delta, weight = \
                self.update_block(net, inp, corr, flow, dz, twist)

            # Predicted x,y,d updated by revision map
            # target is r+pi(TX) from equation 16 in section 3.3
            target = coords1_xyz.permute(0,3,1,2) + delta
            target = target.contiguous()

            # step_inplace performs the following equations: 16, 15, 7-10, and T'=exp(deltax-hat)*T
            # Gauss-Newton step
            # Ts = se3_field.step(Ts, ae, target, weight, depth1_r8, intrinsics_r8)
            Ts = se3_field.step_inplace(Ts, ae, target, weight, depth1_r8, intrinsics_r8)

            if train_mode:
                flow2d_rev = target.permute(0,2,3,1)[...,:2] - coords0
                flow2d_rev = se3_field.cvx_upsample(8 * flow2d_rev, mask)

                Ts_up = se3_field.upsample_se3(Ts, mask)
                flow2d_est, flow3d_est, valid = pops.induced_flow(Ts_up, depth1, intrinsics)

                flow_est_list.append(flow2d_est)
                flow_rev_list.append(flow2d_rev)

        if train_mode:
            return flow_est_list, flow_rev_list

        Ts_up = se3_field.upsample_se3(Ts, mask)
        return Ts_up

