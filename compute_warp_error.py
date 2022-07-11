#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:wpaifang
# datetime:2022/1/29 16:58
# software: PyCharm
# function:
import glob
import subprocess
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import os
import sys
from dataset import get_flow
from skimage import img_as_float
from PIL import ImageFile
import cv2
import math

ImageFile.LOAD_TRUNCATED_IMAGES = True
FLO_TAG = 202021.25


def optical_flow_warping(x, flo, pad_mode="zeros"):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    pad_mode (optional): ref to https://pytorch.org/docs/stable/nn.functional.html#grid-sample
        "zeros": use 0 for out-of-bound grid locations,
        "border": use border values for out-of-bound grid locations
    """
    # print(x)
    # print(x.shape)
    # exit()
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    vgrid = grid + flo  # warp后，新图每个像素对应原图的位置

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode=pad_mode)

    mask = torch.ones(x.size())
    mask = F.grid_sample(mask, vgrid)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    return output * mask


# def get_flow(im1, im2):
# #     im1 = np.array(im1)
# #     im2 = np.array(im2)
# #     im1 = im1.astype(float) / 255.
# #     im2 = im2.astype(float) / 255.
# #
# #     # Flow Options:
# #     alpha = 0.012
# #     ratio = 0.75
# #     minWidth = 20
# #     nOuterFPIterations = 7
# #     nInnerFPIterations = 1
# #     nSORIterations = 30
# #     colType = 0  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
# #
# #     u, v, im2W = pyflow.coarse2fine_flow(im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
# #                                          nSORIterations, colType)
# #     flow = np.concatenate((u[..., None], v[..., None]), axis=2)
# #     # flow = rescale_flow(flow,0,1)
# #     return flow


def read_flo(filename):
    with open(filename, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)

        if tag != FLO_TAG:
            sys.exit('Wrong tag. Invalid .flo file %s' % filename)
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            # print 'Reading %d x %d flo file' % (w, h)

            data = np.fromfile(f, np.float32, count=2 * w * h)

            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (h, w, 2))

    return flow


def save_flo(flow, filename):
    with open(filename, 'wb') as f:
        tag = np.array([FLO_TAG], dtype=np.float32)

        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        tag.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)


def read_img(filename, grayscale=0):
    ## read image and convert to RGB in [0, 1]

    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise Exception("Image %s does not exist" % filename)

        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(filename)

        if img is None:
            raise Exception("Image %s does not exist" % filename)

        img = img[:, :, ::-1]  ## BGR to RGB

    img = np.float32(img) / 255.0

    return img


def save_img(img, filename):
    print("Save %s" % filename)

    if img.ndim == 3:
        img = img[:, :, ::-1]  ### RGB to BGR

    ## clip to [0, 1]
    img = np.clip(img, 0, 1)

    ## quantize to [0, 255]
    img = np.uint8(img * 255.0)

    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def resize_flow(flow, W_out=0, H_out=0, scale=0):
    if W_out == 0 and H_out == 0 and scale == 0:
        raise Exception("(W_out, H_out) or scale should be non-zero")

    H_in = flow.shape[0]
    W_in = flow.shape[1]

    if scale == 0:
        y_scale = float(H_out) / H_in
        x_scale = float(W_out) / W_in
    else:
        y_scale = scale
        x_scale = scale

    flow_out = cv2.resize(flow, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR)

    flow_out[:, :, 0] = flow_out[:, :, 0] * x_scale
    flow_out[:, :, 1] = flow_out[:, :, 1] * y_scale

    return flow_out


def img2tensor(img):
    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))

    return img_t


def tensor2img(img_t):
    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img


def run_cmd(cmd):
    print(cmd)
    subprocess.call(cmd, shell=True)


def compute_flow_magnitude(flow):
    flow_mag = flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2

    return flow_mag


def compute_flow_gradients(flow):
    H = flow.shape[0]
    W = flow.shape[1]

    flow_x_du = np.zeros((H, W))
    flow_x_dv = np.zeros((H, W))
    flow_y_du = np.zeros((H, W))
    flow_y_dv = np.zeros((H, W))

    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
    flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
    flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
    flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv


def detect_occlusion(fw_flow, bw_flow):
    ## fw-flow: img1 => img2
    ## bw-flow: img2 => img1

    with torch.no_grad():
        ## convert to tensor
        fw_flow_t = img2tensor(fw_flow)
        bw_flow_t = img2tensor(bw_flow)

        ## warp fw-flow to img2
        fw_flow_w = optical_flow_warping(fw_flow_t, bw_flow_t)

        ## convert to numpy array
        fw_flow_w = tensor2img(fw_flow_w)

    ## occlusion
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5

    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2

    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask
    mask = np.logical_or(mask1, mask2)
    occlusion = np.zeros((fw_flow.shape[0], fw_flow.shape[1]))
    occlusion[mask == 1] = 1

    return occlusion


def pre_flow(method):
    save_dir = '/dataset/ws/cvpr22_rebuttal/flow'
    list_filename = ''
    video_list = []
    if method == 'RDD':
        list_filename = '/dataset/ws/Results/Flow_backbone_de_eq_attn'
        video_list = os.listdir(list_filename)
    elif method == 'rain':
        list_filename = '/dataset/ws/Rain_Flow_2'
        video_list = os.listdir(list_filename)
    elif method == 'gt':
        list_filename = '/dataset/ws/SPAC-SupplementaryMaterials-master/Dataset_Testing_Synthetic'
        video_list = ['a1_GT', 'a2_GT', 'a3_GT', 'a4_GT', 'b1_GT', 'b2_GT', 'b3_GT', 'b4_GT']
    elif method == 'FCDN':
        list_filename = '/dataset/ws/Results/FCDN'
        video_list = os.listdir(list_filename)
    elif method == 'SLDNet':
        list_filename = '/dataset/ws/Results/SLD'
        for i in range(1, 5):
            for j in range(1, 6):
                video_list.append('a{}_Rain_0{}'.format(i, j))
                video_list.append('b{}_Rain_0{}'.format(i, j))
    elif method == 'Fast':
        list_filename = '/dataset/ws/Results/FastDerain'
        for i in range(1, 5):
            for j in range(1, 6):
                video_list.append('a{}_0{}'.format(i, j))
                video_list.append('b{}_0{}'.format(i, j))
    elif method == 'MSCSC':
        list_filename = '/dataset/ws/Results/MSCSC'
        for i in range(1, 5):
            for j in range(1, 6):
                video_list.append('a{}_0{}'.format(i, j))
                video_list.append('b{}_0{}'.format(i, j))
    elif method == 'DCSFN':
        list_filename = '/dataset/ws/Results/DCSFN'
        for i in range(1, 5):
            for j in range(1, 6):
                video_list.append('a{}_0{}'.format(i, j))
                video_list.append('b{}_0{}'.format(i, j))
    elif method == 'MPRNet':
        list_filename = '/dataset/ws/Results/MPRNet'
        for i in range(1, 5):
            for j in range(1, 6):
                video_list.append('a{}_0{}'.format(i, j))
                video_list.append('b{}_0{}'.format(i, j))
    elif method == 'heavyrain':
        list_filename = '/dataset/ws/frames_heavy_test_JPEG'
        video_list = os.listdir(list_filename)
    elif method == 'heavygt':
        list_filename = '/dataset/ws/frames_heavy_test_JPEG'
        video_list = os.listdir(list_filename)
    elif method == 'lightrain':
        list_filename = '/dataset/ws/frames_light_test_JPEG'
        video_list = os.listdir(list_filename)
    elif method == 'lightgt':
        list_filename = '/dataset/ws/frames_light_test_JPEG'
        video_list = os.listdir(list_filename)
    elif method == 'nturain':
        list_filename = '/dataset/ws/SPAC-SupplementaryMaterials-master/Dataset_Testing_Synthetic'
        for i in range(1, 5):
            video_list.append('a{}_Rain'.format(i))
            video_list.append('b{}_Rain'.format(i))
    # list_filename = '/dataset/ws/cvpr22_rebuttal/warperror'
    # video_list = os.listdir(list_filename)
    # # video_list = [video_list[5], video_list[8]] 'a1_01_rain', , 'a1_GT'
    # print(video_list)
    # # video_list = ['DCSFN',  'MPRNet'] # png a1_01
    # # video_list = ['FCDN', 'RDD']  # png
    # video_list = ['RDD']  # png
    # # video_list = ['MSCSC', 'Fast'] # jpg a1_01
    # # exit()
    for video in video_list:
        frame_dir = os.path.join(list_filename, video)
        fw_flow_dir = os.path.join(save_dir, "fw_flow", method, video)
        if not os.path.isdir(fw_flow_dir):
            os.makedirs(fw_flow_dir)

        fw_occ_dir = os.path.join(save_dir, "fw_occlusion", method, video)
        if not os.path.isdir(fw_occ_dir):
            os.makedirs(fw_occ_dir)

        # frame_list = glob.glob(os.path.join(frame_dir, "*.png"))
        frame_list = []
        if method == 'RDD':
            frame_list = ['recover%d.png' % t for t in range(4, 18)]
        elif method == 'rain':
            frame_list = ['%05d.jpg' % t for t in range(4, 18)]
        elif method == 'gt':
            frame_list = ['%05d.jpg' % t for t in range(4, 18)]
        elif method == 'FCDN':
            frame_list = ['%05d.png' % t for t in range(4, 18)]
        elif method == 'SLDNet':
            frame_list = ['%05d_res.png' % t for t in range(4, 17)]
        elif method == 'Fast':
            frame_list = ['%05d.jpg' % t for t in range(4, 18)]
        elif method == 'MSCSC':
            frame_list = ['%05d.jpg' % t for t in range(4, 18)]
        elif method == 'DCSFN':
            frame_list = ['%05d.png' % t for t in range(4, 18)]
        elif method == 'MPRNet':
            frame_list = ['%05d.png' % t for t in range(4, 18)]
        elif method == 'heavyrain':
            frame_list = ['rfc-%d.jpg' % t for t in range(4, 18)]
        elif method == 'heavygt':
            frame_list = ['gtc-%d.jpg' % t for t in range(4, 18)]
        elif method == 'lightrain':
            frame_list = ['rfc-%d.jpg' % t for t in range(4, 18)]
        elif method == 'lightgt':
            frame_list = ['gtc-%d.jpg' % t for t in range(4, 18)]
        elif method == 'nturain':
            frame_list = ['%05d.jpg' % t for t in range(4, 18)]

        for t in range(len(frame_list) - 1):

            print("Compute flow on %s frame %d" % (video, t + 4))

            ### load input images
            # print("%05d.jpg" % (t + 4))
            # exit()
            if method in ['Fast', 'MSCSC', 'DCSFN', 'MPRNet']:
                filename = os.path.join(list_filename, video + '_' + frame_list[t])
                img1 = read_img(filename)
                filename = os.path.join(list_filename, video + '_' + frame_list[t + 1])
                img2 = read_img(filename)
            else:
                img1 = read_img(os.path.join(frame_dir, frame_list[t]))
                img2 = read_img(os.path.join(frame_dir, frame_list[t + 1]))

            ### resize image
            size_multiplier = 64
            H_orig = img1.shape[0]
            W_orig = img1.shape[1]

            H_sc = int(math.ceil(float(H_orig) / size_multiplier) * size_multiplier)
            W_sc = int(math.ceil(float(W_orig) / size_multiplier) * size_multiplier)

            img1 = cv2.resize(img1, (W_sc, H_sc))
            img2 = cv2.resize(img2, (W_sc, H_sc))

            fw_flow = get_flow(img1, img2)
            bw_flow = get_flow(img2, img1)

            ### resize flow
            fw_flow = resize_flow(fw_flow, W_out=W_orig, H_out=H_orig)
            bw_flow = resize_flow(bw_flow, W_out=W_orig, H_out=H_orig)

            ### compute occlusion
            fw_occ = detect_occlusion(bw_flow, fw_flow)

            ### save flow
            output_flow_filename = os.path.join(fw_flow_dir, "%05d.flo" % (t + 4))
            if not os.path.exists(output_flow_filename):
                save_flo(fw_flow, output_flow_filename)

            ### save occlusion map
            output_occ_filename = os.path.join(fw_occ_dir, "%05d.png" % (t + 4))
            if not os.path.exists(output_occ_filename):
                save_img(fw_occ, output_occ_filename)


def warp_error(method):
    save_dir = '/dataset/ws/cvpr22_rebuttal/flow'
    ## print average if result already exists
    metric_filename = os.path.join('/home/wangshuai/project/RBPN/cvpr22', "WarpError.txt")
    list_filename = ''
    video_list = []
    if method == 'RDD':
        list_filename = '/dataset/ws/Results/Flow_backbone_de_eq_attn'
        video_list = os.listdir(list_filename)
    elif method == 'rain':
        list_filename = '/dataset/ws/Rain_Flow_2'
        video_list = os.listdir(list_filename)
    elif method == 'gt':
        list_filename = '/dataset/ws/SPAC-SupplementaryMaterials-master/Dataset_Testing_Synthetic'
        video_list = ['a1_GT', 'a2_GT', 'a3_GT', 'a4_GT', 'b1_GT', 'b2_GT', 'b3_GT', 'b4_GT']
    elif method == 'FCDN':
        list_filename = '/dataset/ws/Results/FCDN'
        video_list = os.listdir(list_filename)
    elif method == 'SLDNet':
        list_filename = '/dataset/ws/Results/SLD'
        for i in range(1, 5):
            for j in range(1, 6):
                video_list.append('a{}_Rain_0{}'.format(i, j))
                video_list.append('b{}_Rain_0{}'.format(i, j))
    elif method == 'Fast':
        list_filename = '/dataset/ws/Results/FastDerain'
        for i in range(1, 5):
            for j in range(1, 6):
                video_list.append('a{}_0{}'.format(i, j))
                video_list.append('b{}_0{}'.format(i, j))
    elif method == 'MSCSC':
        list_filename = '/dataset/ws/Results/MSCSC'
        for i in range(1, 5):
            for j in range(1, 6):
                video_list.append('a{}_0{}'.format(i, j))
                video_list.append('b{}_0{}'.format(i, j))
    elif method == 'DCSFN':
        list_filename = '/dataset/ws/Results/DCSFN'
        for i in range(1, 5):
            for j in range(1, 6):
                video_list.append('a{}_0{}'.format(i, j))
                video_list.append('b{}_0{}'.format(i, j))
    elif method == 'MPRNet':
        list_filename = '/dataset/ws/Results/MPRNet'
        for i in range(1, 5):
            for j in range(1, 6):
                video_list.append('a{}_0{}'.format(i, j))
                video_list.append('b{}_0{}'.format(i, j))
    elif method == 'heavyrain':
        list_filename = '/dataset/ws/frames_heavy_test_JPEG'
        video_list = os.listdir(list_filename)
    elif method == 'heavygt':
        list_filename = '/dataset/ws/frames_heavy_test_JPEG'
        video_list = os.listdir(list_filename)
    elif method == 'lightrain':
        list_filename = '/dataset/ws/frames_light_test_JPEG'
        video_list = os.listdir(list_filename)
    elif method == 'lightgt':
        list_filename = '/dataset/ws/frames_light_test_JPEG'
        video_list = os.listdir(list_filename)
    elif method == 'nturain':
        list_filename = '/dataset/ws/SPAC-SupplementaryMaterials-master/Dataset_Testing_Synthetic'
        for i in range(1, 5):
            video_list.append('a{}_Rain'.format(i))
            video_list.append('b{}_Rain'.format(i))
    # ### load video list
    # list_filename = '/dataset/ws/cvpr22_rebuttal/warperror'
    # video_list = os.listdir(list_filename)
    # # video_list = [video_list[8]] # , video_list[8]
    # # video_list = [video_list[5], video_list[8]] 'a1_01_rain', , 'a1_GT'
    # # video_list = ['DCSFN',  'MPRNet'] # png a1_01
    # # video_list = ['FCDN', 'RDD']  # png
    # video_list = ['RDD']  # png
    # # video_list = ['MSCSC', 'Fast'] # jpg a1_01
    # # exit()
    # print(video_list)

    ### start evaluation
    err_all = np.zeros(len(video_list))

    for v in range(len(video_list)):

        video = video_list[v]

        frame_dir = os.path.join(list_filename, video)
        flow_dir = os.path.join('/dataset/ws/cvpr22_rebuttal/flow', "fw_flow", method, video)
        occ_dir = os.path.join('/dataset/ws/cvpr22_rebuttal/flow', "fw_occlusion", method, video)
        frame_list = []
        if method == 'RDD':
            frame_list = ['recover%d.png' % t for t in range(4, 18)]
        elif method == 'rain':
            frame_list = ['%05d.jpg' % t for t in range(4, 18)]
        elif method == 'gt':
            frame_list = ['%05d.jpg' % t for t in range(4, 18)]
        elif method == 'FCDN':
            frame_list = ['%05d.png' % t for t in range(4, 18)]
        elif method == 'SLDNet':
            frame_list = ['%05d_res.png' % t for t in range(4, 17)]
        elif method == 'Fast':
            frame_list = ['%05d.jpg' % t for t in range(4, 18)]
        elif method == 'MSCSC':
            frame_list = ['%05d.jpg' % t for t in range(4, 18)]
        elif method == 'DCSFN':
            frame_list = ['%05d.png' % t for t in range(4, 18)]
        elif method == 'MPRNet':
            frame_list = ['%05d.png' % t for t in range(4, 18)]
        elif method == 'heavyrain':
            frame_list = ['rfc-%d.jpg' % t for t in range(4, 18)]
        elif method == 'heavygt':
            frame_list = ['gtc-%d.jpg' % t for t in range(4, 18)]
        elif method == 'lightrain':
            frame_list = ['rfc-%d.jpg' % t for t in range(4, 18)]
        elif method == 'lightgt':
            frame_list = ['gtc-%d.jpg' % t for t in range(4, 18)]
        elif method == 'nturain':
            frame_list = ['%05d.jpg' % t for t in range(4, 18)]

        err = 0
        for t in range(1, len(frame_list)):
            ### load input images
            if method in ['Fast', 'MSCSC', 'DCSFN', 'MPRNet']:
                filename = os.path.join(list_filename, video + '_' + frame_list[t - 1])
                img1 = read_img(filename)
                filename = os.path.join(list_filename, video + '_' + frame_list[t])
                img2 = read_img(filename)
            else:
                filename = os.path.join(frame_dir, frame_list[t - 1])
                img1 = read_img(filename)
                filename = os.path.join(frame_dir, frame_list[t])
                img2 = read_img(filename)

            print("Evaluate Warping Error on %s: video %d / %d, %s" % (
                video, v + 1, len(video_list), filename))

            ### load flow
            filename = os.path.join(flow_dir, "%05d.flo" % (t - 1 + 4))
            flow = read_flo(filename)

            # ### load occlusion mask
            filename = os.path.join(occ_dir, "%05d.png" % (t - 1 + 4))
            occ_mask = read_img(filename)
            noc_mask = 1 - occ_mask

            img2 = img2tensor(img2)
            flow = img2tensor(flow)

            warp_img2 = optical_flow_warping(img2, flow)
            warp_img2 = tensor2img(warp_img2)
            ## compute warping error
            diff = np.multiply(warp_img2 - img1, noc_mask)
            # diff = warp_img2 - img1

            N = np.sum(noc_mask)
            # print(N)
            if N == 0:
                N = diff.shape[0] * diff.shape[1] * diff.shape[2]

            # print(len([np.isnan(diff)]))
            diff = np.nan_to_num(diff)
            err += np.sum(np.square(diff)) / N

            # print(diff)
            # print(diff.shape)
            # print(np.sum(diff))
            # isnan = np.isnan(diff)
            # print(True in isnan)

            # err += np.sum(np.square(diff))

            # print(err)
            # exit()

        err_all[v] = err / (len(frame_list) - 1)

    print("\nAverage Warping Error = %f\n" % (err_all.mean()))
    print(err_all)
    err_all = np.append(err_all, err_all.mean())
    print("Save %s" % metric_filename)
    # np.savetxt(metric_filename, err_all, fmt="%f")
    with open(metric_filename, 'a') as f:
        f.write(method + '\n')
        # np.savetxt(f, err_all, fmt="%f")
        for item in err_all[:-1]:
            f.write(str(item))
            f.write('\t')
        f.write('\n')
        f.write(str(err_all[-1]) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='RDD')
    opt = parser.parse_args()
    # dataset = opt.dataset
    # mtd = 'RDD'
    pre_flow(opt.method)
    warp_error(opt.method)

