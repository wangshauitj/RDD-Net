# encoding: utf-8
from os.path import join

import numpy as np
import cv2
import math
# from skimage.measure import compare_ssim, compare_psnr
import skimage
from skimage.metrics import structural_similarity
from torch.quantization import quantize
import argparse
import os

"""
@version: ??
@author: wpaifang
@contact: wangshuai201909@tju.edu.cn
@software: PyCharm
@file: eval_psnr_ssim.py
@time: 2020/8/7 16:27
"""

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=49)
parser.add_argument('--dataset', type=str, default='light')


def psnr(target, ref):
    # 将图像格式转为float64
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    # 直接相减，求差值
    diff = ref_data - target_data
    # 按第三个通道顺序把三维矩阵拉平
    diff = diff.flatten('C')
    # 计算MSE值
    rmse = math.sqrt(np.mean(diff ** 2.))
    # 精度
    eps = np.finfo(np.float64).eps
    if (rmse == 0):
        rmse = eps
    return 20 * math.log10(255.0 / rmse)


def ssim(imageA, imageB):
    # 为确保图像能被转为灰度图
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)

    # 通道分离，注意顺序BGR不是RGB
    (B1, G1, R1) = cv2.split(imageA)
    (B2, G2, R2) = cv2.split(imageB)

    # convert the images to grayscale BGR2GRAY
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # 方法一
    (grayScore, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    # print("gray SSIM: {}".format(grayScore))

    # 方法二
    (score0, diffB) = structural_similarity(B1, B2, full=True)
    (score1, diffG) = structural_similarity(G1, G2, full=True)
    (score2, diffR) = structural_similarity(R1, R2, full=True)
    aveScore = (score0 + score1 + score2) / 3
    # print("BGR average SSIM: {}".format(aveScore))

    return grayScore, aveScore


def ssim_2(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


# for j in range(4, 29):
#     path1 = '/home/wangshuai/project/RBPN-PyTorch/Results/frames_light_test_JPEG/1/recover{}.png'.format(j)
#     path2 = '/dataset/frames_light_test_JPEG/1/gtc-{}.jpg'.format(j)
#     img1 = cv2.imread(path1)
#     img2 = cv2.imread(path2)
#     # s_psnr = psnr(img1, img2)
#     # print(s_psnr)
#     s_psnr = skimage.measure.compare_psnr(img1, img2, 255)
#     # _, s_ssim = ssim(img1, img2)
#     s_ssim = calculate_ssim(img1, img2)
#     # print(s_psnr)
#     # print(s_ssim)
#     print("{},{}".format(s_psnr, s_ssim))
# exit()


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def Tensor2np(tensor_list, rgb_range):
    def _Tensor2numpy(tensor, rgb_range):
        array = np.transpose(quantize(tensor, rgb_range).numpy(), (1, 2, 0)).astype(np.uint8)
        return array

    return [_Tensor2numpy(tensor, rgb_range) for tensor in tensor_list]


if __name__ == '__main__':
    opt = parser.parse_args()
    dataset = opt.dataset
    print(dataset)
    if dataset == 'light':
        print('1111111')
        alist = os.listdir('/dataset/ws/frames_light_test_JPEG/')  # image_dir : /dataset/frames_heavy_train_JPEG
        alist = sorted(int(i) for i in alist)
        gd_dir = [join('/dataset/ws/frames_light_test_JPEG/', str(x)) for x in alist]
        pred_dir = [join('/home/wangshuai/project/RBPN/Results/frames_light_test_JPEG/', str(x)) for x in alist]
        sum = 0.0
        s_psnr = 0.0
        s_ssim = 0.0
        for i in alist:
            # path1 = gd_dir[i - 1] + '/' + 'gtc-5.jpg'
            # path2 = pred_dir[i - 1] + '/' + 'RBPNF7.png'
            # # print(path2)
            # # print(path1)
            # img1 = cv2.imread(path1)
            # img2 = cv2.imread(path2)
            # # print(img2)
            # # exit()
            # # print(img1.shape)
            # if img1.shape[0] == 500 and img1.shape[1] == 889:
            #     img1 = img1[0:496, 0:888, :]
            # s_psnr += psnr(img1, img2)
            # _, tmp_ssim = ssim(img1, img2)
            # s_ssim += tmp_ssim
            # sum += 1.0
            for j in range(4, 29):
                path1 = gd_dir[i - 1] + '/' + 'gtc-' + str(j) + '.jpg'
                path2 = pred_dir[i - 1] + '/' + 'recover' + str(j) + '.png'
                img1 = cv2.imread(path1)
                img2 = cv2.imread(path2)
                # print(img1.shape)
                if img1.shape[0] == 500 and img1.shape[1] == 889:
                    img1 = img1[0:496, 0:888, :]
                # s_psnr += psnr(img1, img2)
                tmp_psnr = psnr(img1, img2)
                s_psnr += tmp_psnr
                _, tmp_ssim = ssim(img1, img2)
                s_ssim += tmp_ssim
                sum += 1.0
        #         print("{},{}".format(tmp_psnr, tmp_ssim))
        # exit()
        s_psnr = s_psnr / sum
        s_ssim = s_ssim / sum
        with open('Results/eval_record_light.txt', 'a') as f:
            f.write('eval_model_epoch_{}_PSNR_{}_SSIM_{}_sum_{}'.format(opt.epoch, s_psnr, s_ssim, sum))
            f.write('\n')
    elif dataset == 'heavy':
        print('222222')
        alist = os.listdir('/dataset/ws/frames_heavy_test_JPEG/')  # image_dir : /dataset/frames_heavy_train_JPEG
        alist = sorted(int(i) for i in alist)
        gd_dir = [join('/dataset/ws/frames_heavy_test_JPEG/', str(x)) for x in alist]
        pred_dir = [join('/home/wangshuai/project/RBPN/Results/frames_heavy_test_JPEG/', str(x)) for x in alist]
        sum = 0.0
        s_psnr = 0.0
        s_ssim = 0.0
        for i in alist:
            # path1 = gd_dir[i - 1] + '/' + 'gtc-5.jpg'
            # path2 = pred_dir[i - 1] + '/' + 'RBPNF9.png'
            # # print(path2)
            # # print(path1)
            # img1 = cv2.imread(path1)
            # img2 = cv2.imread(path2)
            # # print(img2)
            # # exit()
            # # print(img1.shape)
            # if img1.shape[0] == 500 and img1.shape[1] == 889:
            #     img1 = img1[0:496, 0:888, :]
            # s_psnr += psnr(img1, img2)
            # _, tmp_ssim = ssim(img1, img2)
            # s_ssim += tmp_ssim
            # sum += 1.0
            for j in range(5, 28):
                path1 = gd_dir[i - 1] + '/' + 'gtc-' + str(j) + '.jpg'
                path2 = pred_dir[i - 1] + '/' + 'recover' + str(j) + '.png'
                img1 = cv2.imread(path1)
                img2 = cv2.imread(path2)
                # print(img1.shape)
                if img1.shape[0] == 500 and img1.shape[1] == 889:
                    img1 = img1[0:496, 0:888, :]
                # print(path2)
                # print(path1)
                s_psnr += psnr(img1, img2)
                _, tmp_ssim = ssim(img1, img2)
                s_ssim += tmp_ssim
                sum += 1.0
        s_psnr = s_psnr / sum
        s_ssim = s_ssim / sum
        with open('Results/eval_record_heavy.txt', 'a') as f:
            f.write('eval_model_epoch_{}_PSNR_{}_SSIM_{}_sum_{}'.format(opt.epoch, s_psnr, s_ssim, sum))
            f.write('\n')
    elif dataset == 'NTURain':
        alist = os.listdir(
            '/dataset/ws/SPAC-SupplementaryMaterials-master/Dataset_Testing_Synthetic')  # image_dir : /dataset/frames_heavy_train_JPEG
        alist = [x[:-4] for x in alist if any(x.find(name) >= 0 and x.endswith('.rar') for name in ['GT'])]
        alist = sorted(alist)
        gd_dir = [join('/dataset/ws/SPAC-SupplementaryMaterials-master/Dataset_Testing_Synthetic/', str(x)) for x in
                  alist]
        pred_dir = [join('/home/wangshuai/project/RBPN/Results/NTURain/', str(x)) for x in alist]
        sum = 0.0
        s_psnr = 0.0
        s_ssim = 0.0
        for i in range(len(alist)):
            p_dir = os.path.dirname(pred_dir[i]) + '/' + str(i + 1)
            tot = len(os.listdir(p_dir))
            print(tot)
            nFrames = 7

            for j in range(1 + int(nFrames / 2), tot + 1 + int(nFrames / 2)):
                path1 = gd_dir[i] + '/' + str(j).zfill(5) + '.jpg'
                # print(path1)
                path2 = os.path.dirname(pred_dir[i]) + '/' + str(i + 1) + '/' + 'recover' + str(j) + '.png'
                # print(path2)
                img1 = cv2.imread(path1)
                img2 = cv2.imread(path2)
                # print(img1.shape)
                s_psnr += psnr(img1, img2)
                _, tmp_ssim = ssim(img1, img2)
                s_ssim += tmp_ssim
                # print("{},{}".format(s_psnr, s_ssim))
                # exit()
                sum += 1.0

            # print(s_psnr / sum)
            # print(s_ssim / sum)
            # exit()
        s_psnr = s_psnr / sum
        s_ssim = s_ssim / sum
        with open('Results/eval_record_NTURain.txt', 'a') as f:
            f.write('eval_model_epoch_{}_PSNR_{}_SSIM_{}_sum_{}'.format(opt.epoch, s_psnr, s_ssim, sum))
            f.write('\n')
    elif dataset == 'Flow':
        # alist = os.listdir('/dataset/ws/SPAC-SupplementaryMaterials-master/Dataset_Testing_Synthetic')
        # alist = [x[:-4] for x in alist if any(x.find(name) >= 0 and x.endswith('.rar') for name in ['GT'])]
        # alist = sorted(alist)
        # gd_dir = [join('/dataset/ws/SPAC-SupplementaryMaterials-master/Dataset_Testing_Synthetic/', str(x)) for x in
        #           alist]
        prex = ['a1', 'a2', 'a3', 'a4', 'b1', 'b2', 'b3', 'b4']
        gd_dir = 'data/RainMotion/Test_GT'
        # pred_dir = '/dataset/ws/Results/Flow_backbone_de_eq_attn/'
        pred_dir = 'Results/Flow_backbone_de_eq_attn/'
        sum = 0.0
        s_psnr = 0.0
        s_ssim = 0.0
        for i in range(1, 40):
            gd_path = gd_dir[int(i / 5)]
            # pred_path = '/dataset/ws/Results/Flow_backbone/' + str(int(i / 5) + 1) + '_' \
            #             + str(int(i % 5) + 1)
            # pred_path = '/dataset/ws/cvpr22_rebuttal/n5/' + str(int(i / 5) + 1) + '_' \
            #             + str(int(i % 5) + 1)
            # pred_path = '/dataset/ws/cvpr22_rebuttal/womotion/' + str(int(i / 5) + 1) + '_' \
            #             + str(int(i % 5) + 1)
            pred_path = 'Results/Flow_backbone_de_eq_attn/' + str(int(i / 5) + 1) + '_' \
                        + str(int(i % 5) + 1)
            gd_path = 'data/RainMotion/Test_GT/' + prex[int(i/5)] + '_GT'
            o_psnr = 0.0
            o_ssim = 0.0
            o_sum = 0.0
            for j in range(4, 18):
                path1 = gd_path + '/' + str(j).zfill(5) + '.jpg'
                # print(path1)
                path2 = pred_path + '/' + 'recover' + str(j) + '.png'
                # print(path2)
                # exit()
                img1 = cv2.imread(path1)
                img2 = cv2.imread(path2)
                tmp_psnr = psnr(img1, img2)

                # print(tmp_psnr)
                # print(path1)
                # print(path2)
                s_psnr += tmp_psnr
                _, tmp_ssim = ssim(img1, img2)
                s_ssim += tmp_ssim
                sum += 1.0

                o_psnr += tmp_psnr
                o_ssim += tmp_ssim
                o_sum += 1.0
                # print('tmp_psnr:{:.2f}'.format(tmp_psnr))
            print('{}_{},PSNR:{},SSIM:{}'.format(str(int(i / 5) + 1), str(int(i % 5) + 1), o_psnr / o_sum,
                                                 o_ssim / o_sum))
        s_psnr = s_psnr / sum
        s_ssim = s_ssim / sum
        print(s_psnr)
        print(s_ssim)
        # with open('Results/eval_record_Flow.txt', 'a') as f:
        #     f.write('eval_model_epoch_{}_PSNR_{}_SSIM_{}_sum_{}'.format(opt.epoch, s_psnr, s_ssim, sum))
        #     f.write('\n')
        with open('eval_results.txt', 'a') as f:
            f.write('eval_model_epoch_{}_PSNR_{}_SSIM_{}_sum_{}'.format(opt.epoch, s_psnr, s_ssim, sum))
            f.write('\n')
    # path1 = '/home/wangshuai/project/DBPN-Pytorch/Results/frames_heavy_test_JPEG/1/recover1.png'
    # path2 = '/dataset/frames_heavy_test_JPEG/1/gtc-1.jpg'
    # img1 = cv2.imread(path1)
    # img2 = cv2.imread(path2)
    # s_psnr = psnr(img1, img2)
    # print(s_psnr)
    # s_psnr = skimage.measure.compare_psnr(img1, img2, 255)
    # _, s_ssim = ssim(img1, img2)
    # print(s_psnr)
    # print(s_ssim)
