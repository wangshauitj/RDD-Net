import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
# import pyflow
from skimage import img_as_float
from random import randrange
import os.path
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def is_rar_file(filename):
    return any(filename.endswith(extension) for extension in [".rar", ".zip"])


def load_img(filepath, nFrames, scale, other_dataset):
    seq = [i for i in range(1, nFrames)]
    # random.shuffle(seq) #if random sequence
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)

        char_len = len(filepath)
        neigbor = []

        for i in seq:
            index = int(filepath[char_len - 7:char_len - 4]) - i
            file_name = filepath[0:char_len - 7] + '{0:03d}'.format(index) + '.png'

            if os.path.exists(file_name):
                temp = modcrop(Image.open(filepath[0:char_len - 7] + '{0:03d}'.format(index) + '.png').convert('RGB'),
                               scale).resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame is not exist')
                temp = input
                neigbor.append(temp)
    else:
        target = modcrop(Image.open(join(filepath, 'im' + str(nFrames) + '.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
        neigbor = [modcrop(Image.open(filepath + '/im' + str(j) + '.png').convert('RGB'), scale).resize(
            (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC) for j in reversed(seq)]

    return target, input, neigbor


def load_img_future(filepath, nFrames, scale, other_dataset):
    tt = int(nFrames / 2)
    if other_dataset:
        target = modcrop(Image.open(filepath).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)

        char_len = len(filepath)
        neigbor = []
        if nFrames % 2 == 0:
            seq = [x for x in range(-tt, tt) if x != 0]  # or seq = [x for x in range(-tt+1,tt+1) if x!=0]
        else:
            seq = [x for x in range(-tt, tt + 1) if x != 0]
        # random.shuffle(seq) #if random sequence
        for i in seq:
            index1 = int(filepath[char_len - 7:char_len - 4]) + i
            file_name1 = filepath[0:char_len - 7] + '{0:03d}'.format(index1) + '.png'

            if os.path.exists(file_name1):
                temp = modcrop(Image.open(file_name1).convert('RGB'), scale).resize(
                    (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
                neigbor.append(temp)
            else:
                print('neigbor frame- is not exist')
                temp = input
                neigbor.append(temp)

    else:
        target = modcrop(Image.open(join(filepath, 'im4.png')).convert('RGB'), scale)
        input = target.resize((int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC)
        neigbor = []
        seq = [x for x in range(4 - tt, 5 + tt) if x != 4]
        # random.shuffle(seq) #if random sequence
        for j in seq:
            neigbor.append(modcrop(Image.open(filepath + '/im' + str(j) + '.png').convert('RGB'), scale).resize(
                (int(target.size[0] / scale), int(target.size[1] / scale)), Image.BICUBIC))
    return target, input, neigbor


def load_img_future_de_rain(filepath, nFrames, img_id):
    tt = int(nFrames / 2)
    img_id = img_id + tt
    target, input, neigbor = None, None, None
    if filepath.split('/')[3].split('-')[0] == 'SPAC':
        targetPath = os.path.dirname(filepath) + '/' + filepath.split('/')[5].split('_')[0] + '_GT'
        target = Image.open(targetPath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
        input = Image.open(filepath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
        # print(targetPath + '/' + str(img_id).zfill(5) + '.jpg')
        # print(filepath + '/' + str(img_id).zfill(5) + '.jpg')
        # exit()
        neigbor = []
        seq = [x for x in range(img_id - tt, img_id + 1 + tt) if x != img_id]
        for j in seq:
            neigbor.append(Image.open(filepath + '/' + str(j).zfill(5) + '.jpg').convert('RGB'))
            # print(filepath + '/' + str(j).zfill(5) + '.jpg')
        # print(filepath + '/' + str(img_id).zfill(5) + '.jpg')
        # print(targetPath)
        # exit()
    elif filepath.split('/')[3].split('_')[0] == 'frames':
        target = Image.open(filepath + '/' + 'gtc-' + str(img_id) + '.jpg').convert('RGB')
        input = Image.open(filepath + '/' + 'rfc-' + str(img_id) + '.jpg').convert('RGB')  # .resize((888, 496))
        neigbor = []
        # print(filepath + '/' + 'gtc-5.jpg')
        # print(filepath + '/' + 'rfc-5.jpg')
        seq = [x for x in range(img_id - tt, img_id + 1 + tt) if x != img_id]
        for j in seq:
            neigbor.append(Image.open(filepath + '/' + 'rfc-' + str(j) + '.jpg').convert('RGB'))
            # print(filepath + '/' + 'rfc-' + str(j) + '.jpg')
        if target.size == (889, 500):
            target = target.crop((0, 0, 888, 496))
            input = input.crop((0, 0, 888, 496))
            for j in range(len(neigbor)):
                neigbor[j] = neigbor[j].crop((0, 0, 888, 496))
    elif filepath.split('/')[3] == 'rain_real':
        # print(filepath + '/' + str(img_id).zfill(5) + '.jpg')
        # exit()
        target = None
        input = Image.open(filepath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
        input = input.resize((int(1280 * 0.8), int(720 * 0.8)), Image.ANTIALIAS)
        # input = input.resize((832, 512), Image.ANTIALIAS)
        neigbor = []
        seq = [x for x in range(img_id - tt, img_id + 1 + tt) if x != img_id]
        for j in seq:
            tmp_nei = Image.open(filepath + '/' + str(j).zfill(5) + '.jpg').convert('RGB')
            tmp_nei = tmp_nei.resize((int(1280 * 0.8), int(720 * 0.8)), Image.ANTIALIAS)
            # tmp_nei = tmp_nei.resize((832, 512), Image.ANTIALIAS)
            neigbor.append(tmp_nei)

        # exit()
    # if target is None:
    #     print('read false')
    #     exit()
    return target, input, neigbor


def load_img_future_de_rain_flow(filepath, nFrames, img_id):
    tt = int(nFrames / 2)
    img_id = img_id + tt
    target, input, neigbor, tar_rain = None, None, None, None
    # if filepath.split('/')[3].split('_')[1] == 'Flow':
    num_dir = filepath.split('/')[3].split('_')[0] + '_GT'  # t1_GT  a1_GT
    if 't' in num_dir:
        targetPath = 'data/RainMotion/Test_GT/' + num_dir
    else:
        targetPath = 'data/RainMotion/Test_GT/' + num_dir
    target = Image.open(targetPath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    input = Image.open(filepath + '/' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    tar_rain = Image.open(filepath + '/rs-' + str(img_id).zfill(5) + '.jpg').convert('RGB')
    neigbor = []
    seq = [x for x in range(img_id - tt, img_id + 1 + tt) if x != img_id]
    # seq = [img_id]
    for j in seq:
        neigbor.append(Image.open(filepath + '/' + str(j).zfill(5) + '.jpg').convert('RGB'))
    a = filepath.split('/')[-1].split('_')[0][1]
    b = filepath.split('/')[-1].split('_')[2][1]
    base_path = filepath + '/motion_{}_{}.txt'.format(a, b)
    motion = np.loadtxt(base_path, delimiter=',')
    tar_motion = np.ones([128, 128, 2])
    # tar_motion = np.ones([480, 640, 2])
    tar_motion[:, :, 0] = tar_motion[:, :, 0] * motion[img_id - 1][0]
    # tar_motion[:, :, 1] = tar_motion[:, :, 1] * motion[img_id - 1][1]
    tar_motion[:, :, 1] = tar_motion[:, :, 1] * motion[img_id - 1][2]

    # exit()
    if target is None:
        print('read false')
        exit()
    return target, input, neigbor, tar_rain, tar_motion


def load_img_future_de_rain_test(filepath, nFrames, img_id):
    tt = int(nFrames / 2)
    img_id = img_id + nFrames / 2  # ---------------
    target = Image.open(filepath + '/' + 'gtc-' + str(img_id) + '.jpg').convert('RGB')
    input = Image.open(filepath + '/' + 'rfc-' + str(img_id) + '.jpg').convert('RGB')  # .resize((888, 496))
    neigbor = []

    seq = [x for x in range(img_id - tt, img_id + 1 + tt) if x != img_id]
    for j in seq:
        neigbor.append(Image.open(filepath + '/' + 'rfc-' + str(j) + '.jpg').convert('RGB'))
        # print(filepath + '/' + 'rfc-' + str(j) + '.jpg')
    if target.size == (889, 500):
        target = target.crop((0, 0, 888, 496))
        input = input.crop((0, 0, 888, 496))
        for j in range(len(neigbor)):
            neigbor[j] = neigbor[j].crop((0, 0, 888, 496))
    return target, input, neigbor


def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo);
    iw = iw - (iw % modulo);
    img = img.crop((0, 0, ih, iw))
    return img


def get_patch(img_in, img_tar, img_nn, img_rain, patch_size, scale, nFrames, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))  # [:, iy:iy + ip, ix:ix + ip]
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))  # [:, ty:ty + tp, tx:tx + tp]
    img_nn = [j.crop((iy, ix, iy + ip, ix + ip)) for j in img_nn]  # [:, iy:iy + ip, ix:ix + ip]
    img_rain = img_rain.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar, img_nn, img_rain, info_patch


def augment(img_in, img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = [ImageOps.flip(j) for j in img_nn]
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = [ImageOps.mirror(j) for j in img_nn]
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = [j.rotate(180) for j in img_nn]
            info_aug['trans'] = True

    return img_in, img_tar, img_nn, info_aug


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


class DeRainDatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, data_augmentation, file_list, other_dataset, patch_size,
                 future_frame, transform=None):
        super(DeRainDatasetFromFolder, self).__init__()
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.data_augmentation = data_augmentation
        self.other_dataset = other_dataset
        self.patch_size = patch_size
        self.future_frame = future_frame

        alist = os.listdir(image_dir)  # image_dir : /dataset/frames_heavy_train_JPEG
        self.image_num = 0
        self.index_compute = []

        self.image_filenames = [join(image_dir, x) for x in alist]  #

        image_num = 0
        for i in range(len(self.image_filenames)):
            image_list = os.listdir(self.image_filenames[i])
            for img in image_list:
                if img.endswith('jpg') and 'rs' not in img:
                    image_num += 1
            # image_num += len(os.listdir(self.image_filenames[i]))
            image_num = image_num - self.nFrames + 1
            self.index_compute.append(image_num)
        self.image_num = self.index_compute[-1]

    def __getitem__(self, index):
        file_id = 0
        index = index + 1
        for i in range(len(self.index_compute)):
            if self.index_compute[i] >= index:
                file_id = i
                break
        img_id = index if file_id == 0 else index - int(self.index_compute[file_id - 1])

        if self.future_frame:
            target, input, neigbor, tar_rain, tar_motion = load_img_future_de_rain_flow(
                self.image_filenames[file_id], self.nFrames, img_id)

        if self.patch_size != 0:
            input, target, neigbor, tar_rain, _ = get_patch(input, target, neigbor, tar_rain, self.patch_size, 1,
                                                            self.nFrames)

        if self.data_augmentation:
            input, target, neigbor, _ = augment(input, target, neigbor)

        flow = [[] for j in neigbor]

        bicubic = rescale_img(input, self.upscale_factor)

        if self.transform:
            target = self.transform(target)
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]
            tar_rain = self.transform(tar_rain)
            tar_motion = self.transform(tar_motion)

        # print(input.shape)
        return input, target, neigbor, flow, bicubic, tar_rain, tar_motion

    def __len__(self):
        return self.image_num


class DeRainDatasetFromFolderTest(data.Dataset):
    def __init__(self, image_dir, nFrames, upscale_factor, file_list, other_dataset, future_frame, transform=None):
        super(DeRainDatasetFromFolderTest, self).__init__()
        self.nFrames = nFrames
        self.upscale_factor = upscale_factor
        self.transform = transform
        self.other_dataset = other_dataset
        self.future_frame = future_frame

        self.image_num = 0
        self.index_compute = []

        self.image_filenames = [join(image_dir, x) for x in os.listdir(image_dir)]  #
        self.image_filenames = sorted(self.image_filenames)
        image_num = 0
        for i in range(len(self.image_filenames)):
            image_list = os.listdir(self.image_filenames[i])
            for img in image_list:
                if img.endswith('jpg') and 'rs' not in img:
                    image_num += 1
            # image_num += len(os.listdir(self.image_filenames[i]))
            image_num = image_num - self.nFrames + 1
            self.index_compute.append(image_num)
        self.image_num = self.index_compute[-1]

    def __getitem__(self, index):
        file_id = 0
        index = index + 1
        for i in range(len(self.index_compute)):
            if self.index_compute[i] >= index:
                file_id = i
                break
        img_id = index if file_id == 0 else index - int(self.index_compute[file_id - 1])
        if not os.path.exists(
                'Results/Flow_backbone_de_eq_attn/' + str(int(file_id / 5) + 1) + '_' + str(int(file_id % 5) + 1)):
            os.mkdir(
                'Results/Flow_backbone_de_eq_attn/' + str(int(file_id / 5) + 1) + '_' + str(int(file_id % 5) + 1))

        if self.future_frame:
            target, input, neigbor, tar_rain, tar_motion = load_img_future_de_rain_flow(
                self.image_filenames[file_id], self.nFrames, img_id)

        flow = [[] for j in neigbor]

        bicubic = rescale_img(input, self.upscale_factor)

        file = str(int(file_id / 5) + 1) + '_' + str(int(file_id % 5) + 1) + '/recover' + str(
            img_id + int(self.nFrames / 2)) + '.png'

        if self.transform:
            # target = self.transform(target)
            target = []
            input = self.transform(input)
            bicubic = self.transform(bicubic)
            neigbor = [self.transform(j) for j in neigbor]

            tar_rain = self.transform(tar_rain)
            tar_motion = self.transform(tar_motion)

        return input, target, neigbor, flow, bicubic, file, tar_rain, tar_motion

    def __len__(self):
        return self.image_num  # ---------------
