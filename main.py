from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from rdd import Net as RDD_Net
from data import get_training_set_derain
import pdb
import socket
import time

# Training settings batchSize和卡数相同
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=5, help='testing batch size')
parser.add_argument('--start_epoch', type=int, default=1, help='Starting epoch for continuing training')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=8, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=8, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='data/RainMotion/Test')  #
parser.add_argument('--file_list', type=str, default='sep_trainlist.txt')
parser.add_argument('--other_dataset', type=bool, default=False, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=128, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='RDD_Net')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--pretrained_sr', default='.pth',
                    help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=False)
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--prefix', default='_propose_RDDNet_wo_rmLoss', help='Location to save checkpoint models')

opt = parser.parse_args()
# gpus_list = range(0, opt.gpus)
# gpus_list = [0, 1, 2]
gpus_list = [0]
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)


def train(epoch):
    epoch_loss = 0
    e_r_loss = 0
    e_m_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, neigbor, flow, _, tar_rain, tar_motion = batch[0], batch[1], batch[2], batch[3], batch[4], \
                                                                batch[5], batch[6]
        if cuda:
            input = Variable(input).cuda(gpus_list[0])
            target = Variable(target).cuda(gpus_list[0])
            # bicubic = Variable(bicubic).cuda(gpus_list[0])
            neigbor = [Variable(j).cuda(gpus_list[0]) for j in neigbor]
            # flow = [Variable(j).cuda(gpus_list[0]).float() for j in flow]
            tar_rain = Variable(tar_rain).cuda(gpus_list[0])
            tar_motion = Variable(tar_motion).cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()
        # prediction, pred_rain, pred_motion = model(input, neigbor, flow)

        try:
            prediction, pred_rain, pred_motion = model(input, neigbor, flow)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception

        # prediction = model(input, neigbor, flow)
        # print(prediction.shape)
        # print(target.shape)
        # exit()

        # if opt.residual:
        #     prediction = prediction + bicubic

        loss = criterion(prediction, target)
        rain_loss = criterion(pred_rain, tar_rain)
        motion_loss = criterion(pred_motion, tar_motion)
        # print(loss)
        # print(rain_loss)
        # print(motion_loss)
        # # exit()
        t1 = time.time()
        epoch_loss += loss.data
        e_r_loss += rain_loss.data
        e_m_loss += motion_loss.data
        # tmp_loss = loss + 0.3 * rain_loss + 0.3 * motion_loss # not use
        tmp_loss = loss + e_r_loss + e_m_loss  # *******  use
        # loss.backward()
        # rain_loss.backward()
        # motion_loss.backward()
        tmp_loss.backward()  # ******* use
        optimizer.step()
        # torch.cuda.empty_cache()

        # print(
        #     "===> Epoch[{}]({}/{}): image_Loss: {:.4f}, rain_Loss: {:.4f}, motion_Loss: {:.4f} , || Timer: {:.4f} sec.".format(
        #         epoch, iteration, len(training_data_loader), loss.item(), rain_loss.item(), motion_loss.item(),
        #         (t1 - t0)))
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration,
                                                                                 len(training_data_loader), loss.item(),
                                                                                 (t1 - t0)))

    # print("===> Epoch {} Complete: Avg. Loss: {:.4f}, rain_Loss: {:.4f}, motion_Loss: {:.4f}".format
    #       (epoch, epoch_loss / len(training_data_loader), e_r_loss / len(training_data_loader),
    #        e_m_loss / len(training_data_loader)))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    # with open('lossn1.txt', 'a') as f:
    #     f.write(str(epoch) + ',' + str(epoch_loss / len(training_data_loader)))
    #     f.write(',' + str(e_r_loss / len(training_data_loader)))
    #     f.write(',' + str(e_m_loss / len(training_data_loader)))
    #     f.write('\n')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    model_out_path = opt.save_folder + str(
        opt.upscale_factor) + 'x_' + hostname + opt.model_type + opt.prefix + "_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set_derain(opt.data_dir, opt.nFrames, opt.upscale_factor, opt.data_augmentation, opt.file_list,
                                    opt.other_dataset, opt.patch_size, opt.future_frame)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)
if opt.model_type == 'RDD_Net':
    model = RDD_Net(num_channels=3, base_filter=256, feat=64, num_stages=3, n_resblock=2, nFrames=opt.nFrames,
                    scale_factor=opt.upscale_factor)

model = torch.nn.DataParallel(model, device_ids=gpus_list)
criterion = nn.L1Loss()

print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        print(model_name)
        # model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.80)
for epoch in range(opt.start_epoch, opt.nEpochs + 1):
    train(epoch)
    # scheduler.step()
    # test()

    # # learning rate is decayed by a factor of 10 every half of total epochs
    # if epoch % 40 == 0:
    #     with open('loss3.txt', 'a') as f:
    #         f.write(str(scheduler.get_lr()[0]))
    #         f.write("/n")
    #     print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
    if (epoch + 1) % (opt.nEpochs / 2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
    if (epoch + 1) % (opt.snapshots) == 0:
        checkpoint(epoch)
