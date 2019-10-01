import os
import torch
import torchvision
import glob
import numpy as np
import torch.optim as optim
import cv2
import matplotlib
import matplotlib.pyplot as plt

from skimage import io, transform
from skimage.transform import resize
from torch.autograd import Variable

import d_net
import config
import time

import vanilla_gan
import vanilla_gan.vanilla_gan
import vanilla_gan.video_gan
import data_loader
import loss_funs

import torch.nn as nn
import torch.nn.functional as F

import config

dtype = config.dtype


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        nin, nout = 3, 32
        self.conv1_depthwise = nn.Conv2d(
            nin, nout, 4, stride=2, padding=1, groups=1
        ).type(dtype)
        # self.conv1_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv1_depthwise.weight)
        # nn.init.xavier_normal(self.conv1_pointwise.weight)
        self.bn1 = nn.BatchNorm2d(32).type(dtype)

        nin, nout = 32, 64
        self.conv2_depthwise = nn.Conv2d(
            nin, nout, 4, stride=2, padding=1, groups=1
        ).type(dtype)
        # self.conv2_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv2_depthwise.weight)
        # nn.init.xavier_normal(self.conv2_pointwise.weight)
        self.bn2 = nn.BatchNorm2d(64).type(dtype)

        nin, nout = 64, 128
        self.conv3_depthwise = nn.Conv2d(
            nin, nout, 4, stride=2, padding=1, groups=1
        ).type(dtype)
        # self.conv3_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv3_depthwise.weight)
        # nn.init.xavier_normal(self.conv3_pointwise.weight)
        self.bn3 = nn.BatchNorm2d(128).type(dtype)

        nin, nout = 128, 1
        self.conv4_depthwise = nn.Conv2d(
            nin, nout, 4, stride=1, padding=1, groups=1
        ).type(dtype)
        # self.conv4_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv4_depthwise.weight)
        # nn.init.xavier_normal(self.conv4_pointwise.weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.type(dtype)
        # Conv 1
        out = self.conv1_depthwise(x)
        # out = self.conv1_pointwise(out)
        out = self.bn1(out)
        out = F.relu(out)

        # Conv 2
        out = self.conv2_depthwise(out)
        # out = self.conv2_pointwise(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Conv 3
        out = self.conv3_depthwise(out)
        # out = self.conv3_pointwise(out)
        out = self.bn3(out)
        out = F.relu(out)

        # Conv 4
        out = self.conv4_depthwise(out)
        # out = self.conv4_pointwise(out)
        if not config.use_wgan_loss:
            out = self.sigmoid(out)

        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(100, 128, 4, stride=4, padding=0).type(dtype)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(64).type(dtype)

        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(32).type(dtype)

        self.deconv4 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)

    def forward(self, x):
        out = self.deconv1(x.type(dtype))
        # TODO: Investigate putting Batch Norm before versus after the RELU layer
        # Resources:
        # https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
        # https://www.youtube.com/watch?v=Xogn6veSyxA&feature=youtu.be&t=325
        out = self.bn1(out)
        out = F.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.deconv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = self.deconv4(out)
        out = torch.tanh(out)

        return out


class GeneratorSkipConnections(nn.Module):
    def make_resblock(self, map_size):
        conv1_depthwise = nn.ConvTranspose2d(
            map_size, map_size, 3, stride=1, padding=1, groups=map_size
        ).type(dtype)
        conv1_pointwise = nn.ConvTranspose2d(map_size, map_size, 1).type(dtype)
        nn.init.xavier_normal(conv1_depthwise.weight)
        nn.init.xavier_normal(conv1_pointwise.weight)
        bn = nn.BatchNorm2d(map_size).type(dtype)
        conv2_depthwise = nn.ConvTranspose2d(
            map_size, map_size, 3, stride=1, padding=1, groups=map_size
        ).type(dtype)
        conv2_pointwise = nn.ConvTranspose2d(map_size, map_size, 1).type(dtype)
        nn.init.xavier_normal(conv2_depthwise.weight)
        nn.init.xavier_normal(conv2_pointwise.weight)

        resblock = nn.ModuleList()
        resblock.append(conv1_depthwise)
        resblock.append(conv1_pointwise)
        resblock.append(bn)
        resblock.append(conv2_depthwise)
        resblock.append(conv2_pointwise)

        return resblock

    def apply_resblock(self, out, resblock):
        out = resblock[0](out)
        out = resblock[1](out)
        out = resblock[2](out)
        out = F.relu(out)
        out = resblock[3](out)
        out = resblock[4](out)

        return out

    def __init__(self):
        super(GeneratorSkipConnections, self).__init__()

        # TODO: Change convolutions to DepthWise Seperable convolutions
        # TODO: Need to fix Mode Collapse that is occuring in the GAN
        # More info: https://www.quora.com/What-does-it-mean-if-all-produced-images-of-a-GAN-look-the-same

        # Upsampling layer
        nin, nout = 100, 128
        self.deconv1_depthwise = nn.ConvTranspose2d(
            nin, nin, 4, stride=4, padding=0, groups=nin
        ).type(dtype)
        self.deconv1_pointwise = nn.ConvTranspose2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.deconv1_depthwise.weight)
        nn.init.xavier_normal(self.deconv1_pointwise.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        # Resnet block
        self.resblock1A = self.make_resblock(128)

        # Upsampling layer
        nin, nout = 128, 64
        self.deconv2_depthwise = nn.ConvTranspose2d(
            nin, nin, 4, stride=2, padding=1, groups=nin
        ).type(dtype)
        self.deconv2_pointwise = nn.ConvTranspose2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.deconv2_depthwise.weight)
        nn.init.xavier_normal(self.deconv2_pointwise.weight)
        self.bn2 = nn.BatchNorm2d(64).type(dtype)

        # Resnet block
        self.resblock2A = self.make_resblock(64)

        # Upsampling layer 3
        nin, nout = 64, 32
        self.deconv3_depthwise = nn.ConvTranspose2d(
            nin, nin, 4, stride=2, padding=1, groups=nin
        ).type(dtype)
        self.deconv3_pointwise = nn.ConvTranspose2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.deconv3_depthwise.weight)
        nn.init.xavier_normal(self.deconv3_pointwise.weight)
        self.bn3 = nn.BatchNorm2d(32).type(dtype)

        # Resnet block
        self.resblock3A = self.make_resblock(32)

        # Upsampling layer 4
        nin, nout = 32, 3
        self.deconv4_depthwise = nn.ConvTranspose2d(
            nin, nin, 4, stride=2, padding=1, groups=nin
        ).type(dtype)
        self.deconv4_pointwise = nn.ConvTranspose2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.deconv4_depthwise.weight)
        nn.init.xavier_normal(self.deconv4_pointwise.weight)

        # Resnet block
        self.resblock4A = self.make_resblock(3)

    def forward(self, x):
        x = x.type(dtype)
        out = x

        # Multi scale image generation seems quite similar to using ResNet skip connections
        # In this case, we only use a single Resnet block instead of the entire Generator so the network is small enough to run on my laptop
        #
        # Upsample 1
        out = self.deconv1_depthwise(out)
        out = self.deconv1_pointwise(out)
        out = self.bn1(out)
        out = upsampled = F.relu(out)

        # Resnet block 1
        out += self.apply_resblock(out.clone(), self.resblock1A)

        # Upsample 2
        out = self.deconv2_depthwise(out)
        out = self.deconv2_pointwise(out)
        out = self.bn2(out)
        out = upsampled = F.relu(out)
        # Resnet block 2
        out += self.apply_resblock(out.clone(), self.resblock2A)

        # Upsample 3
        out = self.deconv3_depthwise(out)
        out = self.deconv3_pointwise(out)
        out = self.bn3(out)
        out = upsampled = F.relu(out)
        # Resnet block 3
        out += self.apply_resblock(out.clone(), self.resblock3A)

        # Upsample 4
        out = self.deconv4_depthwise(out)
        out = self.deconv4_pointwise(out)

        # Resnet block 4
        out += self.apply_resblock(out.clone(), self.resblock4A)

        out = torch.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        nin, nout = 3, 32
        self.conv1_depthwise = nn.Conv2d(
            nin, nout, 4, stride=2, padding=1, groups=1
        ).type(dtype)
        # self.conv1_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv1_depthwise.weight)
        # nn.init.xavier_normal(self.conv1_pointwise.weight)
        self.bn1 = nn.BatchNorm2d(32).type(dtype)

        nin, nout = 32, 64
        self.conv2_depthwise = nn.Conv2d(
            nin, nout, 4, stride=2, padding=1, groups=1
        ).type(dtype)
        # self.conv2_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv2_depthwise.weight)
        # nn.init.xavier_normal(self.conv2_pointwise.weight)
        self.bn2 = nn.BatchNorm2d(64).type(dtype)

        nin, nout = 64, 128
        self.conv3_depthwise = nn.Conv2d(
            nin, nout, 4, stride=2, padding=1, groups=1
        ).type(dtype)
        # self.conv3_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv3_depthwise.weight)
        # nn.init.xavier_normal(self.conv3_pointwise.weight)
        self.bn3 = nn.BatchNorm2d(128).type(dtype)

        nin, nout = 128, 1
        self.conv4_depthwise = nn.Conv2d(
            nin, nout, 4, stride=1, padding=1, groups=1
        ).type(dtype)
        # self.conv4_pointwise = nn.Conv2d(nin, nout, 1).type(dtype)
        nn.init.xavier_normal(self.conv4_depthwise.weight)
        # nn.init.xavier_normal(self.conv4_pointwise.weight)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.type(dtype)
        # Conv 1
        out = self.conv1_depthwise(x)
        # out = self.conv1_pointwise(out)
        out = self.bn1(out)
        out = F.relu(out)

        # Conv 2
        out = self.conv2_depthwise(out)
        # out = self.conv2_pointwise(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Conv 3
        out = self.conv3_depthwise(out)
        # out = self.conv3_pointwise(out)
        out = self.bn3(out)
        out = F.relu(out)

        # Conv 4
        out = self.conv4_depthwise(out)
        # out = self.conv4_pointwise(out)
        if not config.use_wgan_loss:
            out = self.sigmoid(out)

        return out


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(12, 128, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        self.deconv2 = nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(64).type(dtype)

        self.deconv3 = nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(32).type(dtype)

        self.deconv4 = nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)

    def forward(self, x):
        out = self.deconv1(x).type(dtype)
        # TODO: Investigate putting Batch Norm before versus after the RELU layer
        # Resources:
        # https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
        # https://www.youtube.com/watch?v=Xogn6veSyxA&feature=youtu.be&t=325
        out = self.bn1(out)
        out = F.relu(out)

        out = self.deconv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.deconv3(out)
        out = self.bn3(out)
        out = F.relu(out)

        out = self.deconv4(out)
        out = torch.tanh(out)

        return out


class Gen1(nn.Module):
    def __init__(self):
        super(Gen1, self).__init__()

        # Generator #1
        self.g1 = nn.ModuleList()
        self.deconv1 = nn.Conv2d(12, 128, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        self.deconv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(256).type(dtype)

        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(128).type(dtype)

        self.deconv4 = nn.ConvTranspose2d(128, 3, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)

        self.g1.append(self.deconv1)
        self.g1.append(self.bn1)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv2)
        self.g1.append(self.bn2)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv3)
        self.g1.append(self.bn3)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv4)

    def forward(self, x):
        out = x.type(dtype)
        for layer in self.g1:
            out = layer(out)
        return out


class Gen2(nn.Module):
    def __init__(self):
        super(Gen2, self).__init__()

        # Generator #2
        self.g1 = nn.ModuleList()
        self.deconv1 = nn.Conv2d(15, 128, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        self.deconv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(256).type(dtype)

        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(128).type(dtype)

        self.deconv4 = nn.ConvTranspose2d(128, 3, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)

        self.g1.append(self.deconv1)
        self.g1.append(self.bn1)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv2)
        self.g1.append(self.bn2)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv3)
        self.g1.append(self.bn3)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv4)

    def forward(self, x):
        out = x.type(dtype)
        for layer in self.g1:
            out = layer(out)
        return out


class Gen3(nn.Module):
    def __init__(self):
        super(Gen3, self).__init__()

        # Generator #3
        self.g1 = nn.ModuleList()
        self.deconv1 = nn.Conv2d(15, 128, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        self.deconv2 = nn.Conv2d(128, 256, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(256).type(dtype)

        self.deconv3 = nn.ConvTranspose2d(256, 512, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(512).type(dtype)

        self.deconv4 = nn.ConvTranspose2d(512, 256, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)
        self.bn4 = nn.BatchNorm2d(256).type(dtype)

        self.deconv5 = nn.ConvTranspose2d(256, 128, 3, stride=1, padding=1).type(dtype)
        nn.init.xavier_normal(self.deconv5.weight)
        self.bn5 = nn.BatchNorm2d(128).type(dtype)

        self.deconv6 = nn.ConvTranspose2d(128, 3, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)

        self.g1.append(self.deconv1)
        self.g1.append(self.bn1)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv2)
        self.g1.append(self.bn2)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv3)
        self.g1.append(self.bn3)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv4)
        self.g1.append(self.bn4)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv5)
        self.g1.append(self.bn5)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv6)

    def forward(self, x):
        out = x.type(dtype)
        for layer in self.g1:
            out = layer(out)
        return out


class Gen4(nn.Module):
    def __init__(self):
        super(Gen4, self).__init__()

        # Generator #4
        self.g1 = nn.ModuleList()
        self.deconv1 = nn.Conv2d(15, 128, 7, stride=1, padding=3).type(dtype)
        nn.init.xavier_normal(self.deconv1.weight)
        self.bn1 = nn.BatchNorm2d(128).type(dtype)

        self.deconv2 = nn.Conv2d(128, 256, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv2.weight)
        self.bn2 = nn.BatchNorm2d(256).type(dtype)

        self.deconv3 = nn.ConvTranspose2d(256, 512, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv3.weight)
        self.bn3 = nn.BatchNorm2d(512).type(dtype)

        self.deconv4 = nn.ConvTranspose2d(512, 256, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)
        self.bn4 = nn.BatchNorm2d(256).type(dtype)

        self.deconv5 = nn.ConvTranspose2d(256, 128, 5, stride=1, padding=2).type(dtype)
        nn.init.xavier_normal(self.deconv5.weight)
        self.bn5 = nn.BatchNorm2d(128).type(dtype)

        self.deconv6 = nn.ConvTranspose2d(128, 3, 7, stride=1, padding=3).type(dtype)
        nn.init.xavier_normal(self.deconv4.weight)

        self.g1.append(self.deconv1)
        self.g1.append(self.bn1)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv2)
        self.g1.append(self.bn2)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv3)
        self.g1.append(self.bn3)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv4)
        self.g1.append(self.bn4)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv5)
        self.g1.append(self.bn5)
        self.g1.append(nn.ReLU())
        self.g1.append(self.deconv6)

    def forward(self, x):
        out = x.type(dtype)
        for layer in self.g1:
            out = layer(out)
        return out


class VideoGANGenerator(nn.Module):
    """This class implements the full VideoGAN Generator Network.
    Currently a placeholder that copies the Vanilla GAN Generator network
    """

    def __init__(self):
        super(VideoGANGenerator, self).__init__()

        self.up1 = nn.ConvTranspose2d(
            3, 3, 3, stride=2, padding=1, output_padding=1
        ).type(dtype)
        self.up2 = nn.ConvTranspose2d(
            3, 3, 3, stride=2, padding=1, output_padding=1
        ).type(dtype)
        self.up3 = nn.ConvTranspose2d(
            3, 3, 3, stride=2, padding=1, output_padding=1
        ).type(dtype)

        # Generator #1
        self.g1 = Gen1()
        self.g2 = Gen2()
        self.g3 = Gen3()
        self.g4 = Gen4()

    def forward(self, x):
        out = x.type(dtype)

        # TODO: Change the image size
        img1 = F.interpolate(out, size=(4, 4))
        img2 = F.interpolate(out, size=(8, 8))
        img3 = F.interpolate(out, size=(16, 16))
        img4 = out

        out = self.g1(img1)
        upsample1 = self.up1(out)
        out = upsample1 + self.g2(torch.cat([img2, upsample1], dim=1))
        upsample2 = self.up2(out)
        out = upsample2 + self.g3(torch.cat([img3, upsample2], dim=1))
        upsample3 = self.up3(out)
        out = upsample3 + self.g4(torch.cat([img4, upsample3], dim=1))

        # Apply tanh at the end
        out = torch.tanh(out)

        return out


VIDEO_GAN = True
VANILLA_GAN = not VIDEO_GAN


def save_samples(generated_images, iteration, prefix):
    import scipy

    generated_images = generated_images.data.cpu().numpy()

    num_images, channels, cell_h, cell_w = generated_images.shape
    ncols = int(np.sqrt(num_images))
    nrows = int(np.math.floor(num_images / float(ncols)))
    result = np.zeros(
        (cell_h * nrows, cell_w * ncols, channels), dtype=generated_images.dtype
    )
    for i in range(0, nrows):
        for j in range(0, ncols):
            result[
                i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w, :
            ] = generated_images[i * ncols + j].transpose(1, 2, 0)
    grid = result

    if not os.path.exists("output"):
        os.makedirs("output")
    scipy.misc.imsave("output/{}_{:05d}.jpg".format(prefix, iteration), grid)


def sample_noise(batch_size, dim):
    result = torch.rand(batch_size, dim) * 2 - 1
    result = Variable(result).unsqueeze(2).unsqueeze(3)

    return result


def get_emoji_loader(emoji_type):
    from torchvision import datasets
    from torchvision import transforms
    from torch.utils.data import DataLoader

    num_workers = 1
    batch_size = 16
    image_size = 32

    transform = transforms.Compose(
        [
            transforms.Scale(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_path = os.path.join("./emojis", emoji_type)
    test_path = os.path.join("./emojis", "Test_{}".format(emoji_type))

    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    train_dloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dloader, test_dloader


def main():
    SCALE_CONV_FSM_D = [
        [3, 64],
        [3, 64, 128, 128],
        [3, 128, 256, 256],
        [3, 128, 256, 512, 128],
    ]
    SCALE_KERNEL_SIZES_D = [[3], [3, 3, 3], [5, 5, 5], [7, 7, 5, 5]]
    SCALE_FC_LAYER_SIZES_D = [
        [512, 256, 1],
        [1024, 512, 1],
        [1024, 512, 1],
        [1024, 512, 1],
    ]

    loss_fp = open("losses.csv", "w")

    if VIDEO_GAN:
        # TODO: Remove logic.
        if False:
            video_d_net = vanilla_gan.video_gan.Discriminator()
            video_d_net.type(dtype)

            video_g_net = vanilla_gan.video_gan.Generator()
            video_g_net.type(dtype)
        else:
            video_d_net = d_net.DiscriminatorModel(
                kernel_sizes_list=SCALE_KERNEL_SIZES_D,
                conv_layer_fms_list=SCALE_CONV_FSM_D,
                scale_fc_layer_sizes_list=SCALE_FC_LAYER_SIZES_D,
            )
            video_d_net.type(dtype)

            video_g_net = vanilla_gan.video_gan.VideoGANGenerator()
            video_g_net.type(dtype)

        video_d_optimizer = optim.Adam(video_d_net.parameters(), lr=0.0001)
        video_g_optimizer = optim.Adam(video_g_net.parameters(), lr=0.0001)

    # Load Pacman dataset
    max_size = len(os.listdir("train"))
    pacman_dataloader = data_loader.DataLoader(
        "train", min(max_size, 500000), 16, 32, 32, 4
    )

    # Load emojis
    train_dataloader, _ = get_emoji_loader("Windows")

    count = 0
    for i in range(1, 5000):
        for batch in train_dataloader:
            if VIDEO_GAN:
                clips_x, clips_y = pacman_dataloader.get_train_batch()
                clips_x = torch.tensor(np.rollaxis(clips_x, 3, 1)).type(dtype)
                clips_y = torch.tensor(np.rollaxis(clips_y, 3, 1)).type(dtype)

            if VIDEO_GAN:
                video_d_optimizer.zero_grad()
                video_g_optimizer.zero_grad()

            # batch_size x noise_size x 1 x 1
            batch_size = 16
            noise_size = 100
            sampled_noise = sample_noise(batch_size, noise_size)

            # WGAN loss
            # https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

            if VIDEO_GAN:
                video_images = video_g_net(clips_x)
                # TESTING: Vanilla Video Gan
                video_d_loss_real = (video_d_net(clips_y) - 1).pow(2).mean()
                video_d_loss_fake = (video_d_net(video_images)).pow(2).mean()

                # Fake batch
                labels = torch.zeros(batch_size, 4).t().unsqueeze(2).type(dtype)
                video_d_loss_fake = loss_funs.adv_loss(
                    video_d_net(video_images), labels
                )  # TODO: Validate if it's right.
                video_d_optimizer.zero_grad()
                video_d_loss_fake.backward()
                video_d_optimizer.step()

                # Real batch
                labels = torch.ones(batch_size, 4).t().unsqueeze(2).type(dtype)
                video_d_loss_real = loss_funs.adv_loss(
                    video_d_net(clips_y), labels
                )  # TODO: Validate if it's right.
                video_d_optimizer.zero_grad()
                video_d_loss_real.backward()
                video_d_optimizer.step()

                # video_d_loss.backward()
                # video_d_optimizer.step()
                # video_d_loss_real.backward()

                # batch_size x noise_size x 1 x 1
                batch_size = 16
                noise_size = 100
                sampled_noise = sample_noise(batch_size, noise_size)

                # print('G_Time:', end - start)

                # TESTING: Vanilla Video Gan
                video_images = video_g_net(clips_x)
                video_g_loss_fake = (video_d_net(video_images) - 1).pow(2).mean()
                d_preds = video_d_net(video_images).type(
                    dtype
                )  # TODO: Make sure this is working.
                gt_frames = clips_y.type(
                    dtype
                )  # TODO: make clips_y at different scales.
                gen_frames = video_images.type(
                    dtype
                )  # TODO: make the generated frames multi scale.
                video_g_loss = loss_funs.combined_loss(gen_frames, gt_frames, d_preds)
                video_g_loss.backward()
                video_g_optimizer.step()

            if count % 20 == 0:
                if VIDEO_GAN:
                    save_samples(clips_y, count, "video_real")
                    save_samples(video_images, count, "video_fake")

                    loss_fp.write(
                        "{},{},{},{}".format(
                            count, video_d_loss_real, video_d_loss_fake, video_g_loss
                        )
                    )
                torch.save(video_g_net.state_dict(), "generator_net.pth.tmp")
            count += 1

    loss_fp.close()

    # Final Generator save.
    torch.save(video_g_net.state_dict(), "generator_net.pth")


if __name__ == "__main__":
    main()
