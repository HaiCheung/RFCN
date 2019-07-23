import os
import cv2
import math
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from skimage.transform import rotate, resize


def return_list(data_path, data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    return file_list


class testDataset(Dataset):
    def __init__(self, imgs_path, gts_path, crop_size, input_size, data_polar=False):
        self.imgs_path = imgs_path
        self.gts_path = gts_path
        self.crop_size = crop_size
        self.input_size = input_size
        self.data_polar = data_polar
        self.img_h = 0
        self.img_w = 0
        self.imgs, self.gts, self.disk_centers, self.crop_points, self.gt_points = self.get_dataset()
        self.p = 0.5

    def __len__(self):
        return self.imgs.shape[0]

    def disc_crop(self, org_img, crop_size, C_x, C_y):
        tmp_size = int(crop_size / 2)
        disc_region = np.zeros((crop_size, crop_size, org_img.shape[2]), dtype=org_img.dtype)  # (400, 400, 3)
        gt_point = np.array([C_x - tmp_size, C_x + tmp_size, C_y - tmp_size, C_y + tmp_size], dtype=int)  # [787, 1187, 146, 546]
        crop_point = [0, crop_size, 0, crop_size]  # [0, 400, 0, 400]

        if gt_point[0] < 0:
            crop_point[0] = abs(gt_point[0])
            gt_point[0] = 0

        if gt_point[2] < 0:
            crop_point[2] = abs(gt_point[2])
            gt_point[2] = 0

        if gt_point[1] > org_img.shape[0]:
            crop_point[1] = crop_point[1] - (gt_point[1] - org_img.shape[0])
            gt_point[1] = org_img.shape[0]

        if gt_point[3] > org_img.shape[1]:
            crop_point[3] = crop_point[3] - (gt_point[3] - org_img.shape[1])
            gt_point[3] = org_img.shape[1]

        disc_region[crop_point[0]:crop_point[1], crop_point[2]:crop_point[3], :] = org_img[gt_point[0]:gt_point[1], gt_point[2]:gt_point[3], :]

        return disc_region, crop_point, gt_point

    def get_dataset(self):
        file_imgs_list = return_list(self.imgs_path, '.png')

        imgs = []
        gts = []
        disk_centers = []
        crop_points = []
        gt_points = []
        for file_img in file_imgs_list:
            image = cv2.imread(os.path.join(self.imgs_path, file_img[:-4] + '.png'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # print(image.shape)
            if self.img_h == 0 or self.img_w == 0:
                self.img_h, self.img_w = image.shape[0], image.shape[1]

            gt = np.zeros(np.shape(image), dtype=np.uint8)
            org_OD = cv2.imread(os.path.join(self.gts_path, file_img[:-4], 'SoftMap', file_img[:-4] + '_ODsegSoftmap.png'))
            org_OD = cv2.cvtColor(org_OD, cv2.COLOR_BGR2RGB)[:, :, 0]
            org_cup = cv2.imread(os.path.join(self.gts_path, file_img[:-4], 'SoftMap', file_img[:-4] + '_cupsegSoftmap.png'))
            org_cup = cv2.cvtColor(org_cup, cv2.COLOR_BGR2RGB)[:, :, 0]
            gt[org_OD > np.mean(np.unique(org_OD)), 1] = 255
            gt[org_cup > np.mean(np.unique(org_cup)), 2] = 255
            # print(gt.shape)
            gts.append(gt.astype(np.uint8))

            with open(os.path.join(self.gts_path, file_img[:-4], 'AvgBoundary', file_img[:-4] + '_diskCenter.txt')) as f:
                disk_centers.append(np.array([int(i) for i in f.read().split(' ')]))

            for crop_size in self.crop_size:
                crop_img, crop_point, gt_point = self.disc_crop(image, crop_size, disk_centers[-1][0], disk_centers[-1][1])

                if self.data_polar is True:
                    # 首先进行极坐标变换，然后旋转90度
                    crop_img = rotate(cv2.linearPolar(crop_img, (crop_size / 2, crop_size / 2), crop_size / 2,
                                                      cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)
                    crop_img = (crop_img * 255).astype(np.uint8)

                imgs.append((resize(crop_img, (self.input_size, self.input_size, 3), mode='constant') * 255).astype(np.uint8))
                crop_points.append(crop_point)
                gt_points.append(gt_point)

        imgs = np.asarray(imgs)
        gts = np.asarray(gts)
        # print('Test imgs shape: ' + str(imgs.shape))
        # print('Test gts shape: ' + str(gts.shape) + ' (测试集图像大小不相同)')
        return imgs, gts, np.asarray(disk_centers), np.asarray(crop_points), np.asarray(gt_points)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        gt = self.gts[idx]
        disk_center = self.disk_centers[idx]
        crop_point = self.crop_points[idx]
        gt_point = self.gt_points[idx]
        img = Image.fromarray(img)
        gt = Image.fromarray(gt)

        img = TF.to_tensor(img)
        gt = np.transpose(np.array(gt), (2, 0, 1))
        return img, gt, disk_center, crop_point, gt_point

