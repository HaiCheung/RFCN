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


class trainDataset(Dataset):
    def __init__(self, imgs_path, gts_path, crop_size, input_size, data_augmentation=False, data_polar=False):
        self.imgs_path = imgs_path
        self.gts_path = gts_path
        self.crop_size = crop_size
        self.input_size = input_size
        self.data_augmentation = data_augmentation
        self.data_polar = data_polar
        self.imgs, self.gts = self.get_dataset()
        self.p = 0.5

    def __len__(self):
        return self.imgs.shape[0]

    def disc_crop(self, org_img, crop_size, C_x, C_y):
        tmp_size = int(crop_size / 2)
        disc_region = np.zeros((crop_size, crop_size, org_img.shape[2]), dtype=org_img.dtype)
        gt_point = np.array([C_x - tmp_size, C_x + tmp_size, C_y - tmp_size, C_y + tmp_size], dtype=int)
        crop_point = [0, crop_size, 0, crop_size]

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
        for file_img in file_imgs_list:
            image = cv2.imread(os.path.join(self.imgs_path, file_img[:-4] + '.png'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gt = np.zeros(np.shape(image), dtype=np.uint8)
            org_OD = cv2.imread(os.path.join(self.gts_path, file_img[:-4], 'SoftMap', file_img[:-4] + '_ODsegSoftmap.png'))
            org_OD = cv2.cvtColor(org_OD, cv2.COLOR_BGR2RGB)[:, :, 0]
            org_cup = cv2.imread(os.path.join(self.gts_path, file_img[:-4], 'SoftMap', file_img[:-4] + '_cupsegSoftmap.png'))
            org_cup = cv2.cvtColor(org_cup, cv2.COLOR_BGR2RGB)[:, :, 0]
            gt[org_OD > np.mean(np.unique(org_OD)), 1] = 255
            gt[org_cup > np.mean(np.unique(org_cup)), 2] = 255

            with open(os.path.join(self.gts_path, file_img[:-4], 'AvgBoundary', file_img[:-4] + '_diskCenter.txt')) as f:
                disk_center = np.array([int(i) for i in f.read().split(' ')])

            for crop_size in self.crop_size:
                crop_img, crop_point, gt_point = self.disc_crop(image, crop_size, disk_center[0], disk_center[1])
                crop_gt, _, _ = self.disc_crop(gt, crop_size, disk_center[0], disk_center[1])

                if self.data_polar is True:
                    # 首先进行极坐标变换，然后旋转90度
                    crop_img = rotate(cv2.linearPolar(crop_img, (crop_size / 2, crop_size / 2), crop_size / 2,
                                                      cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)
                    crop_gt = rotate(cv2.linearPolar(crop_gt, (crop_size / 2, crop_size / 2), crop_size / 2,
                                                     cv2.INTER_NEAREST + cv2.WARP_FILL_OUTLIERS), -90)
                    crop_img = (crop_img * 255).astype(np.uint8)
                    crop_gt = (crop_gt * 255).astype(np.uint8)

                imgs.append((resize(crop_img, (self.input_size, self.input_size, 3), mode='constant') * 255).astype(np.uint8))
                gts.append((resize(crop_gt, (self.input_size, self.input_size, 3), mode='constant') * 255).astype(np.uint8))

        imgs = np.asarray(imgs)
        gts = np.asarray(gts)
        print('Train imgs shape: ' + str(imgs.shape))
        print('Train gts shape: ' + str(gts.shape))
        return imgs, gts

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __getitem__(self, idx):
        img = self.imgs[idx]
        gt = self.gts[idx]
        h, w = img.shape[0], img.shape[1]
        img = Image.fromarray(img)
        gt = Image.fromarray(gt)

        if self.data_augmentation is True and random.random() < 0.6:
            # 随机水平翻转
            if random.random() > self.p:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            # 随机垂直翻转
            if random.random() > self.p:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
            # 逆时针90度
            if random.random() > self.p:
                img = img.transpose(Image.ROTATE_90)
                gt = gt.transpose(Image.ROTATE_90)
            # 逆时针180度
            if random.random() > self.p:
                img = img.transpose(Image.ROTATE_180)
                gt = gt.transpose(Image.ROTATE_180)
            # 逆时针270度
            if random.random() > self.p:
                img = img.transpose(Image.ROTATE_270)
                gt = gt.transpose(Image.ROTATE_270)
            # 随机裁剪
            img = TF.pad(img, int(math.ceil(h * 1 / 8)))
            gt = TF.pad(gt, int(math.ceil(h * 1 / 8)))
            i, j, h, w = self.get_params(img, (h, w))
            img = img.crop((j, i, j + w, i + h))
            gt = gt.crop((j, i, j + w, i + h))

        return TF.to_tensor(img), TF.to_tensor(np.array(gt))


if __name__ == '__main__':
    train_set = trainDataset(imgs_path='/media/izhangh/软件/GaoJing/datasets/Drishti/Training/Images/',
                             gts_path='/media/izhangh/软件/GaoJing/datasets/Drishti/Training/GT/',
                             crop_size=[400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900],
                             input_size=400,
                             data_augmentation=True,
                             data_polar=True)
