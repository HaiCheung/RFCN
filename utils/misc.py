import os
import scipy
import numpy as np
from skimage.measure import label, regionprops


def mkdir_p(path):
    try:
        os.makedirs(path)
        return path
    except OSError as exc:  # Python >2.5
        import errno
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def BW_img(input, thresholding):
    if input.max() > thresholding:
        binary = input > thresholding
    else:
        binary = input > input.max() / 2.0
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))


def disc_crop(org_img, DiscROI_size, C_x, C_y):
    tmp_size = int(DiscROI_size / 2)
    disc_region = np.zeros((DiscROI_size, DiscROI_size, 3), dtype=org_img.dtype)  # (400, 400, 3)
    crop_coord = np.array([C_x - tmp_size, C_x + tmp_size, C_y - tmp_size, C_y + tmp_size], dtype=int)  # [787, 1187, 146, 546]
    err_coord = [0, DiscROI_size, 0, DiscROI_size]  # [0, 400, 0, 400]

    if crop_coord[0] < 0:
        err_coord[0] = abs(crop_coord[0])
        crop_coord[0] = 0

    if crop_coord[2] < 0:
        err_coord[2] = abs(crop_coord[2])
        crop_coord[2] = 0

    if crop_coord[1] > org_img.shape[0]:
        err_coord[1] = err_coord[1] - (crop_coord[1] - org_img.shape[0])
        crop_coord[1] = org_img.shape[0]

    if crop_coord[3] > org_img.shape[1]:
        err_coord[3] = err_coord[3] - (crop_coord[3] - org_img.shape[1])
        crop_coord[3] = org_img.shape[1]

    disc_region[err_coord[0]:err_coord[1], err_coord[2]:err_coord[3], ] = org_img[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], ]

    return disc_region, err_coord, crop_coord


def return_list(data_path, data_type):
    file_list = [file for file in os.listdir(data_path) if file.lower().endswith(data_type)]
    return file_list


def calc_dice(img_result, gt):
    test_od = img_result[1]
    test_cup = img_result[2]
    od_gt = gt[1, :, :] // 255
    cup_gt = gt[2, :, :] // 255
    od_difference = np.where(test_od != od_gt, 1, 0)
    cup_difference = np.where(test_cup != cup_gt, 1, 0)
    od_dice = 1 - (np.sum(od_difference)) / (np.sum(test_od) + np.sum(od_gt))
    cup_dice = 1 - (np.sum(cup_difference)) / (np.sum(test_cup) + np.sum(cup_gt))
    return od_dice, cup_dice
