import cv2
import argparse
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from skimage.transform import rotate, resize

import models
from utils.test_dataset import testDataset
from utils.misc import *

model_names = sorted(name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='CNN')
# use last save model
parser.add_argument('--device', type=str, default='0', help='GPU device (default: 0)')
parser.add_argument('--test_crop_size', type=str, default='700,750,800,850,900', help='input image size')
parser.add_argument('--threshold_confusion', default=0.4, type=float, help='threshold_confusion')
parser.add_argument('--check_path', type=str, default='logs/Drishti/MNet/CrossEntropyLoss/1/checkpoints/periods/372.pt', help='load model path')

args = vars(parser.parse_args())

checkpoint = torch.load(args['check_path'])
checkpoint['args']['device'] = args['device']
checkpoint['args']['check_path'] = args['check_path']
checkpoint['args']['test_crop_size'] = args['test_crop_size']
checkpoint['args']['threshold_confusion'] = args['threshold_confusion']
args = checkpoint['args']
logs = checkpoint['logs']
print(logs[-1])

os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
cudnn.benchmark = True
torch.cuda.is_available()
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])
threshold_confusion = args['threshold_confusion']

path_arr = args['check_path'].split('/')
data_save_path = mkdir_p(args['check_path'][:-3] + '_result/')

disc_list = [int(test_crop_size) for test_crop_size in args['test_crop_size'].split(',')]

net = models.__dict__[args['model']](num_classes=3, layers=args['layers'], filters=32, inplanes=3)
net.eval().cuda()
net.load_state_dict(checkpoint['net'], strict=False)

file = open(str(data_save_path) + 'performance.txt', 'w+')
for disc_idx in range(len(disc_list)):
    crop_size = disc_list[disc_idx]
    locals()['OD_Dice_' + str(crop_size)] = []
    locals()['cup_Dice_' + str(crop_size)] = []
    test_set = testDataset(imgs_path='datasets/Drishti/Test/Images/',
                           gts_path='datasets/Drishti/Test/Test_GT/',
                           crop_size=[crop_size],
                           input_size=args['input_size'],
                           data_polar=args['data_polar'])

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    with torch.no_grad():
        OD_Dices = []
        cup_Dices = []
        for batch_idx, (test_input, test_target, test_disk_center, test_crop_point, test_gt_point) in enumerate(test_loader):
            test_input = test_input.detach().cuda()
            test_outputs = net(test_input)
            disc_predict = torch.nn.functional.softmax(test_outputs[-1][:, (0, 1), :, :], dim=1).cpu().data.numpy()[0]
            disc_map = (resize(disc_predict[1, :, :], (crop_size, crop_size), mode='constant') * 255).astype(np.uint8)
            cup_predict = torch.nn.functional.softmax(test_outputs[-1][:, (0, 2), :, :], dim=1).cpu().data.numpy()[0]
            cup_map = (resize(cup_predict[1, :, :], (crop_size, crop_size), mode='constant') * 255).astype(np.uint8)
            if args['data_polar'] is True:
                disc_map[-round(crop_size / 3):, :] = 0
                cup_map[-round(crop_size / 2):, :] = 0
                disc_map = cv2.linearPolar(rotate(disc_map, 90), (crop_size / 2, crop_size / 2),
                                           crop_size / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
                cup_map = cv2.linearPolar(rotate(cup_map, 90), (crop_size / 2, crop_size / 2),
                                          crop_size / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

            disc_map = np.array(BW_img(disc_map, threshold_confusion), dtype=int)
            cup_map = np.array(BW_img(cup_map, threshold_confusion), dtype=int)
            test_target = test_target[0].data.numpy()
            img_result = np.zeros(test_target.shape, dtype=np.int8)
            img_result[1, test_gt_point[0][0]:test_gt_point[0][1], test_gt_point[0][2]:test_gt_point[0][3]] = \
                disc_map[test_crop_point[0][0]:test_crop_point[0][1], test_crop_point[0][2]:test_crop_point[0][3]]
            img_result[2, test_gt_point[0][0]:test_gt_point[0][1], test_gt_point[0][2]:test_gt_point[0][3]] = \
                cup_map[test_crop_point[0][0]:test_crop_point[0][1], test_crop_point[0][2]:test_crop_point[0][3]]

            OD_Dice, cup_Dice = calc_dice(img_result, test_target)
            eval('OD_Dice_' + str(crop_size)).append(OD_Dice)
            eval('cup_Dice_' + str(crop_size)).append(cup_Dice)

        print("%3s mean OD_Dice: %4f + %4f, mean cup_Dice: %4f + %4f" % (crop_size,
                                                                         np.mean(eval('OD_Dice_' + str(crop_size))),
                                                                         np.std(eval('OD_Dice_' + str(crop_size))),
                                                                         np.mean(eval('cup_Dice_' + str(crop_size))),
                                                                         np.std(eval('cup_Dice_' + str(crop_size)))))
        file.write("%3s,%4f,%4f,%4f,%4f\n" % (crop_size,
                                                      np.mean(eval('OD_Dice_' + str(crop_size))),
                                                      np.std(eval('OD_Dice_' + str(crop_size))),
                                                      np.mean(eval('cup_Dice_' + str(crop_size))),
                                                      np.std(eval('cup_Dice_' + str(crop_size)))))
file.close()
