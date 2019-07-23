import cv2
import argparse
from tqdm import tqdm
from skimage.transform import rotate, resize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import models
from utils.train_dataset import trainDataset
from utils.test_dataset import testDataset
from utils.ERF import ERF
from utils.misc import *

model_names = sorted(name for name in models.__dict__ if not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--device', type=str, default='0', help='GPU device (default: 0)')
parser.add_argument('--dataset', default='Drishti', choices=['Drishti'])
parser.add_argument('--data_path', type=str, default='datasets/', help='data path')

parser.add_argument('--model', type=str, default='RFCN', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: RFCN)')
parser.add_argument('--recurrent_n', type=int, default=2, help='recurrent num (default: 2)')
parser.add_argument('--layers', type=int, default=4, help='net layers (default: 4)')

parser.add_argument('--loss_weights', type=str, default='0.1,0.1,0.1,0.1,0.6', help='loss_weights')
parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'])
parser.add_argument('--batch_size', type=int, default=4, help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train (default: 400)')

parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

parser.add_argument('--data_augmentation', action='store_true', default=False, help='data augmentation')
parser.add_argument('--data_polar', action='store_true', default=False, help='data polar')
parser.add_argument('--input_size', type=int, default=512, help='input imgs size (default: 400)')
parser.add_argument('--train_crop_size', type=str, default='400,500,550,600,650,700,750,800,850,900', help='input image size')
parser.add_argument('--test_crop_size', type=int, default=700, help='test image size (default: 700)')

parser.add_argument('--threshold_confusion', default=0.4, type=float, help='threshold_confusion')
parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')

# use last save model
parser.add_argument('--load_last', action='store_true', default=False, help='load last model')
parser.add_argument('--load_path', type=str, default='logs/', help='load model path')
parser.add_argument('--logs_path', type=str, default='logs/', help='load model path')

args = vars(parser.parse_args())
os.environ['CUDA_VISIBLE_DEVICES'] = args['device']
cudnn.benchmark = True
torch.cuda.is_available()
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

threshold_confusion = args['threshold_confusion']

if str(args['logs_path']).endswith('/') is False:
    args['logs_path'] += '/'

if args['load_path'] is not None and str(args['load_path']).endswith('/') is False:
    args['load_path'] += '/'

if args['load_last'] is False:
    mkdir_p(args['logs_path'] + args['dataset'] + '/' + args['model'] + '/')
    index = np.sort(np.array(os.listdir(args['logs_path'] + args['dataset'] + '/' + args['model'] + '/'), dtype=int))
    index = index.max() + 1 if len(index) > 0 else 1
    basic_path = args['logs_path'] + args['dataset'] + '/' + args['model'] + '/' + str(index) + '/'
    mkdir_p(basic_path)
    args['load_path'] = basic_path
    max_OD_Dice, max_OD_Dice_cup_Dice, max_cup_Dice, max_cup_Dice_OD_Dice = 0., 0., 0., 0.
    cur_epoch = 0
    logs = [['epoch', 'OD_Dice', 'max_OD_Dice', 'max_OD_Dice_cup_Dice', 'cup_Dice', 'max_cup_Dice', 'max_cup_Dice_OD_Dice']]
else:
    basic_path = args['load_path']
    assert os.path.exists(basic_path), 'Error: Folder not exists'
    assert os.path.isfile(basic_path + 'checkpoints/last.pt'), 'Error: no checkpoint file found!'
    checkpoint = torch.load(basic_path + 'checkpoints/last.pt')
    checkpoint['args']['load_last'] = args['load_last']
    checkpoint['args']['load_path'] = args['load_path']
    args = checkpoint['args']
    max_OD_Dice = checkpoint['max_OD_Dice']
    max_OD_Dice_cup_Dice = checkpoint['max_OD_Dice_cup_Dice']
    max_cup_Dice = checkpoint['max_cup_Dice']
    max_cup_Dice_OD_Dice = checkpoint['max_cup_Dice_OD_Dice']
    cur_epoch = checkpoint['epoch'] + 1
    logs = checkpoint['logs']
    assert cur_epoch < args['epochs'], 'Done，cur_epoch: {}，epochs: {}'.format(cur_epoch, args['epochs'])

print('Current Folder： ' + basic_path)
mkdir_p(basic_path + 'checkpoints/periods/')
mkdir_p(basic_path + 'tensorboard/')
with open(basic_path + 'args.txt', 'w+') as f:
    for arg in args:
        print(str(arg) + ': ' + str(args[arg]))
        f.write(str(arg) + ': ' + str(args[arg]) + '\n')

net = models.__dict__[args['model']](num_classes=3, layers=args['layers'], filters=32, inplanes=3, recurrent_n=args['recurrent_n'])

criterions = []
for i in range(args['layers'] + 1):
    criterions.append(nn.CrossEntropyLoss().cuda())

loss_weights = [float(loss_weight) for loss_weight in args['loss_weights'].split(',')]

net.cuda()
if args['optimizer'] == 'Adam':
    optimizer = optim.Adam(net.parameters(), lr=args['lr'])
elif args['optimizer'] == 'SGD':
    optimizer = optim.SGD(net.parameters(), lr=args['lr'], momentum=0.9, weight_decay=5e-4)

if args['load_last'] is True and cur_epoch > 0:
    net.load_state_dict(checkpoint['net'], strict=False)
    print('load path: ' + basic_path + 'checkpoints/last.pt')

if args['dataset'] == 'Drishti':
    train_set = trainDataset(imgs_path='datasets/Drishti/Training/Images/',
                             gts_path='datasets/Drishti/Training/GT/',
                             crop_size=[int(train_crop_size) for train_crop_size in args['train_crop_size'].split(',')],
                             input_size=args['input_size'],
                             data_augmentation=args['data_augmentation'],
                             data_polar=args['data_polar'])
    test_set = testDataset(imgs_path='datasets/Drishti/Test/Images/',
                           gts_path='datasets/Drishti/Test/Test_GT/',
                           crop_size=[args['test_crop_size']],
                           input_size=args['input_size'],
                           data_polar=args['data_polar'])

train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=0)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

ts_writer = SummaryWriter(log_dir=basic_path + 'tensorboard/', comment=args['model'])
args_str = ''
for arg in args:
    args_str += str(arg) + ': ' + str(args[arg]) + '<br />'
ts_writer.add_text('args', args_str, cur_epoch)


def train():
    global max_OD_Dice, max_OD_Dice_cup_Dice, max_cup_Dice, max_cup_Dice_OD_Dice
    for epoch in range(cur_epoch, args['epochs']):
        # train network
        net.train()
        train_loss = 0
        progress_bar = tqdm(train_loader)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            progress_bar.set_description('Epoch {}/{}'.format(epoch + 1, args['epochs']))

            inputs = Variable(inputs.cuda().detach())
            targets = Variable(targets.long().cuda().detach())

            optimizer.zero_grad()
            outputs = net(inputs)

            loss_ = []
            for criterion, output, loss_weight in zip(criterions, outputs, loss_weights):
                loss_od = criterion(output[:, (0, 1), :, :], targets[:, 1, :, :].reshape(targets.shape[0], targets.shape[2], targets.shape[3]))
                loss_cup = criterion(output[:, (0, 2), :, :], targets[:, 2, :, :].reshape(targets.shape[0], targets.shape[2], targets.shape[3]))
                loss_.append(loss_weight * (0.5 * loss_od + 0.5 * loss_cup))

            loss = sum(loss_)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss='%.3f' % (train_loss / (batch_idx + 1)))

        net.eval()
        with torch.no_grad():
            OD_Dices = []
            cup_Dices = []
            for batch_idx, (test_input, test_target, test_disk_center, test_crop_point, test_gt_point) in enumerate(test_loader):
                test_input = test_input.detach().cuda()
                test_outputs = net(test_input)
                disc_predict = torch.nn.functional.softmax(test_outputs[-1][:, (0, 1), :, :], dim=1).cpu().data.numpy()[0]
                disc_map = (resize(disc_predict[1, :, :], (args['test_crop_size'], args['test_crop_size']), mode='constant') * 255).astype(np.uint8)
                cup_predict = torch.nn.functional.softmax(test_outputs[-1][:, (0, 2), :, :], dim=1).cpu().data.numpy()[0]
                cup_map = (resize(cup_predict[1, :, :], (args['test_crop_size'], args['test_crop_size']), mode='constant') * 255).astype(np.uint8)
                if args['data_polar'] is True:
                    disc_map[-round(args['test_crop_size'] / 3):, :] = 0
                    cup_map[-round(args['test_crop_size'] / 2):, :] = 0
                    disc_map = cv2.linearPolar(rotate(disc_map, 90), (args['test_crop_size'] / 2, args['test_crop_size'] / 2),
                                               args['test_crop_size'] / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
                    cup_map = cv2.linearPolar(rotate(cup_map, 90), (args['test_crop_size'] / 2, args['test_crop_size'] / 2),
                                              args['test_crop_size'] / 2, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)

                disc_map = np.array(BW_img(disc_map, threshold_confusion), dtype=int)
                cup_map = np.array(BW_img(cup_map, threshold_confusion), dtype=int)
                test_target = test_target[0].data.numpy()
                img_result = np.zeros(test_target.shape, dtype=np.int8)
                img_result[1, test_gt_point[0][0]:test_gt_point[0][1], test_gt_point[0][2]:test_gt_point[0][3]] = \
                    disc_map[test_crop_point[0][0]:test_crop_point[0][1], test_crop_point[0][2]:test_crop_point[0][3]]
                img_result[2, test_gt_point[0][0]:test_gt_point[0][1], test_gt_point[0][2]:test_gt_point[0][3]] = \
                    cup_map[test_crop_point[0][0]:test_crop_point[0][1], test_crop_point[0][2]:test_crop_point[0][3]]

                OD_Dice, cup_Dice = calc_dice(img_result, test_target)
                OD_Dices.append(OD_Dice)
                cup_Dices.append(cup_Dice)

            OD_Dice = np.mean(OD_Dices)
            cup_Dice = np.mean(cup_Dices)

            if max_OD_Dice < OD_Dice:
                max_OD_Dice = OD_Dice
                max_OD_Dice_cup_Dice = cup_Dice
            if max_cup_Dice < cup_Dice:
                max_cup_Dice_OD_Dice = OD_Dice
                max_cup_Dice = cup_Dice

            logs.append([epoch, OD_Dice, max_OD_Dice, max_OD_Dice_cup_Dice, cup_Dice, max_cup_Dice, max_cup_Dice_OD_Dice])
            state = {
                'net': net.state_dict(),
                'max_OD_Dice': max_OD_Dice,
                'max_OD_Dice_cup_Dice': max_OD_Dice_cup_Dice,
                'max_cup_Dice': max_cup_Dice,
                'max_cup_Dice_OD_Dice': max_cup_Dice_OD_Dice,
                'epoch': epoch,
                'logs': logs,
                'args': args
            }

            torch.save(state, basic_path + 'checkpoints/periods/{}.pt'.format(epoch))
            torch.save(state, basic_path + 'checkpoints/last.pt')

            lr = get_lr(optimizer)
            ts_writer.add_scalar('train/loss', train_loss / len(train_loader), epoch)
            ts_writer.add_scalar('train/lr', lr, epoch)
            ts_writer.add_scalar('test/OD_Dice', OD_Dice, epoch)
            ts_writer.add_scalar('test/cup_Dice', cup_Dice, epoch)
            tqdm.write('     OD_Dice: {:.4f},     cup_Dice: {:.4f}'.format(OD_Dice, cup_Dice))
            tqdm.write(' max_OD_Dice: {:.4f},     cup_Dice: {:.4f}'.format(max_OD_Dice, max_OD_Dice_cup_Dice))
            tqdm.write('     OD_Dice: {:.4f}, max_cup_Dice: {:.4f}'.format(max_cup_Dice_OD_Dice, max_cup_Dice))
            tqdm.write('learning rate: {:.4f}'.format(lr))


if __name__ == '__main__':
    train()
    ts_writer.close()
