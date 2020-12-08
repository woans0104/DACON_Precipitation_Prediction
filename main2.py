import argparse
import torch.utils.data as data
import model
import ipdb
import os
import dataloader as loader
import torchvision.transforms as transforms
import numpy as np
from medpy.metric import binary
from utils import Logger, AverageMeter, save_checkpoint ,draw_curve ,str2bool
import time
import shutil
import pickle
from optimizers import RAdam
import torch.optim as optim
from model import *
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from sync_batchnorm import convert_model # for sycncronized batchnormailztion
from dataloader import *
import pandas as pd
from predict import predict


parser = argparse.ArgumentParser()
# arguments for dataset
parser.add_argument('--work-dir', default='/data1/JM/dacon_rain')
parser.add_argument('--exp',default="test4", type=str)
parser.add_argument('--train-dataset',default='/data2/woans0104/dacon_rain_dataset/train/')
parser.add_argument('--test-dataset',default='/data2/woans0104/dacon_rain_dataset/test/')

parser.add_argument('--batch-size',default=4,type=int)
parser.add_argument('--arg-mode',default=False,type=str2bool)
parser.add_argument('--arg-thres',default=0.7,type=float)

# arguments for model
parser.add_argument('--arch', default='unet', type=str)
parser.add_argument('--start-channel',default=32,type=int)
parser.add_argument('--coordconv-no', default=[9], nargs='+', type=int)
parser.add_argument('--radious',default=False,type=str2bool)

# arguments for optim & loss
parser.add_argument('--optim',default='sgd',choices=['adam','sgd', 'radam'],type=str)
parser.add_argument('--weight-decay',default=1e-4,type=float)

parser.add_argument('--loss-function',default='bce',type=str)
parser.add_argument('--bce-weight', default=1, type=float)

parser.add_argument('--scheduler',default=True,type=str)
parser.add_argument('--initial-lr',default=0.1,type=float)
parser.add_argument('--lr-schedule', default=[120,150], nargs='+', type=int)


# arguments for dataset
parser.add_argument('--train-size',default=0.8,type=float)
parser.add_argument('--train_mode',default=True,type=str2bool)
parser.add_argument('--test_mode',default=True,type=str2bool)

# arguments for test mode
parser.add_argument('--inplace-test', default=1, type=int)
parser.add_argument('--file-name', default='result_all', type=str)
parser.add_argument('--test-root',default='/data2/woans0104/dacon_rain_dataset/test/',type=str)


args = parser.parse_args()

def main():
    # save input stats for later use
    print(args.work_dir, args.exp)
    work_dir = os.path.join(args.work_dir, args.exp)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    # copy this file to work dir to keep training configuration
    shutil.copy(__file__, os.path.join(work_dir, 'main.py'))
    with open(os.path.join(work_dir, 'args.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # 1.train_dataset
    torch.manual_seed(111)
    torch.cuda.manual_seed(111)
    data_sets = dacon_naive_loader(args.train_dataset)
    indices = torch.randperm(len(data_sets))
    train_indices = indices[:int(len(indices) * 0.8)]
    valid_indices = indices[int(len(indices) * 0.8):]

    torch.manual_seed(111)
    torch.cuda.manual_seed(111)
    trn_loader = torch.utils.data.DataLoader(data_sets, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices))
    torch.manual_seed(111)
    torch.cuda.manual_seed(111)
    val_loader = torch.utils.data.DataLoader(data_sets, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(valid_indices))

    #np.save(os.path.join(work_dir, '{}_test_path.npy'.format(args.train_dataset)), args.test_dataset)

    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))


    # 3.model_select
    my_net, model_name = model_select(args.arch)

    # 4.gpu select
    #my_net = convert_model(my_net)

    #my_net = nn.DataParallel(my_net).cuda()
    my_net = my_net.cuda()
    cudnn.benchmark = True

    # 5.optim

    if args.optim == 'adam':
        gen_optimizer = torch.optim.Adam(my_net.parameters(), lr=args.initial_lr, eps=1e-08)
    elif args.optim == 'sgd':
        gen_optimizer = torch.optim.SGD(my_net.parameters(), lr=args.initial_lr, momentum=0.9,weight_decay=args.weight_decay)
    elif args.optim == 'radam' :
        gen_optimizer = RAdam(my_net.parameters(), lr=args.initial_lr, weight_decay = args.weight_decay)



    # lr decay
    lr_schedule = args.lr_schedule
    lr_scheduler = optim.lr_scheduler.MultiStepLR(gen_optimizer,
                                                  milestones=lr_schedule[:-1],
                                                  gamma=0.1)

    # 6.loss
    if args.loss_function == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([args.bce_weight])).cuda()
    elif args.loss_function == 'mse':
        criterion = nn.MSELoss().cuda()
    elif args.loss_function == 'mae' :
        criterion = nn.L1Loss().cuda()
    elif args.loss_function == 'maemse' :
        criterion_mse = nn.MSELoss().cuda()
        criterion_mae = nn.L1Loss().cuda()



#####################################################################################

    # train


    best_iou = 0
    if args.train_mode:
        for epoch in range(lr_schedule[-1]):

            train(my_net, trn_loader, gen_optimizer, epoch, criterion, trn_logger, trn_raw_logger)
            loss = validate(val_loader, my_net, criterion, epoch, val_logger)
            print('validation : {} '.format(loss))

            lr_scheduler.step()
            checkpoint_filename = 'model_checkpoint.pth'
            save_checkpoint({'epoch': epoch + 1,
                             'state_dict': my_net.state_dict(),
                             'optimizer': gen_optimizer.state_dict()},
                             1, work_dir,
                            filename=checkpoint_filename)

    draw_curve(work_dir,trn_logger,val_logger)

    # validation
    if args.test_mode:
        print('Test mode ...')
        data_sets = dacon_test_loader(args.test_root)
        test_loader = torch.utils.data.DataLoader(data_sets, batch_size=1)


        csv_folder = os.path.join(args.work_dir,'sample_submission.csv')
        my_net1 = Unet2D(in_shape=(9, 40, 40), padding=1, momentum=0.1).cuda()
       #my_net1 = nn.DataParallel(my_net1).cuda()
        cudnn.benchmark = True

        my_net1.load_state_dict(torch.load(os.path.join(work_dir, 'model_best.pth'))['state_dict'])

        predict(test_loader, my_net1, args.test_root, csv_folder, work_dir)





def train(model,train_loader,optimizer,epoch, criterion,logger, sublogger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        input, target = input.float().cuda(), target.float().cuda()

        target_binary = target.clone()
        target_binary[target_binary>0] = 1

        assert len(np.unique(target_binary.cpu())) <= 2 ;print(len(np.unique(target_binary.cpu())))


        output = model(input)

        #import ipdb; ipdb.set_trace()
        loss = criterion(output, target)

        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses))

        if i % 10 == 0:
            sublogger.write([epoch, i, loss.item()])

    logger.write([epoch, losses.avg])

def validate(val_loader, model, criterion_mae, epoch, logger,save_fig=False,work_dir_name=False):

    batch_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.float().cuda()
            target = target.float().cuda()

            output = model(input)

            loss = criterion_mae(output, target)

            losses.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    logger.write([epoch, losses.avg])

    return losses.avg




def model_select(network):

    if network == 'unet':
        my_net = Unet2D(in_shape=(9, 40, 40), padding=1, momentum=0.1,start_channel=args.start_channel)
    elif network == 'unet_coord':
        my_net = Unetcoordconv(in_shape=(9, 40, 40), coordnumber=args.coordconv_no, radius=args.radious)
    else:
        raise ValueError('Not supported network.')

    model_name = network

    return my_net, model_name



#
# def predict(testloader, network, data_dir) :
#     id_list = []
#     value_list = []
#
#     with torch.no_grad():
#         for batch_idx, samples in enumerate(testloader):
#             temp_list = []
#             images, file_name = samples
#             images = images.float().cuda()
#             outputs = network(images)
#             outputs = torch.flatten(outputs, start_dim=1)
#             value_list.append(outputs)
#
#             for i in range(len(file_name)):
#                 file = file_name[i]
#                 file = file.split('_')
#                 file = file[1] + '_' + file[2]
#                 file = file.split('.')[0]
#                 temp_list.append(file)
#             id_list.append(temp_list)
#     value_arr = np.array([])
#
#     for i in range(len(value_list)):
#         value = value_list[i].cpu().numpy()
#         if len(value_arr) == 0:
#             value_arr = value
#         else:
#             value_arr = np.r_[value_arr, value]
#
#     df = pd.DataFrame(value_arr)
#     df['id'] = id_list
#     submit_file = pd.read_csv(data_dir+'sample_submission.csv')
#     columns = submit_file.columns
#     submit_file = submit_file[['id']]
#     submit_file = pd.merge(submit_file, df, on='id', how='left')
#     submit_file.columns = columns
#     submit_file.to_csv(data_dir+'/unique_first_result1.csv', index=False)


if __name__ == '__main__':
    main()
