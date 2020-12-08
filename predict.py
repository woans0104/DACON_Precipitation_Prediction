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

def predict(testloader, network, data_dir, csv_dir, work_dir) :
    id_list = []
    value_list = []

    from tqdm import tqdm

    network.eval()

    with torch.no_grad():
        for i, samples in tqdm(enumerate(testloader)):
            temp_list = []  
            images, file_name = samples
            images = images.float().cuda()
            outputs = network(images)
            outputs = torch.flatten(outputs, start_dim=1)
            value_list.append(outputs)

            for i in range(len(file_name)):
                file = file_name[i]
                file = file.split('_')
                file = file[1] + '_' + file[2]
                file = file.split('.')[0]
                temp_list.append(file)
            id_list += (temp_list)
    value_arr = np.array([])

    for i in range(len(value_list)):
        value = value_list[i].cpu().numpy()
        if len(value_arr) == 0:
            value_arr = value
        else:
            value_arr = np.r_[value_arr, value]
    df = pd.DataFrame(value_arr)
    df['id'] = id_list
    submit_file = pd.read_csv(csv_dir)
    columns = submit_file.columns
    submit_file = submit_file[['id']]
    submit_file = pd.merge(submit_file, df, on='id', how='left')
    submit_file.columns = columns
    submit_file.to_csv(work_dir+'/unique_first_result2.csv', index=False)


if __name__=='__main__':

    test_root = '/data2/woans0104/dacon_rain_dataset/test/'
    data_sets = dacon_test_loader(test_root)
    test_loader = torch.utils.data.DataLoader(data_sets, batch_size=1)
    work_dir = '/data1/JM/dacon_rain/Dacon_exp/Baseline_unet_bach512_sch64'
    csv_folder = '/data1/JM/dacon_rain/sample_submission.csv'

    my_net = Unet2D(in_shape=(9, 40, 40), padding=1, momentum=0.1,start_channel=64).cuda()
    #my_net = convert_model(my_net)
    #my_net = nn.DataParallel(my_net).cuda()
    cudnn.benchmark = True

    save_model = my_net.load_state_dict(torch.load(os.path.join(work_dir, 'model_best.pth'))['state_dict'])

    predict(test_loader, my_net, test_root, csv_folder, work_dir)