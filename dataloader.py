import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from scipy.ndimage.interpolation import rotate


class dacon_test_loader(Dataset) : 
    def __init__(self, file_root) :
        self.file_root = file_root
        self.file_list = os.listdir(file_root)
        self.transforms1 = transforms.ToTensor()
        #self.transforms2 = transforms.Normalize([0.5], [0.5])

    def __len__(self) :
        return len(self.file_list)

    def __getitem__(self, idx) :
        image = np.load(self.file_root + self.file_list[idx])[:,:,:9]
        #image = self.transforms2(self.transforms1(image))
        image = self.transforms1(image)
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
        return image, self.file_list[idx]

class dacon_naive_loader(Dataset) :
    def __init__(self, file_root) :
        self.file_root = file_root
        self.file_list = os.listdir(file_root)
        self.transforms1 = transforms.ToTensor()
        #self.transforms2 = transforms.Normalize([0.5], [0.5])

    def __len__(self) :
        return len(self.file_list)

    def __getitem__(self, idx) :
        file = np.load(self.file_root + self.file_list[idx])
        try :
            label = file[:, :, -1].reshape(40, 40, 1)
            image = file[:, :, :9]
        except :
            print(file.shape)
            print(idx)
        label[label < 0] = 0
        #image = self.transforms2(image) #self.transforms1(image)
        image = self.transforms1(image)  # self.transforms1(image)
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
        label = self.transforms1(label)
        return image, label




class dacon_pre_processing_loader(Dataset) :
    def __init__(self, file_root) :
        self.file_root = file_root
        self.file_list = os.listdir(file_root)
        self.transforms1 = transforms.ToTensor()
        self.images, self.labels = self.get_files()

    def get_files(self) :
        images, labels = [], []
        for i in range(len(self.file_list)) :
            file = np.load(self.file_root + self.file_list[i])
            image = file[:, :, :9]
            label = file[:, :, -1].reshape(40, 40, 1)
            label[label < 0] = 0
            non_zero = len(np.where(label != 0)[0])
            if non_zero < 50 :
                pass
            else :
                images.append(image)
                labels.append(label)
        return images, labels

    def __len__(self) :
        return len(self.images)

    def __getitem__(self, idx) :
        image = self.transforms1(self.images[idx])  # self.transforms1(image)
        image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
        label = self.transforms1(self.labels[idx])
        return image, label




class dacon_add_rotation_loader(Dataset):
    def __init__(self, file_root):
        self.file_root = file_root
        self.file_list = os.listdir(file_root)
        self.transforms1 = transforms.ToTensor()
        self.images, self.labels = self.get_files()

    def get_files(self):
        images, labels = [], []
        for i in range(len(self.file_list)):
            file = np.load(self.file_root + self.file_list[i])
            image = file[:, :, :9]
            label = file[:, :, -1].reshape(40, 40, 1)
            label[label < 0] = 0
            non_zero = len(np.where(label != 0)[0])
            if non_zero < 50:
                pass
            else:
                images.append(image)
                labels.append(label)
        return images, labels

    def rotation(self, image, label) :
        random_num = np.random.randint(0, 3, 1)[0]
        angle = [90, 180, 270]
        image = rotate(image, angle[random_num])
        label = rotate(label, angle[random_num])
        return image, label

    def __len__(self):
        return len(self.images) * 2

    def __getitem__(self, idx):
        if idx < len(self.images):
            image = self.transforms1(self.images[idx])
            image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
            label = self.transforms1(self.labels[idx])
        elif idx >= len(self.images):
            np.random.seed(idx)
            rdx = np.random.randint(0, len(self.images), 1)[0]
            image = self.images[rdx]
            label = self.labels[rdx]


            image, label = self.rotation(image, label)
            image = self.transforms1(image)
            image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
            label = self.transforms1(label)

        return image, label

class dacon_naive_rotation_loader(Dataset):
    def __init__(self, file_root):
        self.file_root = file_root
        self.file_list = os.listdir(file_root)
        self.transforms1 = transforms.ToTensor()
        self.images, self.labels = self.get_files()

    def get_files(self):
        images, labels = [], []
        for i in range(len(self.file_list)):
            file = np.load(self.file_root + self.file_list[i])
            image = file[:, :, :9]
            label = file[:, :, -1].reshape(40, 40, 1)
            label[label < 0] = 0

            images.append(image)
            labels.append(label)

        return images, labels

    def rotation(self, image, label):
        random_num = np.random.randint(0, 3, 1)[0]
        angle = [90, 180, 270]
        image = rotate(image, angle[random_num])
        label = rotate(label, angle[random_num])
        return image, label

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        random_factor = random.random()

        if random_factor >0.5:

            image = self.images[idx]
            label = self.labels[idx]

            image, label = self.rotation(image, label)
            image = self.transforms1(image)
            image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
            label = self.transforms1(label)
        else:

            image = self.transforms1(self.images[idx])
            image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
            label = self.transforms1(self.labels[idx])

        return image, label
