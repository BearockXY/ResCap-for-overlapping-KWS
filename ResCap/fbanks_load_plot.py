import os
import struct
import numpy as np
import torch
from torchvision import transforms, datasets

def read(label):
    # test_data = np.load('home/bearock/Desktop/MultiMnist_V2/data/multimnist/multimnist_test_5_py3.npy')
    # train_data = np.load('home/bearock/Desktop/MultiMnist_V2/data/multimnist/multimnist_train_1_py3.npy')
    if label ==1:
        test_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/new/fb_visual_six_plot.npy',encoding = 'bytes')
        train_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/new/fb_visual_six_plot.npy',encoding = 'bytes')
        print('using speaker dependent')
    if label ==2:
        test_data = np.load('/home/bearock/Desktop/MultiMnist_V2/data/word_mix/fb_features_test_spkinde.npy',encoding = 'bytes')
        train_data = np.load('/home/bearock/Desktop/MultiMnist_V2/data/word_mix/fb_features_train_spkinde.npy',encoding = 'bytes')
        print('using speaker independent')

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    # test_data = np.load('multimnist_test.npy')

    train_out = []
    train_size = 50000
    test_size = 10000


    train_tuple = (train_data[0][0].astype('float')/(train_data[0][0].max()-train_data[0][0].min()),train_data[0][1],train_data[0][2])
    train_out.append(train_tuple)

    test_out = []


    test_tuple = (test_data[0][0].astype('float')/(test_data[0][0].max()-test_data[0][0].min()),test_data[0][1],test_data[0][2])
    test_out.append(test_tuple)
    print('using filter banks, train size',len(train_out),'test size', len(test_out))
    return train_out, test_out

def load_mnist(train_mnistdata,test_mnistdata,batch_size=100):
    """
    Construct dataloaders for training and test data. Data augmentation is also done here.
    :param path: file path of the dataset
    :param download: whether to download the original data
    :param batch_size: batch size
    :param shift_pixels: maximum number of pixels to shift in each direction
    :return: train_loader, test_loader
    """
    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_mnistdata,
        batch_size=batch_size, shuffle=True,drop_last =True, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_mnistdata,
        batch_size=batch_size, shuffle=True,drop_last =True, **kwargs)


    return train_loader, test_loader
