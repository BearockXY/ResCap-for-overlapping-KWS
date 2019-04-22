import os
import struct
import numpy as np
import torch
from torchvision import transforms, datasets

def read(label):
    # test_data = np.load('home/bearock/Desktop/MultiMnist_V2/data/multimnist/multimnist_test_5_py3.npy')
    # train_data = np.load('home/bearock/Desktop/MultiMnist_V2/data/multimnist/multimnist_train_1_py3.npy')
    # if label ==1:
    #     test_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/old/fb_features_ten_test_spkde.npy',encoding = 'bytes')
    #     train_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/old/fb_features_ten_train_spkinde.npy',encoding = 'bytes')
    #     print('using speaker independent')
    # if label ==2:
    #     test_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/old/fb_features_ten_test_spkinde.npy',encoding = 'bytes')
    #     train_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/old/fb_features_ten_train_spkinde.npy',encoding = 'bytes')
    #     print('using speaker dependent')
    if label ==1:
        test_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/test/fb_features_ten_test_spkind.npy',encoding = 'bytes')
        train_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/test/fb_features_ten_train_spkind.npy',encoding = 'bytes')
        print('using speaker independent')
    if label ==2:
        test_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/test/fb_features_ten_test_spkdep.npy',encoding = 'bytes')
        train_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/test/fb_features_ten_train_spkdep.npy',encoding = 'bytes')
        print('using speaker dependent')
    if label ==3:
        test_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/new/fb_features_ten3_test_spkdep.npy',encoding = 'bytes')
        train_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/new/fb_features_ten_train_spkdep.npy',encoding = 'bytes')
        print('using mix3 speaker dependent')
    if label ==4:
        test_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/new/fb_features_ten3_test_spkind.npy',encoding = 'bytes')
        train_data = np.load('/home/xps15/Desktop/capsulenet/data/word_mix/new/fb_features_ten_train_spkdep.npy',encoding = 'bytes')
        print('using mix3 speaker independent')

    np.random.shuffle(train_data)
    np.random.shuffle(test_data)
    # test_data = np.load('multimnist_test.npy')

    train_out = []
    train_size = 54000
    test_size = 36306
    # print(len(train_data))
    # train_size = len(train_data)
    # test_size = len(test_data)
    # train_size = 500
    # test_size = 100
    for i in range(train_size):
        # if train_data[i][0].shape[0] ==98:
        train_tuple = (train_data[i][0].astype('float')/(train_data[i][0].max()-train_data[i][0].min()),train_data[i][1]-1,train_data[i][2]-1)
        train_out.append(train_tuple)
    print('finish training load')
    test_out = []
    for i in range(test_size):
        # if test_data[i][0].shape[0] ==98:
        test_tuple = (test_data[i][0].astype('float')/(test_data[i][0].max()-test_data[i][0].min()),test_data[i][1]-1,test_data[i][2]-1,test_data[i][3]-1)
        test_out.append(test_tuple)
    print('finish testing load')
    print('using filter banks ten, train size',len(train_out),'test size', len(test_out))
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
