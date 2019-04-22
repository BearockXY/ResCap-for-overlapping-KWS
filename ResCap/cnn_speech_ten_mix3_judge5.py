from __future__ import print_function
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam, lr_scheduler
from torchvision import datasets, transforms
# from multimnist_load import load_mnist,read
from torch.autograd import Variable
from time import time
# from mfcc_load import load_mnist,read
from fbanks_load_10_mix3 import load_mnist,read

def order3(input1,input2,input3):
    if input1>input2:
        if input1>input3:
            tag1 = input1
            if input2>input3:
                tag2 = input2
                tag3 = input3
            else:
                tag2 = input3
                tag3 = input2
        else:
            tag2 = input1
            tag3 = input2
            tag1 = input3
    else:
        if input1<input3:
            tag3 = input1
            if input2<input3:
                tag1= input3
                tag2 = input2
            else:
                tag1 = input2
                tag2 = input3
        else:
            tag1 = input2
            tag2 = input1
            tag3 = input3
    return tag1,tag2,tag3

def find_index(input_tensor):
    index1 = input_tensor.max(dim=1)[1]
    maxvalue1 = torch.zeros(len(index1))
    for i in range(len(index1)):
        maxvalue1[i] = input_tensor[i][index1[i]]
        input_tensor[i][index1[i]]=0

    index2 = input_tensor.max(dim=1)[1]
    maxvalue2 = torch.zeros(len(index2))
    for i in range(len(index2)):
        maxvalue2[i] = input_tensor[i][index2[i]]
        input_tensor[i][index2[i]]=0

    index3 = input_tensor.max(dim=1)[1]
    maxvalue3 = torch.zeros(len(index3))
    for i in range(len(index3)):
        maxvalue3[i] = input_tensor[i][index3[i]]
        input_tensor[i][index3[i]]=0

    index4 = input_tensor.max(dim=1)[1]
    maxvalue4 = torch.zeros(len(index4))
    for i in range(len(index4)):
        maxvalue4[i] = input_tensor[i][index4[i]]
        input_tensor[i][index4[i]]=0

    index5 = input_tensor.max(dim=1)[1]

    for i in range(len(index1)):
        input_tensor[i][index1[i]]=maxvalue1[i]
    for i in range(len(index2)):
        input_tensor[i][index2[i]]=maxvalue2[i]
    for i in range(len(index3)):
        input_tensor[i][index3[i]]=maxvalue3[i]
    for i in range(len(index4)):
        input_tensor[i][index4[i]]=maxvalue4[i]

    return index1,index2,index3,index4,index5

def judge_correct(label1,label2,label3,pred1,pred2,pred3,pred4,pred5):
    label1 = label1.long()
    label2 = label2.long()
    label3 = label3.long()
    pred1 = pred1.long()
    pred2 = pred2.long()
    pred3 = pred3.long()
    pred4 = pred4.long()
    pred5 = pred5.long()
    correct1 = 0
    correct2 = 0
    correct3 = 0

    # Miss = np.zeros([12,12])

    for i in range(len(label1)):

        # tag1,tag2,tag3 = order3(pred1[i],pred2[i],pred3[i])

        tag1 = pred1[i]
        tag2 = pred2[i]
        tag3 = pred3[i]
        tag4 = pred4[i]
        tag5 = pred5[i]
        # print(tag1,tag2,tag3)
        ans1,ans2,ans3 = order3(label1[i],label2[i],label3[i])
        correct_count = 0
        # Miss[tag1][ans1] +=1
        # Miss[tag2][ans2] +=1
        # Miss[tag3][ans3] +=1
        # print(ans1,ans2,ans3)
        if ans1 == tag1:
            correct_count += 1
        elif ans1 == tag2:
            correct_count += 1
        elif ans1 == tag3:
            correct_count += 1
        # elif ans1 == tag4:
        #     correct_count += 1
        # elif ans1 == tag5:
        #     correct_count += 1

        if ans2 == tag1:
            correct_count += 1
        elif ans2 == tag2:
            correct_count += 1
        elif ans2 == tag3:
            correct_count += 1
        # elif ans2 == tag4:
        #     correct_count += 1
        # elif ans2 == tag5:
        #     correct_count += 1

        if ans3 == tag1:
            correct_count += 1
        elif ans3 == tag2:
            correct_count += 1
        elif ans3 == tag3:
            correct_count += 1
        # elif ans3 == tag4:
        #     correct_count += 1
        # elif ans3 == tag5:
        #     correct_count += 1
        # print(correct_count)
        if correct_count == 3:
            correct3 +=1
        elif correct_count ==2:
            correct2 +=1
        elif correct_count==1:
            correct1 +=1






        # if label1[i] == pred1[i]:
        #     Miss[label1[i]][pred1[i]] +=1
        #     if label2[i] == pred2[i]:
        #         correct2 +=1
        #         Miss[label2[i]][pred2[i]] +=1
        #         if label3[i] ==pred3[i]:
        #             correct3 +=1
        #             Miss[label3[i]][pred3[i]] +=1
        #
        #     else:
        #         correct1 +=1
        #         Miss[label2[i]][pred2[i]] +=1
        #
        #
        #
        #
        # elif label1[i] == pred2[i]:
        #     Miss[label1[i]][pred2[i]] +=1
        #     if label2[i] == pred1[i]:
        #         correct2 +=1
        #         Miss[label2[i]][pred1[i]] +=1
        #     else:
        #         correct1 +=1
        #         Miss[label2[i]][pred1[i]] +=1
        #
        # else:
        #     Miss[label1[i]][pred1[i]] +=1
        #     Miss[label2[i]][pred2[i]] +=1


    return correct1,correct2,correct3

def Find_correct(inputdata,target1,target2,target3):
    correct1= 0
    correct2= 0
    correct3 = 0
    index_pred1,index_pred2,index_pred3,index_pred4,index_pred5 = find_index(inputdata)
    # y,z = Variable(target1.float().cuda()), Variable(target2.float().cuda())
    y,z,w = Variable(target1.float()), Variable(target2.float()), Variable(target3.float())

    result1, result2,result3 = judge_correct(y,z,w,index_pred1.float(),index_pred2.float(),index_pred3.float(),index_pred4.float(),index_pred5.float())

    correct1 += result1
    correct2 += result2
    correct3 += result3

    return correct1, correct2,correct3

def CELoss(inputdata, target1,target2):
    criterion = nn.CrossEntropyLoss()
    loss1 = criterion(inputdata,target1)
    loss2 = criterion(inputdata,target2)
    return loss1 + loss2

class SpeechResModel(nn.Module):
    n_labels=12
    use_dilation=True
    n_layers=13
    n_feature_maps=45
    def __init__(self):
        super().__init__()
        n_labels = 12
        n_maps = 45
        n_layers = 13
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)


        self.n_layers = 13
        dilation = True
        if dilation:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=int(2**(i // 3)), dilation=int(2**(i // 3)),
                bias=False) for i in range(n_layers)]
        else:
            self.convs = [nn.Conv2d(n_maps, n_maps, (3, 3), padding=1, dilation=1,
                bias=False) for _ in range(n_layers)]
        for i, conv in enumerate(self.convs):
            self.add_module("bn{}".format(i + 1), nn.BatchNorm2d(n_maps, affine=False))
            self.add_module("conv{}".format(i + 1), conv)
        # self.convF = nn.Conv2d(n_maps,n_maps,kernel_size=(3,3),stride=1,dilation=16)
        # self.convF = nn.Conv2d(n_maps,n_maps,kernel_size=(28,28),stride=2)
        self.output = nn.Linear(n_maps, n_labels)
        # self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        # x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            if i == 0:
                if hasattr(self, "pool"):
                    y = self.pool(y)
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        # x = self.convF(x)
        x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        x = torch.mean(x, 2)
        return self.output(x)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(24,6),stride=2)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(12,6),stride=(1,2))
        self.conv3 = nn.Conv2d(256, 128, kernel_size=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(128*12, 328)
        self.fc2 = nn.Linear(328, 192)
        self.fc3 = nn.Linear(192,11)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 128*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(720, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 720)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader,test_loader, optimizer, epoch,ti):

    model.train()
    train_loss = 0
    for batch_idx, (data, target1,target2) in enumerate(train_loader):

        data = data.view_as(torch.Tensor(args.batch_size,1,98,60)).float()
        data, target1,target2 = data.to(device), target1.to(device),target2.to(device)
        optimizer.zero_grad()

        output = model(data)

        loss = CELoss(output, target1,target2)

        loss.backward()
        train_loss +=loss*data.size(0)
        optimizer.step()
        # if batch_idx % args.log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    # print (train_loss)
    test(args, model,device,train_loss,test_loader,epoch,ti)
    torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)



def test(args, model,device,train_loss,test_loader,epoch,ti):
    model.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    correct3 = 0
    MissMat = np.zeros([12,12])
    with torch.no_grad():
        for data,target1,target2,target3 in test_loader:
            # target = tag(target1,target2)
            data = data.view_as(torch.Tensor(args.batch_size,1,98,60)).float()
            data, target1,target2,target3 = data.to(device), target1.to(device),target2.to(device),target3.to(device)

            output = model(data)
            test_loss += CELoss(output,target1,target2)*data.size(0) # sum up batch loss
            # pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            # correct += pred.eq(target.view_as(pred)).sum().item()
            result1,result2,result3 = Find_correct(output,target1,target2,target3)
            # MissMat += Miss
            correct1 += result1
            correct2 += result2
            correct3 += result3

    test_loss /= len(test_loader.dataset)
    # print(MissMat)
    # loss /=
    print('epoch:{:.0f},train_loss:{:.5f},Test set: Average loss: {:.5f}, Accuracy 1: {}/{} ({:.2f}%),Accuracy 2: {}/{} ({:.2f}%),Accuracy 3: {}/{} ({:.2f}%),Time is {:.4f}\n'.format(
        epoch, train_loss,test_loss,
        correct1, len(test_loader.dataset),100. * correct1 / len(test_loader.dataset),
        correct2, len(test_loader.dataset), 100. * correct2 / len(test_loader.dataset),
        correct3, len(test_loader.dataset), 100. * correct3 / len(test_loader.dataset),
        time()-ti))

def main():
    import os
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=26, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_dir', default='./result')

    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")

    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()

    # print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('./data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    train_list,test_list = read(3)
    train_loader, test_loader = load_mnist(train_list,test_list, batch_size=args.batch_size)


    # model = Net().to(device)
    model = SpeechResModel().to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.00001, momentum=0.9)



    if args.weights is not None:  # init the model weights with provided one
        model.load_state_dict(torch.load(args.weights))

    if not args.testing:
        print('training start')
        for epoch in range(1, args.epochs + 1):
            ti = time()

            train(args, model, device, train_loader,test_loader, optimizer, epoch,ti)
        # train(model, train_loader, test_loader, args)

    else:  # testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')

        train_loss = 0
        epoch = 0
        ti =0

        test(args, model,device,train_loss,test_loader,epoch,ti)
        # print('test acc = %.5f, test loss = %.5f' % (test_acc, test_loss))
        # show_reconstruction(model, test_loader, 50, args)


if __name__ == '__main__':
    main()
