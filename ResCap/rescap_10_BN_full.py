import torch
from torch import nn
from torch.optim import Adam, lr_scheduler
from torch.autograd import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F
from capsulelayers_speech_rescap import DenseCapsule, PrimaryCapsule
# from mfcc_load import load_mnist,read
from fbanks_load_10 import load_mnist,read


def judge_correct(label1,label2,pred1,pred2):
    correct1 = 0
    correct2 = 0
    for i in range(len(label1)):
        if label1[i] == pred1[i]:
            if label2[i] == pred2[i]:
                correct2 +=1
            else:
                correct1 +=1
        if label1[i] == pred2[i]:
            if label2[i] == pred1[i]:
                correct2 +=1
            else:
                correct1 +=1
    return correct1,correct2
    # correct = 0
    # for i in range(len(label1)):
    #     if label1[i] == pred1[i]:
    #         if label2[i] == pred2[i]:
    #             correct +=1
    #     if label1[i] == pred2[i]:
    #         if label2[i] == pred1[i]:
    #             correct +=1
    # return correct


def find_index(input_tensor):
    index1 = input_tensor.max(dim=1)[1]
    maxvalue = torch.zeros(len(index1))
    for i in range(len(index1)):
        maxvalue[i] = input_tensor[i][index1[i]]
        input_tensor[i][index1[i]]=0

    index2 = input_tensor.max(dim=1)[1]

    for i in range(len(index1)):
        input_tensor[i][index1[i]]=maxvalue[i]

    return index1,index2

class SpeechResModel(nn.Module):
    def __init__(self, input_size, classes, routings):
        super().__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings
        n_labels = 12
        n_maps = 45
        n_layers = 13
        n_pricap = 30
        self.conv0 = nn.Conv2d(1, n_maps, (3, 3), padding=(1, 1), bias=False)


        self.n_layers = n_layers
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
        # self.output = nn.Linear(n_maps, n_labels)


        self.convF = nn.Conv2d(n_maps, n_maps, kernel_size=(28,28), stride=2)
        self.batchF = nn.BatchNorm2d(n_maps,affine=False)
        self.primarycaps = PrimaryCapsule(n_maps)
        self.digitcaps = DenseCapsule(in_num_caps=36*17, in_dim_caps=n_maps,
                                      out_num_caps=classes, out_dim_caps=16, routings=routings)

        # self.convF = nn.Conv2d(n_maps, n_pricap, kernel_size=(3,3), stride=1, padding=0,dilation=16)
        # self.batchF = nn.BatchNorm2d(n_pricap,affine=False)
        # self.primarycaps = PrimaryCapsule(n_pricap)
        # self.capbatch = nn.BatchNorm2d(n_maps,affine=False)
        # self.digitcaps = DenseCapsule(in_num_caps=1848, in_dim_caps=n_pricap,
        #                               out_num_caps=classes, out_dim_caps=16, routings=routings)
        self.decoder = nn.Sequential(
            nn.Linear(16*classes, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.6),
            nn.Linear(2048, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )
        # self.dropout1 = nn.Dropout(0.6)

        # self.dropout2 = nn.Dropout2d(0.1)
        self.relu = nn.ReLU()

    def forward(self, x,y_p=None):
        # x = x.unsqueeze(1)
        for i in range(self.n_layers + 1):
            y = F.relu(getattr(self, "conv{}".format(i))(x))
            # print(i)
            if i == 0:
                old_x = y
            if i > 0 and i % 2 == 0:
                x = y + old_x
                old_x = x
            else:
                x = y
            if i > 0:
                x = getattr(self, "bn{}".format(i))(x)
        # print(x.size())
        # x = x.view(x.size(0), x.size(1), -1) # shape: (batch, feats, o3)
        # print(x.size())
        # x = torch.mean(x, 2)
        # print(x.size())
        x = self.convF(x)

        x = self.batchF(x)
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        # x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y_p is None:  # during testing, no label given. create one-hot coding using `length`
            index1, index2 = find_index(length)
            y1 = torch.zeros(length.size()).scatter_(1, index1.view(-1, 1).cpu().data, 1.)
            y2 = torch.zeros(length.size()).scatter_(1, index2.view(-1, 1).cpu().data, 1.)
            index =torch.add(y1,y2)
            y_p = Variable(index.cuda())


        reconstruction = self.decoder((x * y_p[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)


class CapsuleNet(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """
    def __init__(self, input_size, classes, routings):
        super(CapsuleNet, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.routings = routings

        # Layer 1: Just a conventional Conv2D layer
        self.conv1 = nn.Conv2d(input_size[0], 256, kernel_size=(6,24), stride=2, padding=0)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        self.primarycaps = PrimaryCapsule(256, 256, 32, kernel_size=(6,12), stride=(2,1), padding=0)

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=8*21*8, in_dim_caps=32,
                                      out_num_caps=classes, out_dim_caps=8, routings=routings)

        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(8*classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()




    def forward(self, x, y=None):
        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)
        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            index1, index2 = find_index(length)
            y1 = torch.zeros(length.size()).scatter_(1, index1.view(-1, 1).cpu().data, 1.)
            y2 = torch.zeros(length.size()).scatter_(1, index2.view(-1, 1).cpu().data, 1.)
            index =torch.add(y1,y2)
            y = Variable(index.cuda())
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, *self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2 + \
        0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()

    L_recon = nn.MSELoss()(x_recon, x)
    # L_recon = torch.sum(torch.sqrt(torch.pow(torch.div(x_recon,x)-torch.log_(torch.div(x_recon,x))-1,2)))/588000
    # print (torch.div(x_recon,x))
    # L_recon = torch.log_(torch.div(x,x_recon)-torch.log_(torch.div(x,x_recon))-1).mean()

    return L_margin + lam_recon * L_recon




def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct1 = 0
    correct2 = 0
    for x, y, z in test_loader:

        x = x.float()
        y = y.long()
        z = z.long()
        # x = x/256
        x = x.view_as(torch.Tensor(args.batch_size,1,98,60))

        z_hot = torch.zeros(z.size(0), 10).scatter_(1, z.view(-1, 1), 1.)
        y_hot = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)
        true_label = torch.add(y_hot,z_hot)
        y, z = Variable(y.float().cuda()), Variable(z.float().cuda())

        x, true_label = Variable(x.cuda(), volatile=True), Variable(true_label.cuda())
        y_pred, x_recon = model(x)
        test_loss += caps_loss(true_label, y_pred, x, x_recon, args.lam_recon).data[0] * x.size(0)  # sum up batch loss
        # y_pred = y_pred.data.max(1)[1]
        # y_true = y.data.max(1)[1]

        index_pred1,index_pred2 = find_index(y_pred)
        # pred1 = torch.zeros(y_pred.size()).scatter_(1, index_pred1.view(-1, 1).cpu().data, 1.)
        # pred2 = torch.zeros(y_pred.size()).scatter_(1, index_pred2.view(-1, 1).cpu().data, 1.)
        # pred_label = torch.add(pred1,pred2)

        result1, result2 = judge_correct(y,z,index_pred1.float(),index_pred2.float())
        correct1 +=result1
        correct2 +=result2
        # correct += 200 - (1000 - pred_label.eq(true_label.cpu()).float().cpu().sum())/2

    test_loss /= len(test_loader.dataset)
    correct1   /= len(test_loader.dataset)
    correct2   /= len(test_loader.dataset)

    return test_loss, correct1, correct2

def train(model, train_loader, test_loader, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print('Begin Training' + '-'*70)
    from time import time
    import csv
    logfile = open(args.save_dir + '/log.csv', 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)
    best_val_acc2 = 0.
    for epoch in range(args.epochs):
        model.train()  # set to training mode
        lr_decay.step()  # decrease the learning rate by multiplying a factor `gamma`
        ti = time()
        training_loss = 0.0
        for i, (x, y, z) in enumerate(train_loader):  # batch training\\

            x = x.float()
            y = y.long()
            z = z.long()
            # x = x/256
            x = x.view_as(torch.Tensor(args.batch_size,1,98,60))

            y = torch.zeros(y.size(0), 10).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
            z = torch.zeros(z.size(0), 10).scatter_(1, z.view(-1, 1), 1.)
            y = torch.add(y,z)

            x, y = Variable(x.cuda()), Variable(y.cuda())  # convert input data to GPU Variable

            optimizer.zero_grad()  # set gradients of optimizer to zero
            y_pred, x_recon = model(x, y)  # forward
            loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.data[0] * x.size(0)  # record the batch loss
            optimizer.step()  # update the trainable parameters with computed gradients

        # compute validation loss and acc
        val_loss, val_acc1,val_acc2 = test(model, test_loader, args)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
                                val_loss=val_loss, val_acc=val_acc2))
        print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc1=%.4f,val_acc2=%.4f, time=%ds"
              % (epoch, training_loss / len(train_loader.dataset),
                 val_loss, val_acc1, val_acc2, time() - ti))
        if val_acc2 > best_val_acc2:  # update best validation acc and save model
            best_val_acc2 = val_acc2
            torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
            print("best val_acc2 increased to %.4f" % best_val_acc2)
    logfile.close()
    torch.save(model.state_dict(), args.save_dir + '/trained_model.pkl')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model

#
# def load_mnist(path='./data', download=False, batch_size=100, shift_pixels=2):
#     """
#     Construct dataloaders for training and test data. Data augmentation is also done here.
#     :param path: file path of the dataset
#     :param download: whether to download the original data
#     :param batch_size: batch size
#     :param shift_pixels: maximum number of pixels to shift in each direction
#     :return: train_loader, test_loader
#     """
#     kwargs = {'num_workers': 1, 'pin_memory': True}
#
#     train_loader = torch.utils.data.DataLoader(
#         datasets.MNIST(path, train=True, download=download,
#                        transform=transforms.Compose([transforms.RandomCrop(size=28, padding=shift_pixels),
#                                                      transforms.ToTensor()])),
#         batch_size=batch_size, shuffle=True, **kwargs)
#
#     test_loader = torch.utils.data.DataLoader(
#         datasets.MNIST(path, train=False, download=download,
#                        transform=transforms.ToTensor()),
#         batch_size=batch_size, shuffle=True, **kwargs)
#
#     return train_loader, test_loader


if __name__ == "__main__":
    import argparse
    import os

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")

    parser.add_argument('--epochs', default=40, type=int)

    parser.add_argument('--batch_size', default=50, type=int)

    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")

    parser.add_argument('--lr_decay', default=0.85, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")

    parser.add_argument('--lam_recon', default=0.02*784, type=float,
                        help="The coefficient for the loss of decoder")

    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0

    parser.add_argument('--shift_pixels', default=2, type=int,
                        help="Number of pixels to shift at most in each direction.")

    parser.add_argument('--data_dir', default='./data',
                        help="Directory of data. If no data, use \'--download\' flag to download it")

    parser.add_argument('--download', action='store_true',
                        help="Download the required data.")

    parser.add_argument('--save_dir', default='./result')

    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")

    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")

    args = parser.parse_args()

    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load data
    train_list,test_list = read(2)

    # train_list = torch.LongTensor(train_list)
    # test_list = torch.LongTensor(test_list)
    train_loader, test_loader = load_mnist(train_list,test_list, batch_size=args.batch_size)
    print('Data loading finish')
    # define model
    model = SpeechResModel(input_size=[1, 98, 60], classes=10, routings=3)
    model = torch.nn.DataParallel(model)
    model.cuda(1)
    print(model)

    # train or test
    if args.weights is not None:  # init the model weights with provided one
        model.load_state_dict(torch.load(args.weights))

    if not args.testing:
        train(model, train_loader, test_loader, args)

    else:  # testing
        if args.weights is None:
            print('No weights are provided. Will test using random initialized weights.')

        val_loss, val_acc1,val_acc2 = test(model, test_loader, args)
        print('test acc1 = %.5f,test acc2 = %.5f, test loss = %.5f' % (val_acc1,val_acc2, val_loss))
        # show_reconstruction(model, test_loader, 50, args)
