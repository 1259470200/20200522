import torch
import os
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
from PIL import Image
import torchvision
import torch.nn as nn
from collections import OrderedDict

BASE_DIR = r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\dataset'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def compute_dice(y_pred, y_true):
    """
    :param y_pred: 4-d tensor, value=[0, 1]
    :param y_true: 4-d tensor, value=[0, 1]
    :return:
    """
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    y_pred, y_true = np.round(y_pred).astype(int), np.round(y_true).astype(int)
    return np.sum(y_pred[y_true == 1])*2/(np.sum(y_pred)+np.sum(y_true))


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features*2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features*2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features*4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features*8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block((features*8)*2, features*8, name='dec4')

        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name='dec3')

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name='dec2')

        self.upconv1 = nn.ConvTranspose2d(features * 2, features * 1, kernel_size=2, stride=2)
        self.decoder1 = UNet._block((features * 1) * 2, features * 1, name='dec1')

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + 'conv1',
                        nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
                    ),
                    (name + 'normal', nn.BatchNorm2d(num_features=features),),
                    (name + 'relu1', nn.ReLU(inplace=True),),
                    (
                        name + 'conv2',
                        nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
                    ),
                    (name + 'norma2', nn.BatchNorm2d(num_features=features),),
                    (name + 'relu2', nn.ReLU(inplace=True),),
                ]
            )
        )


class MyDataset(Dataset):
    # 需要自己写一个Dataset类，并且要继承从torch中import的Dataset基类，然后重写__len__和__getitem__两个方法，否则会报错
    # 此外还需要写__init__，传入数据所在路径和transform(用于数据预处理)
    def __init__(self, data_dir, transform=None):
        """
        :param data_dir: 读取的数据所在的路径
        :param transform: 数据预处理参数
        """
        self.data_info = self.dataInfo(data_dir)  # 用来读取数据信息(数据路径，标签)
        self.transform = transform

    def __getitem__(self, index):  # 根据索引读取数据路径再读取数据
        path_img, label_path = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        label = Image.open(label_path).convert('L')  # label是二值图像？不用转换吧，但是此处不转换后面的transforms不知道会不会出问题了

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)  # image做什么样的变换，那么label也做对应的变换
        else:  # 避免未作transforms而忘记把图像数据转化为tensor
            img = torch.tensor(img)
            label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def dataInfo(data_dir):  # 自定义函数用来获取数据信息，输入为数据所在路径，返回为一个元组(图像路径，标签路径)
        # 先读取所有的图像数据路径
        img_path = os.path.join(data_dir, 'images')
        imgs = os.listdir(img_path)
        # imgs.sort(key=lambda x: int(x.split('_')[0]))  # 根据图片标号从小到大排序
        label_path = os.path.join(data_dir, '1st_manual')
        labels = os.listdir(label_path)
        # labels.sort(key=lambda x: int(x.split('_')[0]))
        data_info = list()
        for i in range(len(imgs)):
            imgp = os.path.join(img_path, imgs[i])
            labelp = os.path.join(label_path, labels[i])
            data_info.append((imgp, labelp))
        return data_info


if __name__ == '__main__':
    # config
    lr = 0.01
    BATCH_SIZE = 1  # 由于cuda容量比较小，所以只能是batch_size为1，为2都gpu内存不够哦
    max_epoch = 10
    start_epoch = 0
    lr_step = 50
    val_interval = 3
    checkpoint_interval = 20
    vis_num = 10
    mask_thres = 0.5
    # train_dir = r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\dataset\DRIVE\training'
    # valid_dir = r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\dataset\DRIVE\test'
    train_dir = r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\dataset\CHASEDB1\training'
    valid_dir = r'C:\Users\Administrator\Desktop\2020spring\MedicalImageAnalysis\dataset\CHASEDB1\test'
    # 数据转换，这一步必不可少，必须得先转化为tensor才行
    transform_compose = transforms.Compose([
        # transforms.Resize((560, 560)),  # 此处resize要注意为8的倍数，不然后面拼接可能会出现问题
        transforms.Resize((960, 960)),  # 对于CHASEDB1数据集应该resize为960*960的大小，其他啥都不用变了，都是一样的
        transforms.ToTensor()
    ])

    # step1: prepare data
    train_set = MyDataset(data_dir=train_dir, transform=transform_compose)  # 需要重写Dataset
    valid_set = MyDataset(data_dir=valid_dir, transform=transform_compose)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=True, drop_last=False)

    # step2: model
    net = UNet(in_channels=3, out_channels=1, init_features=32)
    net.to(device)

    # step3: loss
    loss_fn = nn.MSELoss()

    # step4: optimize
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=0.1)

    # step5: iterate
    train_curve = []
    valid_curve = []
    train_dice_curve = []
    valid_dice_curve = []
    for epoch in range(start_epoch, max_epoch):
        train_loss_total = 0.
        train_dice_total = 0.
        net.train()
        for iter, (inputs, labels) in enumerate(train_loader):
            if torch.cuda.is_available:
                inputs, labels = inputs.to(device), labels.to(device)
            # forward
            outputs = net(inputs)
            # backward
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print
            train_dice = compute_dice(outputs.ge(mask_thres).cpu().data.numpy(), labels.cpu())
            train_dice_curve.append(train_dice)
            train_curve.append(loss.item())
            train_loss_total += loss.item()
            print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] running_loss: {:.4f}, mean_loss: {:.4f}"
                  "running_dice: {:.4f} lr:{}".format(epoch, max_epoch, iter+1, len(train_loader), loss.item(), train_loss_total/(iter+1), train_dice, scheduler.get_lr()))
        scheduler.step()

        if (epoch+1) % checkpoint_interval == 0:
            checkpoint = {"model_state_dict": net.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch}
            path_checkpoint = './checkpoint_{}_epoch.pkl'.format(epoch)
            torch.save(checkpoint, path_checkpoint)

        if (epoch+1) % val_interval == 0:
            net.eval()
            valid_loss_total = 0
            valid_dice_total = 0
            with torch.no_grad():
                for j, (inputs, labels) in enumerate(valid_loader):
                    if torch.cuda.is_available():
                        inputs, labels = inputs.to(device), labels.to(device)
                    outputs = net(inputs)
                    loss = loss_fn(outputs, labels)
                    valid_loss_total += loss.item()
                    valid_dice = compute_dice(outputs.ge(mask_thres).cpu().data, labels.cpu())
                    valid_dice_total += valid_dice
                valid_loss_mean = valid_loss_total/len(valid_loader)
                valid_dice_mean = valid_dice_total/len(valid_loader)
                valid_curve.append(valid_loss_mean)
                valid_dice_curve.append(valid_dice_mean)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] mean_loss: {:.4f} dice_mean: {:.4f}".format(epoch, max_epoch, valid_loss_mean, valid_dice_mean))
    # 可视化
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(valid_loader):
            if idx > vis_num:
                break
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            pred = outputs.ge(mask_thres)
            mask_pred = outputs.ge(0.5).cpu().data.numpy().astype('uint8')

            img_hwc = inputs.cpu().data.numpy()[0, :, :, :].transpose((1, 2, 0))
            label = labels.cpu().data.numpy()[0, :, :, :].transpose((1, 2, 0)).astype('uint8')
            plt.figure()
            plt.subplot(131).imshow(img_hwc), plt.title('input_image')
            label = label.squeeze()*255
            plt.subplot(132).imshow(label, cmap='gray'), plt.title('ground truth')
            # mask_pred_gray = np.transpose(mask_pred.squeeze()*255)
            mask_pred_gray = mask_pred.squeeze() * 255
            plt.subplot(133).imshow(mask_pred_gray, cmap='gray'), plt.title('predict')
            # plt.show()
            # plt.pause(1)
            # plt.close()




