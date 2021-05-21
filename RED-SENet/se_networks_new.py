import os
import numpy as np
import torch.nn as nn
import glob
import torch

def load_model(model_name, model_dir):  # vgg19  ./vgg
    model = eval('models.%s(init_weights=False)' % model_name)
    path_format = os.path.join(model_dir, '%s-[a-z0-9]*.pth' % model_name)
    print(path_format)
    model_path = glob.glob(path_format)[0]

    model.load_state_dict(torch.load(model_path))
    return model

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# nn.Conved是2D卷积层，而F.conv2d是2D卷积操作
class RED_CNN(nn.Module):
    def __init__(self, out_ch=96):
        super(RED_CNN, self).__init__()
        self.conv_first = nn.Conv2d(1, out_ch, kernel_size=5, stride=1, padding=0)  # 第一层卷积层
        self.conv1 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t1 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t2 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t3 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t4 = nn.ConvTranspose2d(out_ch, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv_t_last = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)  # 最后一层卷积层
        self.se1 = SELayer(out_ch)
        self.se2 = SELayer(out_ch)

        self.relu = nn.ReLU()  # 添加一层用ReLU函数作为激活层的激活函数
        """
        PyTorch-------前向传播函数forward
        神经网络的典型处理如下所示：
        1. 定义可学习参数的网络结构（堆叠各层和层的设计）；
        2. 数据集输入；
        3. 对输入进行处理（由定义的网络层进行处理）,主要体现在网络的前向传播；
        4. 计算loss ，由Loss层计算；
        5. 反向传播求梯度；
        6. 根据梯度改变参数值,最简单的实现方式（SGD）为:
           weight = weight - learning_rate * gradient
        """
    def forward(self, x):  # 编码器方向称为前向方向
        # encoder
        residual_1 = x
        out = self.relu(self.conv_first(x))
        out = self.relu(self.conv1(out))
        residual_2 = out
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        residual_3 = out
        out = self.relu(self.conv4(out))

        # decoder
        out = self.conv_t1(out)
        out = self.se1(out)
        out += residual_3
        out = self.conv_t2(self.relu(out))
        out = self.conv_t3(self.relu(out))
        out = self.se2(out)
        out += residual_2
        out = self.conv_t4(self.relu(out))
        out = self.conv_t_last(self.relu(out))
        out += residual_1
        out = self.relu(out)
        return out