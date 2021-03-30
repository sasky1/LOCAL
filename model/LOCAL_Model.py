import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import math
from collections import OrderedDict
import torch.nn.modules
torch.cuda.manual_seed_all(1)
torch.manual_seed(1)

class LOCAL_Net(nn.Module):
    def __init__(self):
        super(LOCAL_Net, self).__init__()

        self.conv1_1_a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2_a = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1_a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2_a = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1_a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2_a = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3_a = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1_a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2_a = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3_a = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1_a = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2_a = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3_a = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv1_1_b = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2_b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1_b = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2_b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1_b = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2_b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3_b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1_b = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2_b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3_b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1_b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2_b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3_b = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        ## -------------Refine Module-------------

        ## attention 5
        self.attention_decoder5_a = nn.Sequential(nn.Conv2d(512, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(512, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(512, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.Sigmoid())
        self.attention_decoder5_b = nn.Sequential(nn.Conv2d(512, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(512, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(512, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.Sigmoid())


        self.mask5_o = nn.Sequential(nn.Conv2d(512 * 4, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 2, 3, padding=1),
                                     nn.Sigmoid())

        ## attention 4

        self.attention_decoder4_a = nn.Sequential(nn.Conv2d(512 + 1, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(512, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(512, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.Sigmoid())
        self.attention_decoder4_b = nn.Sequential(nn.Conv2d(512 + 1, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(512, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(512, 512, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(512),
                                                  nn.Sigmoid())


        self.mask4_o = nn.Sequential(nn.Conv2d(512 * 4 + 2, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 2, 3, padding=1),
                                     nn.Sigmoid())

        ## attention 3

        self.attention_decoder3_a = nn.Sequential(nn.Conv2d(256 + 1, 256, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(256),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(256, 256, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(256),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(256, 256, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(256),
                                                  nn.Sigmoid())
        self.attention_decoder3_b = nn.Sequential(nn.Conv2d(256 + 1, 256, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(256),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(256, 256, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(256),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(256, 256, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(256),
                                                  nn.Sigmoid())


        self.mask3_o = nn.Sequential(nn.Conv2d(256 * 4 + 2, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 2, 3, padding=1),
                                     nn.Sigmoid())

        ## attention 2

        self.attention_decoder2_a = nn.Sequential(nn.Conv2d(128 + 1, 128, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(128),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(128, 128, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(128),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(128, 128, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(128),
                                                  nn.Sigmoid())
        self.attention_decoder2_b = nn.Sequential(nn.Conv2d(128 + 1, 128, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(128),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(128, 128, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(128),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(128, 128, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(128),
                                                  nn.Sigmoid())


        self.mask2_o = nn.Sequential(nn.Conv2d(128 * 4 + 2, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 2, 3, padding=1),
                                     nn.Sigmoid())

        ## attention 1

        self.attention_decoder1_a = nn.Sequential(nn.Conv2d(64 + 1, 64, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(64),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 64, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(64),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 64, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(64),
                                                  nn.Sigmoid())
        self.attention_decoder1_b = nn.Sequential(nn.Conv2d(64 + 1, 64, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(64),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 64, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(64),
                                                  nn.ReLU(inplace=True),
                                                  nn.Conv2d(64, 64, 3, dilation=2, padding=2),
                                                  nn.BatchNorm2d(64),
                                                  nn.Sigmoid())


        self.mask1_o = nn.Sequential(nn.Conv2d(64 * 4 + 2, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 128, 3, padding=1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(128, 2, 3, padding=1),
                                     nn.Sigmoid())

        self.upscore2 = nn.Upsample(scale_factor=2, mode='bilinear')

        vgg16 = models.vgg16(pretrained=True)
        weights = vgg16.state_dict()
        weights_index = ['features.0.weight', 'features.2.weight', 'features.5.weight', 'features.7.weight',
                         'features.10.weight', 'features.12.weight', 'features.14.weight', 'features.17.weight',
                         'features.19.weight', 'features.21.weight', 'features.24.weight', 'features.26.weight',
                         'features.28.weight']

        index = 0

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if index < 13:
                    m.weight.data = weights[weights_index[index]]
                    index += 1
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, input):
        hx = input
        # -------------Encoder------------
        x = F.relu(self.conv1_1_a(input))
        conv1_2 = F.relu(self.conv1_2_a(x))
        h1_a = F.max_pool2d(conv1_2, kernel_size=2, stride=2)

        x = F.relu(self.conv2_1_a(h1_a))
        conv2_2 = F.relu(self.conv2_2_a(x))
        h2_a = F.max_pool2d(conv2_2, kernel_size=2, stride=2)

        x = F.relu(self.conv3_1_a(h2_a))
        x = F.relu(self.conv3_2_a(x))
        conv3_3 = F.relu(self.conv3_3_a(x))
        h3_a = F.max_pool2d(conv3_3, kernel_size=2, stride=2)

        x = F.relu(self.conv4_1_a(h3_a))
        x = F.relu(self.conv4_2_a(x))
        conv4_3 = F.relu(self.conv4_3_a(x))
        h4_a = F.max_pool2d(conv4_3, kernel_size=3, stride=2, padding=1)

        x = F.relu(self.conv5_1_a(h4_a))
        x = F.relu(self.conv5_2_a(x))
        conv5_3 = F.relu(self.conv5_3_a(x))
        h5_a = F.max_pool2d(conv5_3, kernel_size=3, stride=2, padding=1)

        x = F.relu(self.conv1_1_b(input))
        conv1_2 = F.relu(self.conv1_2_b(x))
        h1_b = F.max_pool2d(conv1_2, kernel_size=2, stride=2)

        x = F.relu(self.conv2_1_b(h1_b))
        conv2_2 = F.relu(self.conv2_2_b(x))
        h2_b = F.max_pool2d(conv2_2, kernel_size=2, stride=2)

        x = F.relu(self.conv3_1_b(h2_b))
        x = F.relu(self.conv3_2_b(x))
        conv3_3 = F.relu(self.conv3_3_b(x))
        h3_b = F.max_pool2d(conv3_3, kernel_size=2, stride=2)

        x = F.relu(self.conv4_1_b(h3_b))
        x = F.relu(self.conv4_2_b(x))
        conv4_3 = F.relu(self.conv4_3_b(x))
        h4_b = F.max_pool2d(conv4_3, kernel_size=3, stride=2, padding=1)

        x = F.relu(self.conv5_1_b(h4_b))
        x = F.relu(self.conv5_2_b(x))
        conv5_3 = F.relu(self.conv5_3_b(x))
        h5_b = F.max_pool2d(conv5_3, kernel_size=3, stride=2, padding=1)

        M5_a = self.upscore2(h5_a)
        M5_b = self.upscore2(h5_b)

        att_a_ = self.attention_decoder5_a(M5_a)
        att_b_ = self.attention_decoder5_b(M5_b)

        M5_a = torch.cat((1 * att_b_.mul(M5_a), M5_a), dim=1)

        M5_b = torch.cat((1 * att_a_.mul(M5_b), M5_b), dim=1)

        M5 = self.mask5_o(torch.cat((M5_a, M5_b), dim=1))  # 16

        M4_a = self.upscore2(h4_a)
        M4_b = self.upscore2(h4_b)

        att_a_ = self.attention_decoder4_a(torch.cat((self.upscore2(M5[:, 0, :, :].unsqueeze(1)), M4_a), dim=1))
        att_b_ = self.attention_decoder4_b(torch.cat((self.upscore2(M5[:, 1, :, :].unsqueeze(1)), M4_b), dim=1))

        M4_a = torch.cat((self.upscore2(M5[:, 0, :, :].unsqueeze(1)), M4_a, 1 * att_b_.mul(M4_a)), dim=1)
        M4_b = torch.cat((self.upscore2(M5[:, 1, :, :].unsqueeze(1)), M4_b, 1 * att_a_.mul(M4_b)), dim=1)

        M4 = self.mask4_o(torch.cat((M4_a, M4_b), dim=1))  # 8

        M3_a = self.upscore2(h3_a)
        M3_b = self.upscore2(h3_b)

        att_a_ = self.attention_decoder3_a(torch.cat((self.upscore2(M4[:, 0, :, :].unsqueeze(1)), M3_a), dim=1))
        att_b_ = self.attention_decoder3_b(torch.cat((self.upscore2(M4[:, 1, :, :].unsqueeze(1)), M3_b), dim=1))

        M3_a = torch.cat((self.upscore2(M4[:, 0, :, :].unsqueeze(1)), M3_a, 1 * att_b_.mul(M3_a)), dim=1)
        M3_b = torch.cat((self.upscore2(M4[:, 1, :, :].unsqueeze(1)), M3_b, 1 * att_a_.mul(M3_b)), dim=1)

        M3 = self.mask3_o(torch.cat((M3_a, M3_b), dim=1))  # 4

        M2_a = self.upscore2(h2_a)
        M2_b = self.upscore2(h2_b)

        att_a_ = self.attention_decoder2_a(torch.cat((self.upscore2(M3[:, 0, :, :].unsqueeze(1)), M2_a), dim=1))
        att_b_ = self.attention_decoder2_b(torch.cat((self.upscore2(M3[:, 1, :, :].unsqueeze(1)), M2_b), dim=1))

        M2_a = torch.cat((self.upscore2(M3[:, 0, :, :].unsqueeze(1)), M2_a, 1 * att_b_.mul(M2_a)), dim=1)
        M2_b = torch.cat((self.upscore2(M3[:, 1, :, :].unsqueeze(1)), M2_b, 1 * att_a_.mul(M2_b)), dim=1)

        M2 = self.mask2_o(torch.cat((M2_a, M2_b), dim=1))  # 2

        M1_a = self.upscore2(h1_a)
        M1_b = self.upscore2(h1_b)

        att_a_ = self.attention_decoder1_a(torch.cat((self.upscore2(M2[:, 0, :, :].unsqueeze(1)), M1_a), dim=1))
        att_b_ = self.attention_decoder1_b(torch.cat((self.upscore2(M2[:, 1, :, :].unsqueeze(1)), M1_b), dim=1))

        M1_a = torch.cat((self.upscore2(M2[:, 0, :, :].unsqueeze(1)), M1_a, 1 * att_b_.mul(M1_a)), dim=1)
        M1_b = torch.cat((self.upscore2(M2[:, 1, :, :].unsqueeze(1)), M1_b, 1 * att_a_.mul(M1_b)), dim=1)

        M1 = self.mask1_o(torch.cat((M1_a, M1_b), dim=1))


        return M1, M2, M3, M4, M5

