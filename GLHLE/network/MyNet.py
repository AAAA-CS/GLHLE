import torch
from torch import nn


class mynet(nn.Module):
    def __init__(self, band, classes):
        super(mynet, self).__init__()
        self.conv3d_01 = nn.Sequential(
                nn.Conv3d(1, 24, kernel_size=(1, 1, 7), padding=(0, 0, 0), stride=(1, 1, 2)),
                nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                nn.Mish(inplace=True)
            )
        self.conv3d_02 = nn.Sequential(
                nn.Conv3d(24, 24, kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=(1, 1, 1)),
                nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                nn.Mish(inplace=True)
            )
        self.conv3d_03 = nn.Sequential(
                nn.Conv3d(24, 24, kernel_size=(1, 1, 3), padding=(0, 0, 1), stride=(1, 1, 1)),
                nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
                nn.Mish(inplace=True)
            )
        self.conv3d_04 = nn.Sequential(
                nn.Conv3d(24, 96, kernel_size=(1, 1, ((30 - 7) // 2 + 1)), padding=(0, 0, 0), stride=(1, 1, 1)),
                nn.BatchNorm3d(96, eps=0.001, momentum=0.1, affine=True),
                nn.Mish(inplace=True)
            )

        self.conv_DR = nn.Sequential(
                nn.Conv2d(in_channels=band, out_channels=30, kernel_size=1, padding=0, stride=1),
                nn.BatchNorm2d(30, eps=0.001, momentum=0.1, affine=True),
                nn.Mish()
            )

        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=32, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )

        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )

        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, padding=1, kernel_size=3, stride=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )

        self.convC = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, padding=0, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True),
            nn.Mish(inplace=True)
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, classes)

        )

        self.projection = nn.Sequential(

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)

        )

    def forward(self, X):

        X= self.conv_DR(X)
        spe1 = self.conv3d_01(X.permute(0, 2, 3, 1).unsqueeze(1))

        spe2 = self.conv3d_02(spe1)
        spe3 = self.conv3d_03(spe2)

        spe4 = self.conv3d_04(spe3)
        spe4 = spe4.squeeze(-1)

        spa1 = self.conv2d_1(X)
        spa2 = self.conv2d_2(spa1)

        ss = torch.cat((spa2, spe4), dim=1)

        ss = self.convC(ss)
        ss = self.gap(ss)
        ss = ss.view(ss.size(0), -1)

        projection = self.projection(ss)
        cl = self.fc(ss)
        return projection, cl
