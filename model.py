from dataloader import *
import torch.nn as nn


class VGGBased(nn.Module):
    def __init__(self, num_classes):
        super(VGGBased, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

        # stretched output of features 512, 4, 15
        self.classifier = nn.Sequential(
            nn.Linear(128 * 10 * 12, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # read classes
    dataset = UPBDataset("dataset")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    # load model
    net = VGGBased(2).cuda()

    # iterate through model
    for i, data in enumerate(dataloader, 0):
        out = net(data['img'].cuda())