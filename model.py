from dataloader import *
import torch.nn as nn

import torch
import torch.nn as nn


class CAE(nn.Module):
	"""
	This AE module will be fed 3x128x128 patches from the original image
	Shapes are (batch_size, channels, height, width)

	Latent representation: 16x16x16 bits per patch => 30KB per image (for 720p)
	"""

	def __init__(self):
		super(CAE, self).__init__()

		self.encoded = None

		# ENCODER

		# 64x64x64
		self.e_conv_1 = nn.Sequential(
			nn.ZeroPad2d((1, 2, 1, 2)),
			nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True)
		)

		# 128x32x32
		self.e_conv_2 = nn.Sequential(
			nn.ZeroPad2d((1, 2, 1, 2)),
			nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True)
		)

		# 128x32x32
		self.e_block_1 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x32x32
		self.e_block_2 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x32x32
		self.e_block_3 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 16x16x16
		self.e_conv_3 = nn.Sequential(
			nn.ZeroPad2d((1, 2, 1, 2)),
			nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(5, 5), stride=(2, 2)),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True)
			# nn.Tanh()
		)

		# DECODER

		# 128x32x32
		self.d_up_conv_1 = nn.Sequential(
			nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1)),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2))
		)

		# 128x32x32
		self.d_block_1 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x32x32
		self.d_block_2 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 128x32x32
		self.d_block_3 = nn.Sequential(
			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)),
		)

		# 256x64x64
		self.d_up_conv_2 = nn.Sequential(
			nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.ConvTranspose2d(in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2))
		)

		# 3x128x128
		self.d_up_conv_3 = nn.Sequential(
			nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)),
			nn.BatchNorm2d(16),
			nn.ReLU(inplace=True),

			nn.ZeroPad2d((1, 1, 1, 1)),
			nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(2, 2), stride=(2, 2)),
		)

	def forward(self, x):
		ec1 = self.e_conv_1(x)
		ec2 = self.e_conv_2(ec1)
		eblock1 = self.e_block_1(ec2) + ec2
		eblock2 = self.e_block_2(eblock1) + eblock1
		eblock3 = self.e_block_3(eblock2) + eblock2
		ec3 = self.e_conv_3(eblock3)  # in [-1, 1] from tanh activation

		# # stochastic binarization
		# with torch.no_grad():
		# 	rand = torch.rand(ec3.shape).cuda()
		# 	prob = (1 + ec3) / 2
		# 	eps = torch.zeros(ec3.shape).cuda()
		# 	eps[rand <= prob] = (1 - ec3)[rand <= prob]
		# 	eps[rand > prob] = (-ec3 - 1)[rand > prob]

		# # encoded tensor
		# self.encoded = 0.5 * (ec3 + eps + 1)  # (-1|1) -> (0|1)
		self.encoded = ec3

		return self.decode(self.encoded)

	def decode(self, encoded):
		# y = encoded * 2.0 - 1  # (0|1) -> (-1|1)
		y = encoded

		uc1 = self.d_up_conv_1(y)
		dblock1 = self.d_block_1(uc1) + uc1
		dblock2 = self.d_block_2(dblock1) + dblock1
		dblock3 = self.d_block_3(dblock2) + dblock2
		uc2 = self.d_up_conv_2(dblock3)
		dec = self.d_up_conv_3(uc2)

		return dec

if __name__ == "__main__":
    # read classes
    dataset = UPBDataset("test")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    # load model
    net = CAE().cuda()

    # iterate through model
    for i, data in enumerate(dataloader, 0):
        out = net(data['img'].cuda())
        print(out.shape)