#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchaudio
import torchsnooper


# @torchsnooper.snoop()
class SyncNetModel(nn.Module):
	def __init__(self, nOut=1024, stride=1):
		super(SyncNetModel, self).__init__()

		self.netcnnaud = nn.Sequential(
			# (b, 1, 128, time)
			nn.Conv2d(1, 96, kernel_size=(5, 7), stride=(1, 1), padding=(2, 2)),
			# (b, 96, 128, t-2)
			nn.BatchNorm2d(96),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
			# (b, 96,

			nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 1), padding=(1, 1)),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 1)),

			nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1)),
			nn.BatchNorm2d(384),
			nn.ReLU(inplace=True),

			nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),

			nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),

			nn.Conv2d(256, 512, kernel_size=(4, 1), padding=(0, 0), stride=(1, stride)),
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)

		self.netfcaud = nn.Sequential(
			nn.Conv1d(512, 512, kernel_size=1),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Conv1d(512, nOut, kernel_size=1),
		)

		self.netfclip = nn.Sequential(
			nn.Conv1d(512, 512, kernel_size=1),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Conv1d(512, nOut, kernel_size=1),
		)

		self.netfcspk = nn.Sequential(
			nn.Conv1d(512, 512, kernel_size=1),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Conv1d(512, nOut, kernel_size=1),
		)

		self.netfcface = nn.Sequential(
			nn.Conv1d(512, 512, kernel_size=1),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Conv1d(512, nOut, kernel_size=1),
		)

		self.netcnnlip = nn.Sequential(
			nn.Conv3d(3, 96, kernel_size=(5, 7, 7), stride=(stride, 2, 2), padding=0),
			nn.BatchNorm3d(96),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

			nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
			nn.BatchNorm3d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),

			nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
			nn.BatchNorm3d(256),
			nn.ReLU(inplace=True),

			nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
			nn.BatchNorm3d(256),
			nn.ReLU(inplace=True),

			nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
			nn.BatchNorm3d(256),
			nn.ReLU(inplace=True),
			nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2)),

			nn.Conv3d(256, 512, kernel_size=(1, 6, 6), padding=0),
			nn.BatchNorm3d(512),
			nn.ReLU(inplace=True),
		)

		self.instancenorm = nn.InstanceNorm1d(40)
		self.torchfb = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400,
		                                                    hop_length=160, f_min=0.0, f_max=8000, pad=0, n_mels=40)

	# @torchsnooper.snoop()
	def forward_vid(self, x):
		## Image stream
		# x = (5, 3, 40, 480, 480)
		#       x = (5, 3, 40, 360, 360)
		#           x = (5, 3, 40, 240, 240)

		mid = self.netcnnlip(x)
		# mid = (5, 512, 36, 9, 9)
		#       mid = (5, 512, 36, 5, 5)
		#           mid = (5, 512, 36, 1, 1)
		mid = mid.view((mid.size()[0], mid.size()[1], -1))  # N x (ch x 24)
		# mid = (5, 512, 2916)
		#       mid = (5, 512, 900)
		#           mid = (5, 512, 36)

		out1 = self.netfclip(mid)
		# out1 = (5, 1024, 2916)
		#       out1 = (5, 1024, 900)
		#           out1 = (5, 1024, 36)
		out2 = self.netfcface(mid)
		# out2 = (5, 1024, 2916)
		#       out2 = (5, 1024, 900)
		#           out2 = (5, 1024, 36)

		return out1, out2

	# @torchsnooper.snoop()
	def forward_aud(self, x):
		## Audio stream
		# x = (batch, time=max_audio)
		# (5, 25840)

		x = self.torchfb(x)+1e-6
		# x = (batch, n_mels=40, self.nMaxFrames*4+2 = 162)
		# (5, 40, 162)
		x = self.instancenorm(x.log())
		x = x[:, :, 1:-1].detach()
		# x = (batch, n_mels=40, self.nMaxFrames*4=160)
		# (5, 40, 160)

		mid = self.netcnnaud(x.unsqueeze(1))  # N x ch x 24 x M
		# N=40(帧数，也可能等于160)     ch=1(通道数，单声道还是双声道)      24(不知道)     M(不知道)
		# mid = (batch, 512, 1, 36)
		mid = mid.view((mid.size()[0], mid.size()[1], -1))  # N x (ch x 24)
		# mid = (5, 512, 36)

		out1 = self.netfcaud(mid)
		# out1 = (5, 1024, 36)
		out2 = self.netfcspk(mid)
		# out2 = (5, 1024, 36)

		return out1, out2
