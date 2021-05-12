import os
import time

import torchsnooper

from DatasetLoader import MyDataLoader
from models import SyncNetModelFBank
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def SampleFromTime(a):
	batch_size = a.size()[0]
	time_size = a.size()[2]
	b = []
	for i in range(batch_size):
		b.append(a[i, :, np.random.randint(0, time_size)])
	c = torch.stack(b)
	return c


# @torchsnooper.snoop()
def SingleModalityInfoNCE(feature_tensor, label_tensor, temperature=0.07):
	if len(feature_tensor.size()) == 2:
		feature_tensor = feature_tensor.unsqueeze(2)
	elif len(feature_tensor.size()) != 3:
		raise RuntimeError
	batch_size = feature_tensor.size()[0]
	cos_score = F.cosine_similarity(feature_tensor.expand(-1, -1, batch_size),
	                                feature_tensor.expand(-1, -1, batch_size).transpose(0, 2))
	loss = torch.zeros(1, dtype=torch.float32).cuda()
	for i in range(batch_size):
		numerator = torch.zeros(1, dtype=torch.float32).cuda()
		denominator = torch.zeros(1, dtype=torch.float32).cuda()
		for j in range(batch_size):
			if i == j: continue
			if label_tensor[i] == label_tensor[j]:
				numerator += torch.exp(cos_score[i, j]/temperature)
			denominator += torch.exp(cos_score[i, j]/temperature)
		if numerator == 0:
			numerator = torch.ones(1, dtype=torch.float32).cuda()
		loss -= torch.log((numerator/denominator))
	return loss, cos_score


def top1(score_mat, label):
	max_idx = score_mat.topk(2, dim=1)
	correct_cnt = 0
	total_cnt = len(max_idx)
	for i in range(total_cnt):
		# max_idx[1][i][1]
		# 第一个 1 指的是 topk的序号（0对应的是值）
		# 第二个 i 指的是 第i个样本
		# 第三个 1 指的是 次相近的序号，因为最相近的一定是本身
		if label[i] == label[max_idx[1][i][1]]:
			correct_cnt += 1
	return correct_cnt/total_cnt


class Meter(object):
	def __init__(self, name, display, fmt=':f'):
		self.name = name
		self.display = display
		self.fmt = fmt
		self.start_time = 0
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.start_time = 0
		self.time = 0

	def set_start_time(self, start_time):
		self.start_time = start_time

	def update(self, val, n=1):
		self.val = val
		self.sum += val*n
		self.count += n
		self.avg = self.sum/self.count
		self.time = val-self.start_time

	def __str__(self):
		fmtstr = '{name}:{'+self.display+self.fmt+'},'
		return fmtstr.format(**self.__dict__)


# @torchsnooper.snoop()
def main():
	batch_size = 30
	learning_rate = 0.001
	exp_path = 'data/exp12/'
	model_path = exp_path+'model'
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	result_path = exp_path+'result.txt'
	fw = open(result_path, 'w')
	print('Start loading dataset')
	# train_loader = MyDataLoader("data/test.txt", batch_size)
	train_loader = MyDataLoader("data/train.txt", batch_size)
	# valid_loader = MyDataLoader("data/test.txt", batch_size)
	print('Finish loading dataset')
	model = SyncNetModelFBank.SyncNetModel(stride=2).cuda()
	optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
	temperature = 0.07
	loss = Meter('Loss', 'avg', ':.2f')
	epoch_time = Meter('Time', 'time', ':3.0f')

	for epoch in range(100):
		print('\nEpoch: %d'%epoch)
		batch_cnt = 0
		epoch_time.set_start_time(time.time())
		for data in train_loader:
			data_video, data_audio, data_label = data
			data_video, data_audio = data_video.cuda(), data_audio.cuda()
			_, audio_id = model.forward_aud(data_audio)
			_, video_id = model.forward_vid(data_video)
			# (batch_size, feature, time_size)

			audio_random_id = SampleFromTime(audio_id).unsqueeze(2)
			video_random_id = SampleFromTime(video_id).unsqueeze(2)
			# (batch_size, feature)

			audio_id_loss, _ = SingleModalityInfoNCE(audio_random_id, data_label, temperature)
			video_id_loss, _ = SingleModalityInfoNCE(video_random_id, data_label, temperature)
			# audio_acc = top1(audio_score, data_label)
			# video_acc = top1(video_score, data_label)

			final_loss = audio_id_loss+video_id_loss
			final_loss.backward()
			optim.step()
			batch_cnt += 1
			loss.update(final_loss.item())
			epoch_time.update(time.time())
			print('\rBatch:(%02d/%d)    %s    %s      '%(batch_cnt, len(train_loader), epoch_time, loss), end='')
		# print('Loss: %.2f\nAudio ID ACC:%.2f%%\tVideo ID ACC:%.2f%%'%(final_loss.item(), audio_acc*100, video_acc*100))
		print('Epoch: %3d\t%s'%(epoch, loss), file=fw)
		loss.reset()
		torch.cuda.empty_cache()
		torch.save(model.state_dict(), model_path+"/model%09d.model"%epoch)


main()
