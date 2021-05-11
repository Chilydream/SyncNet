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


def SingleModalityInfoNCE(feature_tensor, label_tensor, temperature=0.07):
	if len(feature_tensor.size())==2:
		feature_tensor = feature_tensor.unsqueeze(2)
	elif len(feature_tensor.size())!=3:
		raise RuntimeError
	batch_size = feature_tensor.size()[0]
	cos_score = F.cosine_similarity(feature_tensor.expand(-1, -1, batch_size),
	                                feature_tensor.expand(-1, -1, batch_size).transpose(0, 2))
	loss = torch.zeros(1, dtype=torch.float32)
	for i in range(batch_size):
		numerator = torch.zeros(1, dtype=torch.float32)
		denominator = torch.zeros(1, dtype=torch.float32)
		for j in range(batch_size):
			if i==j: continue
			if label_tensor[i]==label_tensor[j]:
				numerator += torch.exp(cos_score[i, j]/temperature)
			denominator += torch.exp(cos_score[i, j]/temperature)
		if numerator==0:
			numerator = torch.ones(1, dtype=torch.float32)
		loss -= torch.log((numerator/denominator)+torch.finfo(torch.float32))
	return loss, cos_score


def top1(score_mat, label):
	_, max_idx = score_mat.argmax(dim=1)
	correct_cnt = 0
	total_cnt = len(max_idx)
	for i in range(total_cnt):
		if label[i]==label[max_idx[i]]:
			correct_cnt += 1
	return correct_cnt/total_cnt


batch_size = 30
learning_rate = 0.001
train_loader = MyDataLoader("data/train.txt", batch_size)
valid_loader = MyDataLoader("data/test.txt", batch_size)
model = SyncNetModelFBank.SyncNetModel().cuda()
optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)
temperature = 0.07

for epoch in range(100):
	print('Epoch: %d'%epoch)
	for data in train_loader:
		data_video, data_audio, data_label = data
		data_video, data_audio = data_video.cuda(), data_audio.cuda()
		audio_ct, audio_id = model.forward_aud(data_audio)
		video_ct, video_id = model.forward_vid(data_video)
		time_size = audio_ct.size()[2]
		id_loss = torch.zeros(1, dtype=torch.float32)
		# (batch_size, feature, time_size)

		audio_random_id = SampleFromTime(audio_id).unsqueeze(2)
		video_random_id = SampleFromTime(video_id).unsqueeze(2)
		# (batch_size, feature)

		audio_id_loss, audio_score = SingleModalityInfoNCE(audio_id, data_label)
		video_id_loss, video_score = SingleModalityInfoNCE(video_id, data_label)
		audio_acc = top1(audio_score, data_label)
		video_acc = top1(video_score, data_label)

		final_loss = audio_id_loss+video_id_loss
		final_loss.backward()
		optim.step()
		print('Loss: %.2f\nAudio ID ACC:%.2f\tVideo ID ACC:%.2f'%(final_loss.detach().cpu(), audio_acc, valid_loader))
