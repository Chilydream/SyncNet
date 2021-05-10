#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, os, random, time, pdb, numpy, importlib

import torchsnooper

from dataLoader import loadWAV
from accuracy import accuracy


class LossScale(nn.Module):
	def __init__(self, init_w=10.0, init_b=-5.0):
		super(LossScale, self).__init__()

		self.wI = nn.Parameter(torch.tensor(init_w))
		self.bI = nn.Parameter(torch.tensor(init_b))

		self.wC = nn.Parameter(torch.tensor(init_w))
		self.bC = nn.Parameter(torch.tensor(init_b))


class SyncNet(nn.Module):

	def __init__(self, model=None, maxFrames=200, learning_rate=0.01, nOut=1024, temporal_stride=1,
	             disentangle_step=10, **kwargs):
		# nout指的是embedding的维数
		super(SyncNet, self).__init__()

		SyncNetModel = importlib.import_module(model).__getattribute__("SyncNetModel")

		self.learning_rate = learning_rate
		self.__S__ = SyncNetModel(nOut=nOut, stride=temporal_stride).cuda()
		self.__L__ = LossScale().cuda()

		self.__optimizer__ = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

		self.__max_frames__ = maxFrames

	def disentangle_step(self, loader):
		criterion = torch.nn.CrossEntropyLoss()
		self.__S__.eval()
		self.__L__.eval()
		id2content_L = LossScale().cuda()
		content2id_L = LossScale().cuda()
		dis_para = []
		for pname, p in id2content_L.named_parameters():
			dis_para.append(p)
		for pname, p in content2id_L.named_parameters():
			dis_para.append(p)
		dis_optim = optim.SGD(dis_para, lr=0.01, momentum=0.9, weight_decay=1e-5)
		for data in loader:
			data_v, data_a, id_label = data
			# data_a 的形状是（batch，max_audio）
			# data_v 的形状是（batch，3，max_frames，w，h）
			# id_label 的形状是 （batch，1）

			# ==================== FORWARD PASS ====================
			out_a, out_A = self.__S__.forward_aud(data_a.cuda())
			out_v, out_V = self.__S__.forward_vid(data_v.cuda())
			# out_a 和 out_v 代表的是内容特征
			# out_A 和 out_V 代表的是身份特征

			time_size = out_V.size()[2]
			# 使用 id信息去找 content
			loss_id2content = self.sync_loss(out_V, out_v, criterion, id2content_L)[0]

			# 使用 content信息去找 id
			stepsize = loader.batch_size
			label_id = torch.arange(stepsize).cuda()

			ri = random.randint(0, time_size-1)
			out_AA = torch.mean(out_A, 2, keepdim=True)
			out_aA = out_a[:, :, [ri]]

			idoutput = F.cosine_similarity(out_aA.expand(-1, -1, stepsize),
			                               out_AA.expand(-1, -1, stepsize).transpose(0, 2))\
			           *content2id_L.wC+content2id_L.bC

			loss_content2id = criterion(idoutput, label_id)
			whole_loss = loss_content2id+loss_id2content
			whole_loss.backward()
			dis_optim.step()
		content2id_L.eval()
		id2content_L.eval()
		self.__S__.train()
		self.__L__.train()
		for data in loader:
			data_v, data_a = data
			out_a, out_A = self.__S__.forward_aud(data_a.cuda())
			out_v, out_V = self.__S__.forward_vid(data_v.cuda())
			# out_a 和 out_v 代表的是内容特征
			# out_A 和 out_V 代表的是身份特征

			time_size = out_V.size()[2]
			# 使用 id信息去找 content
			loss_id2content = self.sync_loss(out_V, out_v, criterion, id2content_L, dis2=True)

			# 使用 content信息去找 id
			stepsize = loader.batch_size
			label_id = torch.arange(stepsize).cuda()

			ri = random.randint(0, time_size-1)
			out_AA = torch.mean(out_A, 2, keepdim=True)
			out_aA = out_a[:, :, [ri]]

			idoutput = F.cosine_similarity(out_aA.expand(-1, -1, stepsize),
			                               out_AA.expand(-1, -1, stepsize).transpose(0, 2))\
			           *content2id_L.wC+content2id_L.bC

			loss_content2id = idoutput.std()
			whole_loss = loss_content2id+loss_id2content
			whole_loss.backward()
			self.__optimizer__.step()

	def sync_loss(self, out_v, out_a, criterion, L, dis2=False):

		batch_size = out_a.size()[0]
		feature_size = out_a.size()[1]
		time_size = out_a.size()[2]
		merge_size = (time_size-1)//3+1

		label = torch.arange(merge_size).cuda()

		nloss = 0
		prec1 = 0

		for ii in range(0, batch_size):
			ft_v = out_v[[ii], :, :].transpose(2, 0)
			# ft_v = (15, 1024, 1)
			ft_a = out_a[[ii], :, :].transpose(2, 0)
			# ft_a = (15, 1024, 1)
			ft_v_merge = torch.zeros(((time_size-1)//3+1, feature_size, 1), dtype=torch.float32).cuda()
			ft_a_merge = torch.zeros(((time_size-1)//3+1, feature_size, 1), dtype=torch.float32).cuda()
			for itime in range(0, time_size):
				ft_v_merge[(itime-1)//3] += ft_v[itime]
				ft_a_merge[(itime-1)//3] += ft_a[itime]
			# output = F.cosine_similarity(ft_v.expand(-1, -1, time_size),
			#                              ft_a.expand(-1, -1, time_size).transpose(0, 2)) \
			#          *self.__L__.wC+self.__L__.bC
			output = F.cosine_similarity(ft_v_merge.expand(-1, -1, merge_size),
			                             ft_a_merge.expand(-1, -1, merge_size).transpose(0, 2)) \
			         *L.wC+L.bC
			if dis2:
				dumb_score = output.std()
				return dumb_score
			# output = (15, 15)
			p1 = accuracy(output.detach().cpu(), label.detach().cpu(), topk=(1,))[0]

			nloss += criterion(output, label)
			prec1 += p1

		nloss = nloss/batch_size
		prec1 = prec1/batch_size

		return nloss, prec1

	def train_network(self, loader=None, evalmode=None, alpC=1.0, alpI=1.0, alpR=1.0):

		print('Content loss %f Identity loss %f'%(alpC, alpI))

		if evalmode:
			self.eval()
		else:
			self.train()

		# ==================== ====================

		counter = 0

		index = 0

		loss = 0
		eer = 0

		top1_sy = 0
		top1_id = 0

		criterion = torch.nn.CrossEntropyLoss()
		stepsize = loader.batch_size
		label_id = torch.arange(stepsize).cuda()

		for data in loader:

			tstart = time.time()

			self.zero_grad()

			data_v, data_a = data
			# data_a 的形状是（batch，max_audio）
			# data_v 的形状是（batch，3，max_frames，w，h）

			# ==================== FORWARD PASS ====================

			if evalmode:
				with torch.no_grad():
					out_a, out_A = self.__S__.forward_aud(data_a.cuda())
					out_v, out_V = self.__S__.forward_vid(data_v.cuda())

			else:
				out_a, out_A = self.__S__.forward_aud(data_a.cuda())
				out_v, out_V = self.__S__.forward_vid(data_v.cuda())
			# out_a 和 out_v 代表的是内容特征
			# out_A 和 out_V 代表的是身份特征

			time_size = out_V.size()[2]

			ri = random.randint(0, time_size-1)
			out_AA = torch.mean(out_A, 2, keepdim=True)
			out_VA = out_V[:, :, [ri]]

			# sync loss and accuracy
			nloss_sy, p1s = self.sync_loss(out_v, out_a, criterion, self.__L__)

			# identity loss and accuracy
			idoutput = F.cosine_similarity(out_VA.expand(-1, -1, stepsize),
			                               out_AA.expand(-1, -1, stepsize).transpose(0, 2))\
			           *self.__L__.wI+self.__L__.bI

			nloss_id = criterion(idoutput, label_id)

			p1i = accuracy(idoutput.detach().cpu(), label_id.detach().cpu(), topk=(1,))[0]

			# A_norm = torch.abs(1-out_A.norm(2, dim=1)).mean()
			# a_norm = torch.abs(1-out_a.norm(2, dim=1)).mean()
			# V_norm = torch.abs(1-out_V.norm(2, dim=1)).mean()
			# v_norm = torch.abs(1-out_v.norm(2, dim=1)).mean()
			# reg_loss = A_norm+a_norm+V_norm+v_norm
			# ==================== Divergence Loss ====================

			nloss = alpC*nloss_sy+alpI*nloss_id
			# nloss = alpC*nloss_sy + alpI*nloss_id + alpR*reg_loss

			if not evalmode:
				nloss.backward()
				self.__optimizer__.step()

			loss += nloss.detach().cpu()
			top1_sy += p1s[0]
			top1_id += p1i[0]

			counter += 1

			telapsed = time.time()-tstart

			sys.stdout.write("\rProc (%d/%d): %.3fHz "%(index, loader.nFiles, stepsize/telapsed))
			sys.stdout.write("Ls %.3f SYT1 %2.3f%% "%(loss/counter, top1_sy/counter))
			sys.stdout.write("IDT1 %2.3f%% "%(top1_id/counter))

			# ==================== CALCULATE LOSSES ====================

			sys.stdout.flush()

			index += stepsize

		sys.stdout.write("\n")

		return (loss/counter, top1_id/counter)

	## ===== ===== ===== ===== ===== ===== ===== =====
	## Evaluate from list
	## ===== ===== ===== ===== ===== ===== ===== =====

	def evaluateFromListSave(self, listfilename, print_interval=10, test_path='', num_eval=10):

		self.eval()

		lines = []
		files = []
		filedict = {}
		feats = {}
		tstart = time.time()

		## Read all lines
		with open(listfilename) as listfile:
			while True:
				line = listfile.readline()
				if (not line):
					break

				data = line.split()

				files.append(data[1])
				files.append(data[2])
				lines.append(line)

		setfiles = list(set(files))
		setfiles.sort()

		## Save all features to file
		for idx, file in enumerate(setfiles):

			inp1 = loadWAV(os.path.join(test_path, file), self.__max_frames__*4, evalmode=True,
			               num_eval=num_eval).cuda()

			out_a, out_A = self.__S__.forward_aud(inp1.cuda())
			out_AA = torch.mean(out_A, 2)

			feats[file] = out_AA.detach().cpu()

			telapsed = time.time()-tstart

			if idx%print_interval == 0:
				sys.stdout.write("\rReading %d: %.2f Hz, embed size %d"%(idx, idx/telapsed, out_AA.size()[1]))

		print('')
		all_scores = []
		all_labels = []
		tstart = time.time()

		## Read files and compute all scores
		for idx, line in enumerate(lines):

			data = line.split()

			ref_feat = feats[data[1]].cuda()
			com_feat = feats[data[2]].cuda()

			dist = F.cosine_similarity(ref_feat.unsqueeze(-1).expand(-1, -1, num_eval),
			                           com_feat.unsqueeze(-1).expand(-1, -1, num_eval).transpose(0,
			                                                                                     2)).detach().cpu().numpy()

			score = numpy.mean(dist)

			all_scores.append(score)
			all_labels.append(int(data[0]))

			if idx%print_interval == 0:
				telapsed = time.time()-tstart
				sys.stdout.write("\rComputing %d: %.2f Hz"%(idx, idx/telapsed))
				sys.stdout.flush()

		print('\n')

		return (all_scores, all_labels)

	## ===== ===== ===== ===== ===== ===== ===== =====
	## Update learning rate
	## ===== ===== ===== ===== ===== ===== ===== =====

	def updateLearningRate(self, alpha):

		learning_rate = []
		for param_group in self.__optimizer__.param_groups:
			param_group['lr'] = param_group['lr']*alpha
			learning_rate.append(param_group['lr'])

		return learning_rate

	## ===== ===== ===== ===== ===== ===== ===== =====
	## Save parameters
	## ===== ===== ===== ===== ===== ===== ===== =====

	def saveParameters(self, path):

		torch.save(self.state_dict(), path)

	## ===== ===== ===== ===== ===== ===== ===== =====
	## Load parameters
	## ===== ===== ===== ===== ===== ===== ===== =====

	def loadParameters(self, path):

		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")

				if name not in self_state:
					print("%s is not in the model."%origname)
					continue

			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(
					origname, self_state[name].size(), loaded_state[origname].size()))
				continue

			self_state[name].copy_(param)
