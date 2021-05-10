#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import numpy
import random
import pdb
import os
import threading
import time
from queue import Queue
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from dataLoader import *


class MyDataset(Dataset):
	def __init__(self, dataset_file_name, eval_mode=False, maxFrames=40):
		self.dataset_file_name = dataset_file_name
		self.info_list = []
		self.data_list = []
		self.ndata = None
		self.eval_mode = eval_mode
		self.maxFrames = maxFrames

		people_dict = dict()
		people_cnt = 0
		with open(dataset_file_name) as listfile:
			while True:
				line = listfile.readline()
				if not line:
					break
				data = line.split()
				people = data[0].strip().split('/')[-2]
				if len(data) == 4:
					if abs(int(data[3]))-abs(int(data[2]))>=maxFrames+4:
						if people not in people_dict.keys():
							people_dict[people] = people_cnt
							people_cnt += 1
							data.append(people_cnt)
							if eval_mode:
								self.info_list.append(data)
						else:
							data.append(people_dict[people])
						if not eval_mode:
							self.info_list.append(data)
					else:
						print('%s is too short'%(data[0]))
				else:
					raise
		for info in self.info_list:
			mp4name, wavname = info[0], info[1]
			mp4data = get_frames(mp4name, max_frames=self.maxFrames, start_frame=0)
			wavdata = loadWAV(wavname, max_frames=self.maxFrames*4, start_frame=0)
			mp4data, wavdata = mp4data.squeeze(), wavdata.squeeze()
			self.data_list.append((mp4data, wavdata, info[-1]))
		self.ndata = len(self.data_list)
		print('Evalmode %s - %d clips'%(self.eval_mode, len(self.data_list)))

	def __getitem__(self, item):
		return self.data_list[item]

	def __len__(self):
		return self.ndata


class MyDataLoader(DataLoader):
	def __init__(self, dataset_file_name, batch_size, eval_mode=False, **kwargs):
		self.dataset = MyDataset(dataset_file_name, eval_mode)
		super().__init__(self.dataset, shuffle=True, batch_size=batch_size, drop_last=True)
		self.nFiles = len(self.dataset)
		self.maxQueueSize = 0

	def qsize(self):
		return 0


class DatasetLoader(object):
	def __init__(self, dataset_file_name, nPerEpoch, nBatchSize, maxFrames, nDataLoaderThread, maxQueueSize=10,
	             evalmode=False, **kwargs):
		self.dataset_file_name = dataset_file_name
		self.nPerEpoch = nPerEpoch
		self.nWorkers = nDataLoaderThread
		self.nMaxFrames = maxFrames
		self.batch_size = nBatchSize
		self.maxQueueSize = maxQueueSize

		self.data_list = []
		self.data_epoch = []

		self.nFiles = 0
		self.evalmode = evalmode

		self.dataLoaders = []

		with open(dataset_file_name) as listfile:
			while True:
				line = listfile.readline()
				if not line:
					break

				data = line.split()

				if len(data) == 4:
					if abs(int(data[3]))-abs(int(data[2]))>=maxFrames+4:
						self.data_list.append(data)
					else:
						print('%s is too short'%(data[0]))
				else:
					raise

		### Initialize Workers...
		self.datasetQueue = Queue(self.maxQueueSize)

		print('Evalmode %s - %d clips'%(self.evalmode, len(self.data_list)))

	def dataLoaderThread(self, nThreadIndex):

		index = nThreadIndex*self.batch_size

		if index>=self.nFiles:
			return

		while (True):
			if (self.datasetQueue.full() == True):
				time.sleep(1.0)
				continue

			feat_a = []
			feat_i = []

			for filename in self.data_epoch[index:index+self.batch_size]:
				offset = int(filename[2])
				vidlength = int(filename[3])

				firststart = 2-min(offset, 0)
				laststart = vidlength-max(offset, 0)-(self.nMaxFrames+2)

				# if self.evalmode:
				startidx = random.randint(firststart, laststart)

				feat_a.append(loadWAV(filename[1], max_frames=self.nMaxFrames*4, start_frame=startidx*4))
				feat_i.append(get_frames(filename[0], max_frames=self.nMaxFrames, start_frame=startidx+offset-1))
			# feat_a的形状是（batch，(1，max_audio, 2)）
			# feat_i的形状是（batch，1，3，max_frames，h, w）

			data_aud = torch.cat(feat_a, dim=0)
			data_im = torch.cat(feat_i, dim=0)
			# print('aud:', data_aud.shape)
			# print('img:', data_im.shape)
			# data_aud的形状是（batch，max_audio, 2）
			# data_im 的形状是（batch，3，max_frames，h, w）

			self.datasetQueue.put([data_im, data_aud])

			index += self.batch_size*self.nWorkers

			if (index+self.batch_size>self.nFiles):
				break

	def __iter__(self):
		## Shuffle one more
		random.shuffle(self.data_list)

		self.data_epoch = self.data_list[:min(self.nPerEpoch, len(self.data_list))]
		self.nFiles = len(self.data_epoch)

		### Make and Execute Threads...
		for index in range(0, self.nWorkers):
			self.dataLoaders.append(threading.Thread(target=self.dataLoaderThread, args=[index]))
			self.dataLoaders[-1].start()

		return self

	def __next__(self):
		while (True):
			isFinished = True

			if (self.datasetQueue.empty() == False):
				return self.datasetQueue.get()
			for index in range(0, self.nWorkers):
				if (self.dataLoaders[index].is_alive() == True):
					isFinished = False
					break

			if (isFinished == False):
				time.sleep(1.0)
				continue

			for index in range(0, self.nWorkers):
				self.dataLoaders[index].join()

			self.dataLoaders = []
			raise StopIteration

	def __call__(self):
		pass

	def getDatasetName(self):
		return self.dataset_file_name

	def qsize(self):
		return self.datasetQueue.qsize()
