import glob
import os

import cv2
import numpy as np
import torch

from DatasetLoader import DatasetLoader
from SyncNetDist import LossScale
from models.SyncNetModelFBank import SyncNetModel


def mp42wav(filepath):
	# filepath = '/home/ChuyuanXiong/fsx/demo/006_0_jiangying_lecture/0.mp4'
	filepath_no_ext = os.path.splitext(filepath)[0]
	command = "ffmpeg -i "+filepath+" -ac 1 -ar 16000 -vn "+filepath_no_ext+".wav"
	print('command', command)
	os.system(command)


def count_mp4(filename):
	cap = cv2.VideoCapture(filename)
	total_frames = cap.get(7)
	return [total_frames, total_frames]
	counted_frames = 0
	while True:
		ret, image = cap.read()
		if ret == 0:
			break
		else:
			counted_frames += 1
	cap.release()

	if total_frames != counted_frames:
		return [total_frames, counted_frames]
	return [total_frames, counted_frames]


def get_meta():
	peoplelist = os.listdir('/GPUFS/ruc_tliu_1/fsx/SyncNet/data/demo/')
	with open("train.txt", 'w') as fw:
		for people in peoplelist:
			people_path = '/GPUFS/ruc_tliu_1/fsx/SyncNet/data/demo/'+people+'/'
			mp4list = glob.glob(people_path + '*.mp4')
			if len(mp4list) == 5:
				for mp4file in mp4list:
					mp4path = mp4file
					wavpath = mp4file[:-4]+'.wav'
					if os.path.exists(wavpath):
						os.remove(wavpath)
					if not os.path.exists(wavpath):
						mp42wav(mp4path)
					total_frames = count_mp4(mp4path)
					if total_frames[0] != total_frames[1]:
						print('ERROR, frames mismatch:', total_frames[0], total_frames[1])
						continue
					fw.write("%s %s %s %d\n"%(mp4path, wavpath, '0', total_frames[0]))


if __name__ == '__main__':
	# get_meta()
	loader = DatasetLoader("train.txt", nPerEpoch=100000, nBatchSize=30, maxFrames=44, nDataLoaderThread=10)
	S = SyncNetModel(nOut=1024, stride=2).cuda()
	L = LossScale().cuda()
	for data in loader:
		data_i, data_a = data
		out_i, out_I = S.forward_vid(data_i.cuda())
		out_a, out_A = S.forward_aud(data_a.cuda())
		print(out_a.shape, out_A.shape)
		print(out_i.shape, out_I.shape)
		exit(0)

