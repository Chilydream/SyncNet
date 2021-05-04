#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import numpy
import random
import pdb
import os
import cv2
import math
from scipy.io import wavfile


def loadWAV(filename, max_frames, start_frame=0, evalmode=False, num_eval=10):
	# Maximum audio length
	max_audio = max_frames*160+240
	# self.nMaxFrames = 40 的情况下
	# max_audio = 25840
	# self.nMaxFrames = 50 的情况下
	# max_audio = 32240
	start_audio = start_frame*160
	# ques: 为什么传进来的 start frame要乘 4

	# Read wav file and convert to torch tensor
	sample_rate, audio = wavfile.read(filename)
	# audio.shape = (88751)
	# 如果是双声道：audio.shape = (88751, 2)

	audiosize = audio.shape[0]

	if audiosize<=max_audio:
		raise ValueError('Audio clip is too short')

	if evalmode:
		start_frame = numpy.linspace(0, audiosize-max_audio, num=num_eval)
	else:
		start_frame = numpy.array([start_audio])

	feats = []
	for asf in start_frame:
		feats.append(audio[int(asf):int(asf)+max_audio])

	feat = numpy.stack(feats, axis=0)

	feat = torch.FloatTensor(feat)
	# feat size = [1, max_audio=25840]

	# 返回的是（1，max_audio）的 tensor
	return feat


def make_image_square(img):
	s = max(img.shape[0:2])
	f = numpy.zeros((s, s, 3), numpy.uint8)
	ax, ay = (s-img.shape[1])//2, (s-img.shape[0])//2
	f[ay:img.shape[0]+ay, ax:ax+img.shape[1]] = img
	return f


def get_frames(filename, max_frames=100, start_frame=0):
	cap = cv2.VideoCapture(filename)

	cap.set(1, start_frame)

	images = []
	for frame_num in range(0, max_frames):
		ret, image = cap.read()
		image = make_image_square(image)
		image = cv2.resize(image, (240, 240))
		images.append(image)

	cap.release()
	# im的形状是（w，h，3）
	im = numpy.stack(images, axis=3)
	# stack操作后 im的形状是（w，h，3，max_frames）
	im = numpy.expand_dims(im, axis=0)
	# im的形状是（1，w，h，3，max_frames)
	im = numpy.transpose(im, (0, 3, 4, 1, 2))
	# im的形状是（1，3，max_frames，w，h）
	imtv = torch.FloatTensor(im)

	return imtv
