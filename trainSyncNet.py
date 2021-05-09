#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
import sys
import time
import os
import argparse
import pdb
import glob
import torch
from SyncNetDist import SyncNet

from tuneThreshold import tuneThresholdfromScore
from sklearn import metrics
from DatasetLoader import DatasetLoader, MyDataLoader

parser = argparse.ArgumentParser(description="TrainArgs")
# python -u trainSyncNet.py --temporal_stride 1 --nOut 256 --save_path data/exp06
## Data loader
parser.add_argument('--maxFrames', type=int, default=34, help='')
parser.add_argument('--nBatchSize', type=int, default=30, help='')
parser.add_argument('--nTrainPerEpoch', type=int, default=100000, help='')
parser.add_argument('--nTestPerEpoch', type=int, default=10000, help='')
parser.add_argument('--nDataLoaderThread', type=int, default=4, help='')
parser.add_argument('--goon', action='store_true', help='Train from zero or Continue last training.' )
parser.add_argument('--hard_eval', type=bool, default=False)
parser.add_argument('--disentangle', type=bool, default=False)

## Training details
parser.add_argument('--max_epoch', type=int, default=200, help='Maximum number of epochs')
parser.add_argument('--temporal_stride', type=int, default=2, help='')

## Model definition
parser.add_argument('--model', type=str, default="models.SyncNetModelFBank", help='Model name')
parser.add_argument('--nOut', type=int, default=1024, help='Embedding size in the last FC layer')

## Learning rates
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every epoch')

## Joint training params
parser.add_argument('--alphaC', type=float, default=1.0, help='Sync weight')
parser.add_argument('--alphaI', type=float, default=1.0, help='Identity weight')
parser.add_argument('--alphaR', type=float, default=0, help='Regular weight')

## Load and save
parser.add_argument('--initial_model', type=str, default="", help='Initial model weights')
parser.add_argument('--save_path', type=str, default="./data/exp01", help='Path for model and logs')

## Training and test data
parser.add_argument('--train_list', type=str, default="data/train.txt", help='')
parser.add_argument('--verify_list', type=str, default="data/test.txt", help='')

## Speaker recognition test
parser.add_argument('--test_list', type=str, default="voxceleb/test_list.txt", help='Evaluation list')
parser.add_argument('--test_path', type=str, default="voxceleb/voxceleb1", help='Absolute path to the test set')

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')

args = parser.parse_args()

# ==================== MAKE DIRECTORIES ====================

model_save_path = args.save_path+"/model"
result_save_path = args.save_path+"/result"

if not (os.path.exists(model_save_path)):
	os.makedirs(model_save_path)

if not (os.path.exists(result_save_path)):
	os.makedirs(result_save_path)

# ==================== LOAD MODEL ====================

s = SyncNet(**vars(args))

# ==================== EVALUATE LIST ====================

it = 1

if args.goon:
	scorefile = open(result_save_path+"/scores.txt", "a+")
else:
	scorefile = open(result_save_path+"/scores.txt", 'w')

for items in vars(args):
	print(items, vars(args)[items])
	scorefile.write('%s %s\n'%(items, vars(args)[items]))
scorefile.flush()

# ==================== LOAD MODEL PARAMS ====================
if args.goon:
	modelfiles = glob.glob('%s/model0*.model'%model_save_path)
	modelfiles.sort()

	if len(modelfiles)>=1:
		s.loadParameters(modelfiles[-1])
		print("Model %s loaded from previous state!"%modelfiles[-1])
		it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:])+1
	elif args.initial_model != "":
		s.loadParameters(args.initial_model)
		print("Model %s loaded!"%args.initial_model)

	for ii in range(0, it-1):
		clr = s.updateLearningRate(args.lr_decay)
	print(clr[0])
# ==================== EVAL ====================

if args.eval == True:
	sc, lab = s.evaluateFromListSave(args.test_list, print_interval=100, test_path=args.test_path)
	result = tuneThresholdfromScore(sc, lab, [1, 0.1])
	print('EER %2.4f'%result[1])

	quit()

# ==================== LOAD DATA LIST ====================

print('Reading data ...')
start_time = time.time()
# trainLoader = DatasetLoader(args.train_list, nPerEpoch=args.nTrainPerEpoch, maxQueueSize=30, **vars(args))
# valLoader = DatasetLoader(args.verify_list, nPerEpoch=args.nTestPerEpoch, evalmode=True, **vars(args))
trainLoader = MyDataLoader(args.train_list, args.nBatchSize)
valLoader = MyDataLoader(args.verify_list, args.nBatchSize, not args.hard_eval)
end_time = time.time()
print('Reading done. Cost %.0f seconds.'%(end_time-start_time))

# ==================== CHECK SPK ====================

clr = s.updateLearningRate(1)

while (1):
	print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Start Iteration")

	loss, trainacc = s.train_network(trainLoader, evalmode=False, alpI=args.alphaI, alpC=args.alphaC)
	valloss, valacc = s.train_network(valLoader, evalmode=True)
	if args.disentangle:
		s.disentangle_step(trainLoader)
	print(time.strftime("%Y-%m-%d %H:%M:%S"), "%s: IT %d, LR %f, TACC %2.2f, TLOSS %f, VACC %2.2f, VLOSS %f\n"%(
		args.save_path, it, max(clr), trainacc, loss, valacc, valloss))
	scorefile.write(
		"IT %d, LR %f, TACC %2.2f, TLOSS %f, VACC %2.2f, VLOSS %f\n"%(it, max(clr), trainacc, loss, valacc, valloss))
	scorefile.flush()

	# ==================== SAVE MODEL ====================

	clr = s.updateLearningRate(args.lr_decay)

	print(time.strftime("%Y-%m-%d %H:%M:%S"), "Saving model %d"%it)
	s.saveParameters(model_save_path+"/model%09d.model"%it)

	if it>=args.max_epoch:
		quit()

	it += 1
	print("")

scorefile.close()
