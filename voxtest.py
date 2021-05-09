import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

from SyncNetDist import SyncNet
from dataLoader import loadWAV


def cosine_distance(x, y):
	x_norm = np.linalg.norm(x)
	y_norm = np.linalg.norm(y)
	if x_norm*y_norm == 0:
		similiarity = 0
		print(x, y)
	else:
		similiarity = np.dot(x, y.T)/(x_norm*y_norm)
	dist = 1-similiarity
	return dist


def cosine_similarity(x, y):
	x_norm = np.linalg.norm(x)
	y_norm = np.linalg.norm(y)
	if x_norm*y_norm == 0:
		similiarity = 0
		print(x, y)
	else:
		similiarity = np.dot(x, y.T)/(x_norm*y_norm)
	return similiarity


S = SyncNet(model="models.SyncNetModelFBank", maxFrames=30, learning_rate=0.001, temporal_stride=2)
modelpath = "data/exp09.model"
print('Loading model from \'%s\''%modelpath)
S.loadParameters(modelpath)
S.eval()

root_dir = '/data2/Downloads/wav'
wavdata = pickle.load(open("./dataset_1000_pretrain_voice.pkl", 'rb'))
train_list = wavdata['train']
valid_list = wavdata['valid']
train_dict = dict()
valid_dict = dict()
for data_pair in train_list:
	wavpath, index = data_pair
	if index not in train_dict.keys():
		train_dict[index] = [wavpath]
	else:
		train_dict[index].append(wavpath)

for data_pair in valid_list:
	wavpath, index = data_pair
	if index not in valid_dict.keys():
		valid_dict[index] = [wavpath]
	else:
		valid_dict[index].append(wavpath)

if os.path.exists('npy/register_dict.npy'):
	register_dict = np.load('npy/register_dict.npy', allow_pickle=True).item()
else:
	register_dict = dict()
	for people in tqdm(train_dict.keys()):
		id_features = []
		for wavpath in train_dict[people]:
			filename = os.path.join(root_dir, wavpath)
			data_aud = loadWAV(filename, 160)
			out_content, out_id = S.__S__.forward_aud(data_aud.cuda())
			out_id = out_id.cpu().detach().numpy()[0]
			magnitude = np.linalg.norm(out_id, axis=0)
			out_id = out_id/magnitude
			out_id = np.average(out_id, axis=1)
			id_features.append(out_id)
		register_dict[people] = np.average(id_features, axis=0)
	np.save('npy/register_dict.npy', register_dict)


def recog_eval():
	correct_cnt = 0
	total_cnt = 0
	print('Start Recognization Eval:')
	for wavpath, people_valid in valid_list:
		filename = os.path.join(root_dir, wavpath)
		data_aud = loadWAV(filename, 160)
		out_content, out_id = S.__S__.forward_aud(data_aud.cuda())
		out_id = out_id.cpu().detach().numpy()[0]
		magnitude = np.linalg.norm(out_id, axis=0)
		out_id = out_id/magnitude
		out_id = np.average(out_id, axis=1)
		max_sim = None
		max_arg = None
		for people in register_dict.keys():
			sim = cosine_similarity(out_id, register_dict[people])
			if max_sim is None or max_sim<sim:
				max_sim = sim
				max_arg = people
		total_cnt += 1
		if max_arg == people_valid:
			correct_cnt += 1
		print('\rIter:%04d\tAcc:%.3f%%'%(total_cnt, correct_cnt*100/total_cnt), end='')


def veri_eval(threshold):
	if os.path.exists('npy/match_list.npy') and os.path.exists('npy/unmatch_list.npy'):
		match_list = np.load('npy/match_list.npy')
		unmatch_list = np.load('npy/unmatch_list.npy')
	else:
		match_list = []
		unmatch_list = []
		for wavpath, people_valid in tqdm(valid_list):
			filename = os.path.join(root_dir, wavpath)
			data_aud = loadWAV(filename, 160)
			out_content, out_id = S.__S__.forward_aud(data_aud.cuda())
			out_id = out_id.cpu().detach().numpy()[0]
			magnitude = np.linalg.norm(out_id, axis=0)
			out_id = out_id/magnitude
			out_id = np.average(out_id, axis=1)
			for people in register_dict.keys():
				sim = cosine_similarity(out_id, register_dict[people])
				if people == people_valid:
					match_list.append(sim)
				else:
					unmatch_list.append(sim)
		np.save('npy/match_list.npy', match_list)
		np.save('npy/unmatch_list.npy', unmatch_list)
	tn, fp, fn, tp = 0, 0, 0, 0
	for i in match_list:
		if i<threshold:
			fn += 1
		else:
			tp += 1
	for i in unmatch_list:
		if i<threshold:
			tn += 1
		else:
			fp += 1
	acc = tn*100/(tn+fp)
	tpr = tp*100/(tp+fn)
	fpr = fp*100/(tn+fp)
	fnr = fn*100/(tp+fn)
	# print('ACC: %.3f%%    TPR: %.3f%%    FPR: %.3f%%    FNR: %.3f%%'%(acc, tpr, fpr, fnr))
	return fpr, fnr


def bisearch(l, r):
	m = (l+r)/2
	fpr, fnr = veri_eval(m)
	print('Threshold: %.4f   ACC: %.3f%%    FPR: %.3f%%    FNR: %.3f%%'%(m, 100-fpr, fpr, fnr))
	if abs(fnr-fpr)<0.01:
		return
	if fnr>fpr:
		bisearch(l, m)
	else:
		bisearch(m, r)


# exp03: 等错误率 FPR==FNR==35.7% 阈值为 0.833
# exp04: 等错误率 FPR==FNR==35.2% 阈值为 0.816
# exp06: 等错误率 FPR==FNR==28.69% 阈值为 0.4597
# exp09: 等错误率 FPR==FNR==31.95% 阈值为 0.4486
# exp10: 等错误率 FPR==FNR==32.38% 阈值为 0.5698
# for i in range(0, 10):
# 	threshold = 0.8+(i/100)
# 	print('Threshold:%.3f'%threshold)
# 	veri_eval(threshold)
bisearch(0, 1)
