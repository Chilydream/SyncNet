import os
import pickle
import numpy as np

from SyncNetDist import SyncNet
from dataLoader import loadWAV

S = SyncNet(model="models.SyncNetModelFBank", maxFrames=40, learning_rate=0.001, temporal_stride=2)
S.loadParameters("data/model000000100.model")
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


register_dict = dict()
for people in train_dict.keys():
	id_features = []
	for wavpath in train_dict[people]:
		filename = os.path.join(root_dir, wavpath)
		data_aud = loadWAV(filename, 40)
		out_content, out_id = S.__S__.forward_aud(data_aud.cuda())
		out_id = out_id.cpu().detach().numpy()[0]
		magnitude = np.linalg.norm(out_id)
		out_id = out_id/magnitude
		id_features.append(out_id)
	register_dict[people] = np.average(id_features, axis=0)
	print(register_dict[people])

