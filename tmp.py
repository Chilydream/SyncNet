import pickle

wavdata = pickle.load(open("./dataset_1000_pretrain_voice.pkl", 'rb'))
train_list = wavdata['train']
valid_list = wavdata['valid']