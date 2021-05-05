import pickle
import numpy as np

total_cnt = 220
pool = []
for i in range(total_cnt):
	pool.extend([i]*5)

epoch_num = 100000
repeat_num = 0
for i in range(epoch_num):
	batch = np.random.choice(pool, 30, False)
	repeat_num += len(batch)-len(set(batch))
print(repeat_num)
