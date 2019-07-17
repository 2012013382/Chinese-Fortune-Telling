import numpy as np
import sys
def euc_dist(x, y):
	return np.linalg.norm(x - y)

#构造数据
cp_base = []
for i in range(2000):
	cp_base.append(np.random.beta(10,10,2096))

#计算最佳匹配
cand = np.random.beta(10,10,2096)
min_num = sys.maxsize
idex = 0
for i in range(len(cp_base)):
	d = euc_dist(cand, cp_base[i])
	if d < min_num:
		min_num = d
		idex = i

print(cp_base[idex])
print(cand)
print(euc_dist(cand, cp_base[idex]))
