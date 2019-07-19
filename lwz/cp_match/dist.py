import numpy as np
import sys
import pickle
import fnmatch
import os
import natsort

class best_match:
	def __init__(self):
		self.male_cp_base = []
		self.male = []
		self.female_cp_base = []
		self.female = []

		with open("cp_base_features_0.pkl", "rb") as f:
			d = pickle.load(f)
		for e in d:
			self.male_cp_base.append(e['feature'])
			self.male.append(e['label'])

		with open("cp_base_features_1.pkl", "rb") as f:
			d = pickle.load(f)
		for e in d:
			self.female_cp_base.append(e['feature'])
			self.female.append(e['label'])

	def euc_dist(self, x, y):
		return np.linalg.norm(x - y)

	def get_matched_for_male(self, cand):
		min_num = sys.maxsize
		idex = 0
		for i in range(len(self.female_cp_base)):
			d = self.euc_dist(cand, self.female_cp_base[i])
			if d < min_num:
				min_num = d
				idex = i
		return idex

	def get_matched_for_female(self, cand):
		min_num = sys.maxsize
		idex = 0
		for i in range(len(self.male_cp_base)):
			d = self.euc_dist(cand, self.male_cp_base[i])
			if d < min_num:
				min_num = d
				idex = i
		return idex