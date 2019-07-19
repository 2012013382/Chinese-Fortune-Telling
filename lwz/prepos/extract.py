import fnmatch
import os
import natsort
import sys
from feature_extract import feature_extract
import pickle

os.chdir("../../")

# man
a_list = fnmatch.filter(os.listdir('photos'), '*.0.*.png')
a_list = natsort.natsorted(a_list)
features = []
for i in a_list:
	d = {}
	d['feature'] = feature_extract(os.path.join("photos", i))[0]
	d['label'] = i
	features.append(d)

with open("cp_base_features_0.pkl", "wb") as f:
	pickle.dump(features, f)

#woman
a_list = fnmatch.filter(os.listdir('photos'), '*.1.*.png')
a_list = natsort.natsorted(a_list)
features = []
for i in a_list:
	d = {}
	d['feature'] = feature_extract(os.path.join("photos", i))[0]
	d['label'] = i
	features.append(d)

with open("cp_base_features_1.pkl", "wb") as f:
	pickle.dump(features, f)


