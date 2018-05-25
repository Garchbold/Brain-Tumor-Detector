import scipy.io
import numpy as np
import h5py
import pandas as pd

np.set_printoptions(threshold=np.inf)

g = open("a.csv", "w+")
with h5py.File('2300.mat', 'r') as f:
	for i in f:
		#if '__' not in i and 'readme' not in i:
		#print( (f.attrs.values()))
		#g.write( str(f['cjdata/image'].value), ) #print nparray
		np.savetxt("a.csv",f['cjdata/image'].value, delimiter=",")
	#df = pd.DataFrame(f['cjdata/image'].value)
	#df.to_csv("a.csv")

#data = scipy.io.loadmat("brainTumorDataPublic_22993064/2300.mat")

