import scipy.io
import numpy as np
import h5py
import pandas as pd
import os
import errno

newDir = "brain_csvs/"
if not os.path.exists(os.path.dirname(newDir)): #make new directory for all of the new csv files
	try:
		os.makedirs(os.path.dirname(newDir))
	except OSError as e: # race condition guard
		if e.errno != errno.EEXIST:
			raise

np.set_printoptions(threshold=np.inf)

#print(sorted(os.listdir('braintumors_1')))

for file in sorted(os.listdir('braintumors_1')): #iterate through all .mat files in braintumors_1 directory
	currMatFileNum = file.split('.')[0] #current file in braintumors_1 directory; splits 3064.mat into ['3064', 'mat']
	#print("Current mat file number: ", currMatFileNum) #e.g. "3064" from 3064.mat

	csvFileName = "{}.csv".format(currMatFileNum) #fileName of new .csv to write to
	#print("New csv filename: ", csvFileName) #e.g. "3064.csv" from 3064.mat

	try:
	   saveToFile = open("brain_csvs/{}".format(csvFileName), "w+") #actually open/make new .csv file to write to from each .mat file (file object)
	except OSError as e: # race condition guard
		if e.errno != errno.EEXIST:
			raise
		else:
			raise e
		#print(saveToFile)

	try:
		with h5py.File("braintumors_1/{}".format(file), 'r') as f: #open .mat file to save to new .csv file corresponding to that .mat file
			#print(type(f['cjdata/image'].value[0][0]))
			#print(type(int(f['cjdata/image'].value[0][0])))
			print(saveToFile)
			#print(f['cjdata/image'].value)
			try:
			   np.savetxt(saveToFile, f['cjdata/image'].value, delimiter=',') #For formatting: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.savetxt.html
			except ValueError as e:
			   print('Save failed! {}'.format(str(e)))
			   raise SystemError
			except AttributeError as e:
			   print('Save failed! {}'.format(str(e)))
			   raise SystemError
			print("Wrote {0} to {1}".format(file, saveToFile))
	except:
		print("h5py failed!")
		raise

