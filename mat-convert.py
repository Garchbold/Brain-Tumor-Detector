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
	
csvFileName = "brain_scan_data.csv"#csv that will hold each brain scan in one row
all_scan_array = []

for file in sorted(os.listdir('braintumors_1')): #iterate through all .mat files in braintumors_1 directory
	currMatFileNum = file.split('.')[0] #current file in braintumors_1 directory; splits 3064.mat into ['3064', 'mat']
	#print("Current mat file number: ", currMatFileNum) #e.g. "3064" from 3064.mat

	try:
		with h5py.File("braintumors_1/{}".format(file), 'r') as f: #open .mat file to save to new .csv file corresponding to that .mat file
			#print(saveToFile)
			#print(f['cjdata/image'].value)
			scan_array_2d = f['cjdata/image'].value
			print(scan_array_2d)
			scan_array_1d = []
			for row in scan_array_2d:#convert the current brain scan array into one long array
				scan_array_1d.append(row)

			all_scan_array.append(temp)#append the scan_array_1d as one entry in the array of all scans
	except:
		print("h5py failed!")
		raise

try:
   np.savetxt(saveToFile, f['cjdata/image'].value, delimiter=',') #For formatting: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.savetxt.html
except ValueError as e:
   print('Save failed! {}'.format(str(e)))
   raise SystemError
except AttributeError as e:
   print('Save failed! {}'.format(str(e)))
   raise SystemError
print("Wrote {0} to {1}".format(file, saveToFile))

