import scipy.io
import numpy as np
import h5py
import pandas as pd
import os
import errno

def setup_data():
	newDir = "brain_csvs/"
	if not os.path.exists(os.path.dirname(newDir)): #make new directory for all of the new csv files
		try:
			os.makedirs(os.path.dirname(newDir))
		except OSError as e: # race condition guard
			if e.errno != errno.EEXIST:
				raise

	#np.set_printoptions(threshold=np.inf)

	#print(sorted(os.listdir('braintumors_1')))
		
	csvFileName = "brain_scan_data.csv"#csv that will hold each brain scan in one row
	all_scan_array = []
	count = 0

	files = sorted(os.listdir('braintumors_1'))

	for file in files: #iterate through all .mat files in braintumors_1 directory
		if count > 10:
			break
		currMatFileNum = file.split('.')[0] #current file in braintumors_1 directory; splits 3064.mat into ['3064', 'mat']
		print("Converting mat file number: "+ currMatFileNum) #e.g. "3064" from 3064.mat

		try:
			with h5py.File("braintumors_1/{}".format(file), 'r') as f: #open .mat file to save to new .csv file corresponding to that .mat file
				#print(saveToFile)
				#print(f['cjdata/image'].value)
				scan_array_2d = f['cjdata/image'].value
				#print(scan_array_2d)
				scan_array_1d = []
				for row in scan_array_2d:#convert the current brain scan array into one long array
					scan_array_1d.extend(row)
				scan_array_1d.append(1) #target value for a brain that has a tumor
				#print("1d: ", len(scan_array_1d))

				all_scan_array.append(scan_array_1d)#append the scan_array_1d as one entry in the array of all scans
		except:
			print("h5py failed!")
			raise
		print("Appending mat file number: "+ currMatFileNum)
		count += 1

	try:
		pass
		#print("Building ", csvFileName)
		#np.savetxt(csvFileName, all_scan_array, fmt='%i', delimiter=',') #For formatting: https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.savetxt.html
	except ValueError as e:
		print('Save failed! {}'.format(str(e)))
		raise SystemError
	except AttributeError as e:
		print('Save failed! {}'.format(str(e)))
		raise SystemError
	print("Converted "+ str(count) + " scans into {0} x {1} array.".format(len(all_scan_array), len(all_scan_array[0])))

	return np.array(all_scan_array).astype(float)






