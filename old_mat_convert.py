import scipy.io
import numpy as np
import h5py
import pandas as pd
import os
import errno

def setup_data():

	#np.set_printoptions()

	#print(sorted(os.listdir('braintumors_1')))
		
	csvFileName = "brain_scan_data.csv"#csv that will hold each brain scan in one row
	all_scan_array = []
	count = 0
	skip_count = 0
	count_1 = 0
	count_2 = 0
	count_3 = 0

	files = sorted(os.listdir('braintumors_1'), key=lambda f: int(os.path.splitext(f)[0]))

	for file in files: #iterate through all .mat files in braintumors_1 directory
		#if count > 400:
		#	break
		currMatFileNum = file.split('.')[0] #current file in braintumors_1 directory; splits 3064.mat into ['3064', 'mat']
		print("Attemping to convert mat file number: "+ currMatFileNum) #e.g. "3064" from 3064.mat

		#labels: 1 for meningioma, 2 for glioma, 3 for pituitary tumor
		try:
			with h5py.File("braintumors_1/{}".format(file), 'r') as f: #open .mat file to save to new .csv file corresponding to that .mat file

				label = f['cjdata/label'].value[0][0] #label is of type numpy.float64

				scan_array_2d = f['cjdata/image'].value
				#print(scan_array_2d)
				if (len(scan_array_2d)) != 512:
					print("Failed to convert mat file number {}, incorrect format!".format(currMatFileNum))
					skip_count += 1
					pass
					#print(len(scan_array_2d))
					#print(file)
				else:
					scan_array_1d = []
					for row in scan_array_2d:#convert the current brain scan array into one long array
						scan_array_1d.extend(row)
					
					if label == 1: #meningioma
						scan_array_1d.append(1)
						count_1 += 1
					elif label == 2: #glioma
						scan_array_1d.append(2)
						count_2 += 1
					elif label == 3: #pituitary tumor
						scan_array_1d.append(3)
						count_3 += 1

					all_scan_array.append(scan_array_1d)#append the scan_array_1d as one entry in the array of all scans
					print("Success! Appending mat file number: "+ currMatFileNum)
		except:
			print("h5py failed!")
			raise
		count += 1

	print("Converted "+ str(count - skip_count) + " scans into {0} x {1} array.".format(len(all_scan_array), len(all_scan_array[0])))
	print("Skipped " + str(skip_count) + " images due to incorrect size (not 512x512).")
	print("Number of 1's:{0}\nNumber of 2's:{1}\nNumber of 3's:{2}".format(count_1, count_2, count_3))

	new_arr = np.array(all_scan_array, dtype=float)
	#print(new_arr)
	return new_arr