import scipy.io
import numpy as np
import h5py
import pandas as pd
import os
import errno
import scipy.io as sio

def setup_data():

	#np.set_printoptions(threshold=np.inf)
	#print(sorted(os.listdir('braintumors_2')))
		
	all_scan_array = []
	array_no_targets = []
	count = 0

	files = sorted(os.listdir('braintumors_2'), key=lambda f: int(os.path.splitext(f)[0]))

	for file in files: #iterate through all .mat files in braintumors_1 directory
		if count > 1000:# and count < 2500:
			break
			#count+=1
			#pass
	
		#print(file)
		currMatFileNum = file.split('.')[0] #current file in braintumors_1 directory; splits 3064.mat into ['3064', 'mat']
		print("Converting mat file number: "+ currMatFileNum) #e.g. "3064" from 3064.mat

		#labels: 1 for meningioma, 2 for glioma, 3 for pituitary tumor
		try:

			f = sio.loadmat("braintumors_2/{}".format(file))
			#print()
			#print(type(f['im'][0][0][0][0][0][0][0][0]))
			'''
			[ [(array(
				[
					[(array([[1]], dtype=uint8), array(['100360'], dtype='<U6'), 
						array([], shape=(0, 0), dtype=uint8), array([[ARRAY OF NUMS FOR TUMOR BORDER]), 
						array([], shape=(0, 0), dtype=uint8))]],
      				dtype=[('label', 'O'), ('PID', 'O'), ('image', 'O'), ('tumorBorder', 'O'), ('tumorMask', 'O')]),
				)]
      		   ]
      		]
			'''

			#print()
			#with h5py.File("braintumors_2/{}".format(file), 'r') as f: #open .mat file to extract data

			label = f['im'][0][0][0][0][0][0][0][0] #label is of type numpy.uint8

			scan_array_2d = f['ans'] #ans is new 128x128 image
			#print(scan_array_2d)
			scan_array_1d = []
			scan_array_1d_no_targets = []

			if label == 1: #meningioma
				scan_array_1d.append(1)
			elif label == 2: #glioma
				scan_array_1d.append(2)
			elif label == 3: #pituitary tumor
				scan_array_1d.append(3)

			#print("1d before: ", scan_array_1d)

			for row in scan_array_2d:#convert the current brain scan array into one long array
				scan_array_1d_no_targets.extend(row)
			
			#print("1d after: ", scan_array_1d)

			#break
			#else: #label == 0, clean
			#	scan_array_1d.append(0)

			
			#if count%2 == 0:
			#	scan_array_1d.append(1) #target value for a brain that has a tumor
			#else:
			#	scan_array_1d.append(0) #target value for a brain that has a tumor
			#print("1d: ", len(scan_array_1d))
			#print()

			all_scan_array.extend(scan_array_1d)#append the scan_array_1d as one entry in the array of all scans
			array_no_targets.append(scan_array_1d_no_targets)
			#print(all_scan_array)
		except:
			print("h5py failed!")
			raise
		print("Appending mat file number: "+ currMatFileNum)
		count += 1

	#print("Converted "+ str(count) + " scans into {0} x {1} array.".format(len(all_scan_array), len(all_scan_array[0])))

	print(all_scan_array)
	#print(array_no_targets)
	#print(type(all_scan_array))

	new_np_array = np.array(all_scan_array)
	new_no_targets = np.array(array_no_targets)

	#print(temp)
	#print(type(temp))

	return (new_np_array, new_no_targets) #.astype(float)
	#return np.random.permutation(new_np_array)

#setup_data()






