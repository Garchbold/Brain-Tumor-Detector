from __future__ import print_function

import os
import csv
import sys
import shutil
from collections import namedtuple
from os import environ, listdir, makedirs
from os.path import dirname, exists, expanduser, isdir, join, splitext
import hashlib

from sklearn.utils import Bunch
from sklearn.utils import check_random_state

import numpy as np

from sklearn.externals.six.moves.urllib.request import urlretrieve

def load_digits(n_class=10, return_X_y=False):
	"""Load and return the digits dataset (classification).
	Each datapoint is a 8x8 image of a digit.
	=================   ==============
	Classes                         10
	Samples per class             ~180
	Samples total                 1797
	Dimensionality                  64
	Features             integers 0-16
	=================   ==============
	Read more in the :ref:`User Guide <datasets>`.
	Parameters
	----------
	n_class : integer, between 0 and 10, optional (default=10)
		The number of classes to return.
	return_X_y : boolean, default=False.
		If True, returns ``(data, target)`` instead of a Bunch object.
		See below for more information about the `data` and `target` object.
		.. versionadded:: 0.18
	Returns
	-------
	data : Bunch
		Dictionary-like object, the interesting attributes are:
		'data', the data to learn, 'images', the images corresponding
		to each sample, 'target', the classification labels for each
		sample, 'target_names', the meaning of the labels, and 'DESCR',
		the full description of the dataset.
	(data, target) : tuple if ``return_X_y`` is True
		.. versionadded:: 0.18
	This is a copy of the test set of the UCI ML hand-written digits datasets
	http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
	Examples
	--------
	To load the data and visualize the images::
		>>> from sklearn.datasets import load_digits
		>>> digits = load_digits()
		>>> print(digits.data.shape)
		(1797, 64)
		>>> import matplotlib.pyplot as plt #doctest: +SKIP
		>>> plt.gray() #doctest: +SKIP
		>>> plt.matshow(digits.images[0]) #doctest: +SKIP
		>>> plt.show() #doctest: +SKIP
	"""
	module_path = dirname(__file__)
	data = np.loadtxt('digits.csv.gz',
					  delimiter=',')

	print(data) #1797 rows, 65 columns
	#data: each row represents an image encoded in the following way: 
	#	65 columns of integers per row:
	#		the first 64 columns should be split into 8 arrays of 8 integers; the last column (65th) is TARGET number (what integer each image actually represents and this should be predicted correctly)

	#with open(join(module_path, 'descr', 'digits.rst')) as f:
	descr = "BLANK TEST"
	target = data[:, -1].astype(np.int) #ndarray type, contains int64's; for all rows, slice and take only the last column (65) which is the target data
	print(target[0])

	flat_data = data[:, :-1] #ndarray type; for all rows, slice and take all columns EXCEPT the last column (65; the target data)
	print("Flat_data: ", flat_data)
	
	images = flat_data.view() #same as flat_data....
	print("images: ", images)

	#images.shape at this point is equal to (1797, 64)
	images.shape = (-1, 8, 8) #now images.shape is (1797, 8, 8)
	print(images.shape)

	if n_class < 10: #filter out any classes that shouldn't be considered
		print("n_class is < 10")
		#print(target)
		idx = target < n_class
		#print("idx: ", idx)

		flat_data, target = flat_data[idx], target[idx]
		#print("flat_data: ", flat_data)

		#print("target: ", target)

		images = images[idx]
		#print("images[idx]: ", images)

	if return_X_y: #returns a tuple: (image data without target data, target data)
		return flat_data, target


#	data : Bunch
		#Dictionary-like object, the interesting attributes are:

		#'data' = the data to learn, 
		#'images' = the image data (8x8 of integers, an array of size 8, with 8 sub-arrays of size 8) corresponding to each image, 
		#'target' = the classification labels for each sample, i.e. what integers the images actually are supposed to be, 
		#'target_names' = the meaning of the labels; this is just an array of integers from 0 to 9 in this case,
		#'DESCR', the full description of the dataset.
	return Bunch(data=flat_data,
				 target=target,
				 target_names=np.arange(10),
				 images=images,
				 DESCR=descr)

a = load_digits()
print()
print("Returned value: ", a)



